#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <device_atomic_functions.h>

using namespace std;

#define NUM_OPERATIONS 10
#define NUM_TRANSACTIONS 8388608
#define NUM_RECORDS 1000000
#define NUM_RUNS 10
typedef int transactions[NUM_OPERATIONS];

__device__ int lock(int *lock) {
    return atomicCAS(lock, 0, 1);
}

__device__ void unlock(int *lock) {
    atomicExch(lock, 0);
}

__device__ void insertion_sort(int *arr) {
    int i, key, j;
    for (i = 1; i < NUM_OPERATIONS; i++) {
        key = arr[i];
        j = i - 1;

        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

__global__ void transaction_worker(int *lock_table, int *table, transactions *d_transactions, int stride) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * stride;
    int end = (start + stride) - 1;

//        sort phase
    for (int i = start; i < end; ++i) {
        insertion_sort(d_transactions[i]);
    }

//    process transactions
    for (int i = start; i < end; ++i) {
        int locked_indices[NUM_OPERATIONS];
        memset(locked_indices, -1, sizeof(locked_indices));
        int is_abort = 0;

        //growing phase
        int prev = -1;
        for (int j = 0; j < NUM_OPERATIONS; ++j) {
            int idx = d_transactions[i][j];
            if (idx != prev) {
                int is_locked = lock(&lock_table[idx]);
                if (is_locked == 1) {
                    is_abort = 1;
                    break;
                } else {
                    locked_indices[j] = idx;
                }
            }
            prev = idx;
        }

        //critical section
        if (is_abort == 0) {
            for (int j = 0; j < NUM_OPERATIONS; ++j) {
                int idx = locked_indices[j];
                if (idx != -1) {
                    ++table[idx];
                }
            }
        }

        //shrinking phase
        for (int j = 0; j < NUM_OPERATIONS; ++j) {
            int idx = locked_indices[j];
            if (idx != -1) {
                unlock(&lock_table[idx]);
            }
        }
    }
}

int main() {
    string OUTPUT_FP =
            "/home/nicoroble/cuda-s2pl/data/cuda-transactions_readsetsize-" + to_string(NUM_OPERATIONS) + "_transactions-" +
            to_string(NUM_TRANSACTIONS) + "_tablesize-" + to_string(NUM_RECORDS) + ".tsv";

    const vector<int> NUM_THREAD_VEC = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    const vector<int> BLOCK_VEC = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

    for (int x = 0; x < NUM_THREAD_VEC.size(); ++x) {
        int NUM_THREADS = NUM_THREAD_VEC[x];
        int BLOCKS = BLOCK_VEC[x];
        const int STRIDE = NUM_TRANSACTIONS / NUM_THREADS;
        vector<float> runtimes;
        for (int n = 0; n < NUM_RUNS; ++n) {
            transactions *h_transactions;
            transactions *d_transactions;
            int *lock_table;
            int *d_lock_table;
            int *table;
            int *d_table;

            size_t dsize = NUM_TRANSACTIONS * NUM_OPERATIONS * sizeof(int);
            cudaMalloc((void **) &lock_table, NUM_RECORDS * sizeof(int));
            cudaMalloc((void **) &d_lock_table, NUM_RECORDS * sizeof(int));
            cudaMemset(lock_table, 0, NUM_RECORDS * sizeof(int));
            cudaMemcpy(d_lock_table, lock_table, sizeof(int) * NUM_RECORDS, cudaMemcpyHostToDevice);

            table = (int *) malloc(sizeof(int) * NUM_RECORDS);
            for (int i = 0; i < NUM_RECORDS; ++i) {
                table[i] = 0;
            }
            cudaMalloc((void **) &d_table, sizeof(int) * NUM_RECORDS);
            cudaMemcpy(d_table, table, sizeof(int) * NUM_RECORDS, cudaMemcpyHostToDevice);

            //set memory for host arrays
            h_transactions = (transactions *) malloc(dsize);
            // populate h_ arrays
            memset(h_transactions, 0, dsize);

            random_device rd;
            mt19937 eng(rd());
            uniform_int_distribution<int> distribution(0, NUM_RECORDS - 1);

            for (int i = 0; i < NUM_TRANSACTIONS; ++i) {
                for (int j = 0; j < NUM_OPERATIONS; ++j) {
                    h_transactions[i][j] = distribution(eng);
                }
            }
            // Allocate memory on device
            cudaMalloc(&d_transactions, dsize);
            // Do memcopies to GPU
            cudaMemcpy(d_transactions, h_transactions, dsize, cudaMemcpyHostToDevice);

            float elapsed = 0;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, 0);
            transaction_worker<<<BLOCKS, NUM_THREADS / BLOCKS >>>(d_lock_table, d_table, d_transactions, STRIDE);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);

            runtimes.push_back(elapsed);

            free(h_transactions);
            cudaFree(d_transactions);
            cudaFree(d_table);
            cudaFree(lock_table);
            cudaDeviceSynchronize();
        }

        ofstream outfile;
        outfile.open(OUTPUT_FP, ios_base::app);
        float average_runtime = accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
        outfile << NUM_THREADS << "\t" << average_runtime << "\t" << BLOCKS << "\n";
        outfile.close();
    }

    return 0;
}