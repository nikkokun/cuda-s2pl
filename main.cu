#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <random>
#include <chrono>


#define readset_size 10
#define transaction_size 4000000

const int table_size = 1000000;
const int THREADS = 10000;
const int BLOCKS = 100;
const int STRIDE = transaction_size / THREADS;
typedef int transactions[readset_size];

using namespace thrust;
using namespace std;

__device__ int lock(int *lock) {

    return atomicCAS(lock, 0, 1);
}

__device__ void unlock(int *lock) {
    atomicExch(lock, 0);
}

__device__ void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

__device__ int partition (int arr[], int low, int high)
{
    int pivot = arr[high]; // pivot
    int i = (low - 1); // Index of smaller element

    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] < pivot)
        {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

__device__ void quickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
        at right place */
        int pi = partition(arr, low, high);

        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}


__global__ void myKernel(int *lock_table, int *table, transactions *d_transactions, int stride) {

    int tid =  blockIdx.x *blockDim.x + threadIdx.x;
    int start = tid * stride;
    int end = (start + stride) - 1;

//    sort phase
    for(int i = start; i < end; ++i) {
        quickSort(d_transactions[i], 0, readset_size-1);
    }
    //process transactions
    for(int i = start; i < end; ++i) {
        //growing phase
        int prev = -1;
        int abort_index[readset_size];
        memset(abort_index, -1, sizeof(abort_index));
        int is_abort = 0;


        for(int j = 0; j< readset_size; ++j) {
            int idx = d_transactions[i][j];
            if(idx != prev) {
                int is_locked = lock(&lock_table[idx]);
                if(is_locked == 0) {
                    abort_index[j] = idx;
                } else {
                    is_abort = 1;
                    break;
                }
            }
            prev = idx;
        }

        if(is_abort == 0) {
            //critical section
            prev = -1;
            for(int j = 0; j < readset_size; ++j) {
                int idx = d_transactions[i][j];
                if(idx != prev) {
                    table[idx] += 1;
                }
                prev = idx;
            }

            //shrinking phase
            prev = -1;
            for(int j = 0; j < readset_size; ++j) {
                int idx = d_transactions[i][j];
                if(idx != prev) {
                    unlock(&lock_table[idx]);
                }
                prev = idx;
            }
        } else {
            for(int j = 0; j < readset_size; ++j) {
                int idx = abort_index[j];
                if(idx != -1) {
                    unlock(&lock_table[idx]);
                }
                prev = idx;
            }
        }
    }
}

int main() {
    transactions *h_transactions;
    transactions *d_transactions;

    size_t dsize = transaction_size*readset_size*sizeof(int);

    int *lock_table;
    cudaMalloc((void**)&lock_table, table_size * sizeof(int));
    cudaMemset(lock_table, 0, table_size * sizeof(int));

    int *table;
    int *d_table;

    table = (int*)malloc(sizeof(int) * table_size);
    for(int i =0; i < table_size; ++i) {
        table[i] = 0;
    }


    cudaMalloc((void**)&d_table, sizeof(int) * table_size);
    cudaMemcpy(d_table, table, sizeof(int) * table_size, cudaMemcpyHostToDevice);


    //set memory for host arrays
    h_transactions = (transactions *)malloc(dsize);
    // populate h_ arrays
    memset(h_transactions, 0, dsize);

    random_device rd;
    mt19937 eng(rd());
    uniform_int_distribution<> distribution(0, table_size - 1);

    for(int i = 0; i < transaction_size; ++i) {
        for(int j =0; j < readset_size; ++j) {
            h_transactions[i][j] = distribution(eng);
        }
    }


    // Allocate memory on device
    cudaMalloc(&d_transactions, dsize);

    // Do memcopies to GPU
    cudaMemcpy(d_transactions, h_transactions, dsize, cudaMemcpyHostToDevice);

    float elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cout << "starting...\n";

    cudaEventRecord(start, 0);

    myKernel<<<THREADS/BLOCKS,BLOCKS>>>(lock_table, d_table, d_transactions, STRIDE);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    cout << "Finished after " << elapsed << " milliseconds";



    // Do memcopies back to host
    cudaMemcpy(h_transactions, d_transactions, dsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(table, d_table, sizeof(int) * table_size, cudaMemcpyDeviceToHost);


    cudaFree(d_transactions);
    cudaFree(d_table);


    return 0;
}