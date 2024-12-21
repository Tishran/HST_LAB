#pragma once

#include <cuda_runtime.h>

#include <cmath>

__global__ void calculate_lengths_kernel(size_t numVectors, size_t dimVectors,
                                         const int *vectors, double *lengths);

void calculate_lengths_gpu(size_t numVectors, size_t dimVectors,
                           const int *vectors, double *lengths, int blockSize,
                           size_t maxBlockCount);