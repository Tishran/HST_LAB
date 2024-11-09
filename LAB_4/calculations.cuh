#pragma once

#include <cmath>
#include <cuda_runtime.h>

__global__ void calculate_lengths_kernel(size_t numVectors, size_t dimVectors, const int *vectors, double *lengths);

void calculate_lengths(size_t numVectors, size_t dimVectors, const int *vectors, double *lengths, int blockSize,
                       size_t maxBlockCount);