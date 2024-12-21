#include "calculations_gpu.cuh"

__global__ void calculate_lengths_kernel(size_t numVectors, size_t dimVectors,
                                         const int *vectors, double *lengths) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numVectors) {
		size_t squared_sum = 0;
		size_t vector_position = idx * dimVectors;
		for (int curr_dim = 0; curr_dim < dimVectors; ++curr_dim) {
			size_t element_idx = vector_position + curr_dim;
			squared_sum += (size_t)vectors[element_idx] * vectors[element_idx];
		}

		lengths[idx] = sqrt((double)squared_sum);
	}
}

void calculate_lengths_gpu(size_t numVectors, size_t dimVectors,
                           const int *vectors, double *lengths, int blockSize,
                           size_t maxBlockCount) {
	size_t numBlocks = (numVectors + blockSize - 1) / blockSize;

	size_t elementsProcessed = 0;
	size_t vectorsProcessed = 0;
	for (size_t blocks = 0; blocks < numBlocks; blocks += maxBlockCount) {
		size_t currNumBlocks = std::min(numBlocks - blocks, maxBlockCount);
		size_t currNumVectors =
		    std::min(currNumBlocks * blockSize, numVectors - vectorsProcessed);
		size_t currNumElements = currNumVectors * dimVectors;

		int *dVectors;
		double *dLengths;

		cudaMalloc((void **)&dVectors, currNumElements * sizeof(int));
		cudaMalloc((void **)&dLengths, currNumVectors * sizeof(double));

		cudaMemcpy(dVectors, vectors + elementsProcessed,
		           currNumElements * sizeof(int), cudaMemcpyHostToDevice);

		calculate_lengths_kernel<<<currNumBlocks, blockSize>>>(
		    currNumVectors, dimVectors, dVectors, dLengths);

		cudaMemcpy(lengths + vectorsProcessed, dLengths,
		           currNumVectors * sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(dVectors);
		cudaFree(dLengths);

		elementsProcessed += currNumElements;
		vectorsProcessed += currNumVectors;
	}
}