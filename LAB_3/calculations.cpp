#include "calculations.h"

void calculate_lengths(int num_vectors, int dim_vectors, const int *vectors, double* lengths) {
    for (int curr_idx = 0; curr_idx < (long long) num_vectors * dim_vectors; curr_idx += dim_vectors) {
        long long squared_sum = 0;
        for (int curr_dim = 0; curr_dim < dim_vectors; ++curr_dim) {
            squared_sum += (long long) vectors[curr_idx + curr_dim] * vectors[curr_idx + curr_dim];
        }

        lengths[curr_idx / dim_vectors] = sqrt((double) squared_sum);
    }
}