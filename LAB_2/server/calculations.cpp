#include "calculations.h"
#include <iostream>

// TODO: make num threads configurable

void calculate_lengths(int num_vectors, int dim_vectors, const int *vectors, double* lengths) {
    omp_set_dynamic(0);
    omp_set_num_threads(16); // specify as you wish

    #pragma omp parallel for schedule(dynamic)
    for (int curr_idx = 0; curr_idx < (long long) num_vectors * dim_vectors; curr_idx += dim_vectors) {
        long long squared_sum = 0;
        for (int curr_dim = 0; curr_dim < dim_vectors; ++curr_dim) {
            squared_sum += (long long) vectors[curr_idx + curr_dim] * vectors[curr_idx + curr_dim];
        }

        lengths[curr_idx / dim_vectors] = sqrt((double) squared_sum);

//        #pragma omp critical
//        std::cout << "thread num = " << omp_get_thread_num() << ": current vector num = " << curr_idx << std::endl;
    }
}