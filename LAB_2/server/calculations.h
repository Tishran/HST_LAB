#pragma once

#include <cmath>
#include <omp.h>

void calculate_lengths(int num_vectors, int dim_vectors, const int *vectors, double* lengths);