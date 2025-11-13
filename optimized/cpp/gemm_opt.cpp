
/*
C++ optimized skeleton for GEMM. Students should fill in optimizations:
- OpenMP parallelization
- Blocking / tiling
- SIMD intrinsics (AVX2 / AVX512)
- NUMA-aware allocations

Usage: ./gemm_opt N num_threads
*/
/*
C++ optimized skeleton for GEMM.
Usage: ./gemm_opt N num_threads
*/

#include <iostream>
#include <string>
#include <stdexcept>
#include <omp.h>
#include "matrix.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " {matrix_dimension} {num_threads}\n";
        return 1;
    }

    std::size_t N;
    std::size_t T;
    try {
        N = std::stoull(argv[1]);
        T = std::stoull(argv[2]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: Invalid value for matrix_dimension or num_threads.\n";
        return 1;
    }

    omp_set_num_threads(static_cast<int>(T));

    Matrix<double, StorageLayout::RowMajor> matA(N, N, T), matC(N, N, T);
    Matrix<double, StorageLayout::ColMajor> matB(N, N, T);
    
    // matA.randomize(2, 0);
    // matB.randomize(2, 1);
    // matC.initializeZero();

    initialize(matA, matB, matC);

    matA.packMatrix();
    matB.packMatrix();

    double t0 = omp_get_wtime();
    multiply(matA, matB, matC);
    double t1 = omp_get_wtime();
    std::cout << "N=" << N << " T=" << T << " time=" << (t1 - t0) << " seconds\n";

    double s = matC.getChecksum();
    std::cout << "checksum=" << s << "\n";
    return 0;
}