
/*
C++ optimized skeleton for GEMM. Students should fill in optimizations:
- OpenMP parallelization
- Blocking / tiling
- SIMD intrinsics (AVX2 / AVX512)
- NUMA-aware allocations

Usage: ./gemm_opt N num_threads
*/
#include <bits/stdc++.h>
#include <omp.h>
#include "matrix.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " N num_threads\n";
        return 1;
    }
    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    omp_set_num_threads(T);

    Matrix<double, false> matA(N, N, T), matC(N, N, T);
    Matrix<double, true> matB(N, N, T);
    
    matA.randomize();
    matB.randomize();
    matC.initializeZero();

    double t0 = omp_get_wtime();
    multiply(matA, matB, matC);
    double t1 = omp_get_wtime();
    std::cout << "N="<<N<<" T="<<T<<" time="<<(t1-t0)<<" seconds\n";

    double s = matC.getChecksum();
    std::cout << "checksum=" << s << "\n";
    return 0;
}
