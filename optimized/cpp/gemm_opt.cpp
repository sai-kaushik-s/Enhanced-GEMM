
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

using namespace std;

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads\n";
        return 1;
    }
    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    omp_set_num_threads(T);

    Matrix<double> matA(N, N, T), matB(N, N, T, true), matC(N, N, T);
    // reproducible RNG
    matA.randomize();
    matB.randomize();

    double t0 = omp_get_wtime();
    matC = multiply(matA, matB, 32);
    double t1 = omp_get_wtime();
    cout << "N="<<N<<" T="<<T<<" time="<<(t1-t0)<<" seconds\n";

    // matC.print();

    // simple checksum to validate
    double s = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            s += matC(i, j);
        }
    }
    cout << "checksum=" << s << "\n";
    return 0;
}
