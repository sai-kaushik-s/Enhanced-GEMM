#include <iostream>
#include <iomanip>
#include <random>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <unistd.h>
#ifdef __AVX2__
    #include <immintrin.h>
#endif

template<typename T, StorageLayout Layout>
Matrix<T, Layout>::Matrix(std::size_t rows, std::size_t cols, std::size_t numCores) : 
    rowSize(rows), 
    colSize(cols), 
    packSize(32 / sizeof(T)), 
    numCores(numCores)
{
    flatData.resize(rowSize * colSize);

    T* data = flatData.data();
    std::size_t totalSize = rowSize * colSize;
    #pragma omp parallel for num_threads(numCores) schedule(static)
    for (std::size_t i = 0; i < totalSize; ++i) {
        data[i] = T(0);
    }

    long l2Size = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l2Size <= 0) l2Size = 512 * 1024;

    packedMC = 128; 
    packedKC = 256;
    packedNC = 48;

    std::size_t neededL2 = (packedMC * packedKC + packedNC * packedKC) * sizeof(T);

    if (neededL2 > (std::size_t)(l2Size * 0.8)) {
        double scale = (double)(l2Size * 0.8) / neededL2;
        packedMC = std::max(std::size_t(64), (std::size_t)(packedMC * scale));
        packedKC = std::max(std::size_t(128), (std::size_t)(packedKC * scale));
        
        if (scale < 0.5) packedNC = 24; 
    }

    packedNC = (packedNC / 6) * 6;
    if (packedNC < 6) packedNC = 6;
}

template<typename T, StorageLayout Layout>
Matrix<T, Layout>::Matrix(const Matrix<T, Layout>& other) : 
    rowSize(other.rowSize), 
    colSize(other.colSize), 
    packSize(other.packSize), 
    flatData(other.flatData),
    numCores(other.numCores),
    isPacked(other.isPacked),
    packedMC(other.packedMC),
    packedKC(other.packedKC),
    packedNC(other.packedNC),
    packedData(other.packedData)
{
}

template<typename T, StorageLayout Layout>
Matrix<T, Layout>& Matrix<T, Layout>::operator=(const Matrix<T, Layout>& other) {
    if (this == &other) { return *this; }

    flatData = other.flatData; 
    
    rowSize = other.rowSize;
    colSize = other.colSize;
    packSize = other.packSize;
    numCores = other.numCores;
    isPacked = other.isPacked;
    packedMC = other.packedMC;
    packedKC = other.packedKC;
    packedNC = other.packedNC;
    packedData = other.packedData;

    return *this;
}

template<typename T, StorageLayout Layout>
void Matrix<T, Layout>::randomize(int seq_stride, int seq_offset) {
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j) {
            for (int k = 0; k < seq_offset; ++k) {
                dist(rng);
            }
            (*this)(i, j) = static_cast<T>(dist(rng));
            for (int k = seq_offset + 1; k < seq_stride; ++k) {
                dist(rng);
            }
        }
    }
}

template<typename T, StorageLayout Layout>
void Matrix<T, Layout>::write(std::ostream& os) const {
    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j)
            os << std::setw(8) << (*this)(i, j) << " ";
        os << '\n';
    }
}

template<typename T, StorageLayout Layout>
void Matrix<T, Layout>::print() const {
    write(std::cout);
}

template<typename T, StorageLayout Layout>
bool Matrix<T, Layout>::isIdentity() const {
    if (rowSize != colSize) { return false; }

    bool diagonal = true;
    #pragma omp parallel for num_threads(getNumCores()) reduction(&&:diagonal) schedule(static)
    for (std::size_t i = 0; i < rowSize; ++i) {
        if ((*this)(i, i) != T(1)) {
            diagonal = false;
        }
    }
    
    if (!diagonal) { return false; }

    bool offDiagonal = true;
    #pragma omp parallel for num_threads(getNumCores()) reduction(&&:offDiagonal) collapse(2) schedule(static)
    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j) {
            if (i != j && (*this)(i, j) != T(0)) {
                offDiagonal = false;
            }
        }
    }

    return offDiagonal;
}

template<typename T, StorageLayout Layout>
bool Matrix<T, Layout>::isZero() const {
    bool allZero = true;
    #pragma omp parallel for num_threads(getNumCores()) reduction(&&:allZero) collapse(2) schedule(static)
    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j) {
            if ((*this)(i, j) != T(0)) {
                allZero = false;
            }
        }
    }
    return allZero;
}

template<typename T, StorageLayout Layout>
T Matrix<T, Layout>::getChecksum() const {
    T checksum = T(0);
    #pragma omp parallel for num_threads(getNumCores()) reduction(+:checksum) collapse(2) schedule(static)
    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j) {
            checksum += (*this)(i, j);
        }
    }
    return checksum;
}

template<typename T, StorageLayout Layout>
void Matrix<T, Layout>::initializeZero() {
    #pragma omp parallel for num_threads(getNumCores()) collapse(2) schedule(static)
    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j) {
            (*this)(i, j) = T(0);
        }
    }
}

template<typename T, StorageLayout Layout>
std::size_t Matrix<T, Layout>::getRowSize() const { return rowSize; }

template<typename T, StorageLayout Layout>
std::size_t Matrix<T, Layout>::getColSize() const { return colSize; }

template<typename T, StorageLayout Layout>
std::size_t Matrix<T, Layout>::getPackSize() const { return packSize; }

template<typename T, StorageLayout Layout>
std::size_t Matrix<T, Layout>::getNumCores() const { return numCores; }

template<typename T, StorageLayout Layout>
bool Matrix<T, Layout>::getIsPacked() const { return isPacked; }

template<typename T, StorageLayout Layout>
std::size_t Matrix<T, Layout>::getPackedMC() const { return packedMC; }

template<typename T, StorageLayout Layout>
std::size_t Matrix<T, Layout>::getPackedKC() const { return packedKC; }

template<typename T, StorageLayout Layout>
std::size_t Matrix<T, Layout>::getPackedNC() const { return packedNC; }

template<typename T, StorageLayout Layout>
const T* Matrix<T, Layout>::getPackedData() const { return packedData.data(); }

template<typename T, StorageLayout Layout>
T* Matrix<T, Layout>::getData() { return flatData.data(); }

template<typename T, StorageLayout Layout>
const T* Matrix<T, Layout>::getData() const { return flatData.data(); }

template<typename T, StorageLayout Layout>
T& Matrix<T, Layout>::operator()(std::size_t i, std::size_t j) {
    if constexpr (IsTransposed) { 
        return flatData[j * rowSize + i]; 
    } else { 
        return flatData[i * colSize + j]; 
    }
}

template<typename T, StorageLayout Layout>
const T& Matrix<T, Layout>::operator()(std::size_t i, std::size_t j) const {
    if constexpr (IsTransposed) { 
        return flatData[j * rowSize + i]; 
    } else { 
        return flatData[i * colSize + j]; 
    }
}

template<typename T, StorageLayout Layout>
Matrix<T, (Layout == StorageLayout::RowMajor ? StorageLayout::ColMajor : StorageLayout::RowMajor)> Matrix<T, Layout>::transpose() const {
    Matrix<T, (Layout == StorageLayout::RowMajor ? StorageLayout::ColMajor : StorageLayout::RowMajor)> result(getColSize(), getRowSize(), getNumCores());
    #pragma omp parallel for num_threads(getNumCores()) collapse(2) schedule(static)
    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

template<typename T, StorageLayout Layout>
void Matrix<T, Layout>::packMatrix() {
    isPacked = true;

    if constexpr (IsTransposed) {
        std::size_t nBlocksK = (rowSize + packedKC - 1) / packedKC;
        std::size_t nBlocksJ = (colSize + packedNC - 1) / packedNC;
        packedData.resize(nBlocksK * nBlocksJ * packedKC * packedNC);

        #pragma omp parallel for num_threads(numCores) collapse(2) schedule(static)
        for (std::size_t j = 0; j < colSize; j += packedNC) {
            for (std::size_t k = 0; k < rowSize; k += packedKC) {
                std::size_t idx = (j / packedNC) * nBlocksK + (k / packedKC);
                T* ptr = &packedData[idx * packedKC * packedNC];

                for (std::size_t jj = j; jj < std::min(j + packedNC, colSize); ++jj) {
                    for (std::size_t kk = k; kk < std::min(k + packedKC, rowSize); ++kk) {
                        ptr[(jj - j) * packedKC + (kk - k)] = (*this)(kk, jj);
                    }
                }
            }
        }
    } else {
        std::size_t nBlocksI = (rowSize + packedMC - 1) / packedMC;
        std::size_t nBlocksK = (colSize + packedKC - 1) / packedKC;
        packedData.resize(nBlocksI * nBlocksK * packedMC * packedKC);

        #pragma omp parallel for num_threads(numCores) collapse(2) schedule(static)
        for (std::size_t i = 0; i < rowSize; i += packedMC) {
            for (std::size_t k = 0; k < colSize; k += packedKC) {
                std::size_t idx = (i / packedMC) * nBlocksK + (k / packedKC);
                T* ptr = &packedData[idx * packedMC * packedKC];

                for (std::size_t ii = i; ii < std::min(i + packedMC, rowSize); ++ii) {
                    for (std::size_t kk = k; kk < std::min(k + packedKC, colSize); ++kk) {
                        ptr[(ii - i) * packedKC + (kk - k)] = (*this)(ii, kk);
                    }
                }
            }
        }
    }
}

static inline double hsum256_pd(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
    return _mm_cvtsd_f64(s);
}

static inline double hsum512_pd(__m512d v){
    __m256d lo = _mm512_castpd512_pd256(v);
    __m256d hi = _mm512_extractf64x4_pd(v, 1);
    return hsum256_pd(_mm256_add_pd(lo, hi));
}

template<typename T, StorageLayout L_A, StorageLayout L_B, StorageLayout L_C>
void multiply(const Matrix<T, L_A>& A, const Matrix<T, L_B>& B, Matrix<T, L_C>& C) {
    if constexpr (L_A == StorageLayout::RowMajor && L_B == StorageLayout::ColMajor && L_C == StorageLayout::RowMajor) {
        if (A.getColSize() != B.getRowSize()) {
             throw std::invalid_argument("Dimension mismatch in multiply");
        }

        std::size_t aRows = A.getRowSize();
        std::size_t bCols = B.getColSize();
        std::size_t aCols = A.getColSize();

        std::size_t MC = A.getPackedMC(); 
        std::size_t KC = A.getPackedKC();
        std::size_t NC = A.getPackedNC();

        bool offlineA = A.getIsPacked();
        bool offlineB = B.getIsPacked();

        std::size_t nBBlocksK = (B.getRowSize() + KC - 1) / KC;
        std::size_t nABlocksK = (A.getColSize() + KC - 1) / KC;

        #pragma omp parallel num_threads(A.getNumCores())
        {
            std::vector<T> localPA(offlineA ? 0 : MC * KC);
            std::vector<T> localPB(offlineB ? 0 : NC * KC);

            #pragma omp for collapse(2) schedule(static)
            for (std::size_t i = 0; i < aRows; i += MC) {
                for (std::size_t j = 0; j < bCols; j += NC) {
                    for (std::size_t k = 0; k < aCols; k += KC) {

                        const T* ptrPA;
                        const T* ptrPB;
                        
                        const std::size_t iiEnd = std::min(i + MC, aRows);
                        const std::size_t jjEnd = std::min(j + NC, bCols);
                        const std::size_t kkEnd = std::min(k + KC, aCols);

                        if (offlineA) {
                            std::size_t idx = (i / MC) * nABlocksK + (k / KC);
                            ptrPA = A.getPackedData() + (idx * MC * KC);
                        } else {
                            for (std::size_t ii = i; ii < iiEnd; ++ii) {
                                for (std::size_t kk = k; kk < kkEnd; ++kk) {
                                    localPA[(ii - i) * KC + (kk - k)] = A(ii, kk);
                                }
                            }
                            ptrPA = localPA.data();
                        }

                        if (offlineB) {
                            std::size_t idx = (j / NC) * nBBlocksK + (k / KC);
                            ptrPB = B.getPackedData() + (idx * KC * NC);
                        } else {
                            for (std::size_t jj = j; jj < jjEnd; ++jj) {
                                for (std::size_t kk = k; kk < kkEnd; ++kk) {
                                    localPB[(jj - j) * KC + (kk - k)] = B(kk, jj);
                                }
                            }
                            ptrPB = localPB.data();
                        }

                        #ifdef __AVX2__
                            const std::size_t W = 4;
                            const std::size_t PF_DIST = 64;

                            std::size_t kk_vec_end = k + ((kkEnd - k) / W) * W;

                            for (std::size_t ii = i; ii < iiEnd; ++ii) {
                                std::size_t jj = j;
                                const std::size_t jjUnrolledEnd = j + ((jjEnd - j) / 6) * 6;
                                for (; jj < jjUnrolledEnd; jj += 6) {
                                    T sum0 = C(ii, jj + 0); T sum1 = C(ii, jj + 1); T sum2 = C(ii, jj + 2);
                                    T sum3 = C(ii, jj + 3); T sum4 = C(ii, jj + 4); T sum5 = C(ii, jj + 5);
                                    __m256d acc0 = _mm256_setzero_pd(); __m256d acc1 = _mm256_setzero_pd();
                                    __m256d acc2 = _mm256_setzero_pd(); __m256d acc3 = _mm256_setzero_pd();
                                    __m256d acc4 = _mm256_setzero_pd(); __m256d acc5 = _mm256_setzero_pd();
                                    for (std::size_t kk = k; kk < kk_vec_end; kk += W) {
                                        _mm_prefetch((const char*)&ptrPA[(ii - i) * KC + (kk - k + PF_DIST)], _MM_HINT_T0);
                                        _mm_prefetch((const char*)&ptrPB[(jj - j + 0) * KC + (kk - k + PF_DIST)], _MM_HINT_T0);
                                        _mm_prefetch((const char*)&ptrPB[(jj - j + 1) * KC + (kk - k + PF_DIST)], _MM_HINT_T0);

                                        __m256d a = _mm256_loadu_pd(&ptrPA[(ii - i) * KC + (kk - k)]);
                                        __m256d b0 = _mm256_loadu_pd(&ptrPB[(jj - j + 0) * KC + (kk - k)]); __m256d b1 = _mm256_loadu_pd(&ptrPB[(jj - j + 1) * KC + (kk - k)]);
                                        __m256d b2 = _mm256_loadu_pd(&ptrPB[(jj - j + 2) * KC + (kk - k)]); __m256d b3 = _mm256_loadu_pd(&ptrPB[(jj - j + 3) * KC + (kk - k)]);
                                        __m256d b4 = _mm256_loadu_pd(&ptrPB[(jj - j + 4) * KC + (kk - k)]); __m256d b5 = _mm256_loadu_pd(&ptrPB[(jj - j + 5) * KC + (kk - k)]);

                                        acc0 = _mm256_fmadd_pd(a, b0, acc0); acc1 = _mm256_fmadd_pd(a, b1, acc1); acc2 = _mm256_fmadd_pd(a, b2, acc2);
                                        acc3 = _mm256_fmadd_pd(a, b3, acc3); acc4 = _mm256_fmadd_pd(a, b4, acc4); acc5 = _mm256_fmadd_pd(a, b5, acc5);
                                    }
                                    sum0 += hsum256_pd(acc0); sum1 += hsum256_pd(acc1); sum2 += hsum256_pd(acc2);
                                    sum3 += hsum256_pd(acc3); sum4 += hsum256_pd(acc4); sum5 += hsum256_pd(acc5);
                                    for (std::size_t kk = kk_vec_end; kk < kkEnd; ++kk) {
                                        double a_val = ptrPA[(ii - i) * KC + (kk - k)];
                                        sum0 += a_val * ptrPB[(jj - j + 0) * KC + (kk - k)]; sum1 += a_val * ptrPB[(jj - j + 1) * KC + (kk - k)];
                                        sum2 += a_val * ptrPB[(jj - j + 2) * KC + (kk - k)]; sum3 += a_val * ptrPB[(jj - j + 3) * KC + (kk - k)];
                                        sum4 += a_val * ptrPB[(jj - j + 4) * KC + (kk - k)]; sum5 += a_val * ptrPB[(jj - j + 5) * KC + (kk - k)];
                                    }

                                    C(ii, jj + 0) = sum0;  C(ii, jj + 1) = sum1;  C(ii, jj + 2) = sum2;
                                    C(ii, jj + 3) = sum3;  C(ii, jj + 4) = sum4;  C(ii, jj + 5) = sum5;
                                }
                                for (; jj < jjEnd; ++jj) {
                                    T sum = C(ii, jj);
                                    for (std::size_t kk = k; kk < kkEnd; ++kk) {
                                        sum = std::fma(ptrPA[(ii - i) * KC + (kk - k)],
                                                    ptrPB[(jj - j) * KC + (kk - k)], sum);
                                    }
                                    C(ii, jj) = sum;
                                }
                            }
                        #elif defined(__AVX512F__)
                            const std::size_t W = 8;           
                            const std::size_t PF_DIST = 64;    

                            std::size_t kk_vec_end = k + ((kkEnd - k) / W) * W;

                            for (std::size_t ii = i; ii < iiEnd; ++ii) {
                                std::size_t jj = j;

                                const std::size_t jjUnrolledEnd6 = j + ((jjEnd - j) / 6) * 6;
                                for (; jj < jjUnrolledEnd6; jj += 6) {
                                    T sum0 = C(ii, jj + 0); T sum1 = C(ii, jj + 1); T sum2 = C(ii, jj + 2);
                                    T sum3 = C(ii, jj + 3); T sum4 = C(ii, jj + 4); T sum5 = C(ii, jj + 5);

                                    __m512d acc0 = _mm512_setzero_pd(); __m512d acc1 = _mm512_setzero_pd(); __m512d acc2 = _mm512_setzero_pd();
                                    __m512d acc3 = _mm512_setzero_pd(); __m512d acc4 = _mm512_setzero_pd(); __m512d acc5 = _mm512_setzero_pd();

                                    for (std::size_t kk = k; kk < kk_vec_end; kk += W) {
                                        _mm_prefetch((const char*)&ptrPA[(ii - i) * KC + (kk - k + PF_DIST)], _MM_HINT_T0);
                                        _mm_prefetch((const char*)&ptrPB[(jj - j + 0) * KC + (kk - k + PF_DIST)], _MM_HINT_T0);
                                        _mm_prefetch((const char*)&ptrPB[(jj - j + 1) * KC + (kk - k + PF_DIST)], _MM_HINT_T0);

                                        __m512d a  = _mm512_loadu_pd(&ptrPA[(ii - i) * KC + (kk - k)]);
                                        __m512d b0 = _mm512_loadu_pd(&ptrPB[(jj - j + 0) * KC + (kk - k)]); __m512d b1 = _mm512_loadu_pd(&ptrPB[(jj - j + 1) * KC + (kk - k)]);
                                        __m512d b2 = _mm512_loadu_pd(&ptrPB[(jj - j + 2) * KC + (kk - k)]); __m512d b3 = _mm512_loadu_pd(&ptrPB[(jj - j + 3) * KC + (kk - k)]);
                                        __m512d b4 = _mm512_loadu_pd(&ptrPB[(jj - j + 4) * KC + (kk - k)]); __m512d b5 = _mm512_loadu_pd(&ptrPB[(jj - j + 5) * KC + (kk - k)]);

                                        acc0 = _mm512_fmadd_pd(a, b0, acc0); acc1 = _mm512_fmadd_pd(a, b1, acc1); acc2 = _mm512_fmadd_pd(a, b2, acc2);
                                        acc3 = _mm512_fmadd_pd(a, b3, acc3); acc4 = _mm512_fmadd_pd(a, b4, acc4); acc5 = _mm512_fmadd_pd(a, b5, acc5);
                                    }

                                    {
                                        const int rem = static_cast<int>(kkEnd - kk_vec_end);
                                        if (rem > 0) {
                                            const __mmask8 m = (1u << rem) - 1u;
                                            const std::size_t base = (ii - i) * KC + (kk_vec_end - k);
                                            const std::size_t baseB0 = (jj - j + 0) * KC + (kk_vec_end - k);

                                            __m512d a  = _mm512_maskz_loadu_pd(m, &ptrPA[base]);

                                            __m512d b0 = _mm512_maskz_loadu_pd(m, &ptrPB[baseB0 + 0 * KC]); __m512d b1 = _mm512_maskz_loadu_pd(m, &ptrPB[baseB0 + 1 * KC]);
                                            __m512d b2 = _mm512_maskz_loadu_pd(m, &ptrPB[baseB0 + 2 * KC]); __m512d b3 = _mm512_maskz_loadu_pd(m, &ptrPB[baseB0 + 3 * KC]);
                                            __m512d b4 = _mm512_maskz_loadu_pd(m, &ptrPB[baseB0 + 4 * KC]); __m512d b5 = _mm512_maskz_loadu_pd(m, &ptrPB[baseB0 + 5 * KC]);

                                            acc0 = _mm512_fmadd_pd(a, b0, acc0); acc1 = _mm512_fmadd_pd(a, b1, acc1); acc2 = _mm512_fmadd_pd(a, b2, acc2);
                                            acc3 = _mm512_fmadd_pd(a, b3, acc3); acc4 = _mm512_fmadd_pd(a, b4, acc4); acc5 = _mm512_fmadd_pd(a, b5, acc5);
                                        }
                                    }

                                    sum0 += hsum512_pd(acc0); sum1 += hsum512_pd(acc1); sum2 += hsum512_pd(acc2);
                                    sum3 += hsum512_pd(acc3); sum4 += hsum512_pd(acc4); sum5 += hsum512_pd(acc5);

                                    C(ii, jj + 0) = sum0;  C(ii, jj + 1) = sum1;  C(ii, jj + 2) = sum2;
                                    C(ii, jj + 3) = sum3;  C(ii, jj + 4) = sum4;  C(ii, jj + 5) = sum5;
                                }
                                for (; jj < jjEnd; ++jj) {
                                    T sum = C(ii, jj);
                                    for (std::size_t kk = k; kk < kkEnd; ++kk) {
                                        sum = std::fma(ptrPA[(ii - i) * KC + (kk - k)],
                                                    ptrPB[(jj - j) * KC + (kk - k)], sum);
                                    }
                                    C(ii, jj) = sum;
                                }
                            }
                        #else
                            for (std::size_t ii = i; ii < iiEnd; ++ii) {
                                std::size_t jj = j;
                                for (; jj < jjEnd; ++jj) {
                                     T sum = C(ii, jj);
                                     for (std::size_t kk = k; kk < kkEnd; ++kk) {
                                        sum = std::fma(ptrPA[(ii - i) * KC + (kk - k)],
                                                    ptrPB[(jj - j) * KC + (kk - k)], sum);
                                     }
                                     C(ii, jj) = sum;
                                }
                            }
                        #endif
                    }
                }
            }
        }
    }
    else if constexpr (L_A == StorageLayout::RowMajor && L_B == StorageLayout::RowMajor && L_C == StorageLayout::RowMajor) {
       multiply(A, B.transpose(), C);
    }
    else {
        static_assert(L_A != L_A, "Unsupported Matrix multiply combination!");
    }
}

template<typename T, StorageLayout L_A, StorageLayout L_B, StorageLayout L_C>
void initialize(Matrix<T, L_A>& A, Matrix<T, L_B>& B, Matrix<T, L_C>& C) {
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < A.getRowSize(); ++i) {
        for (size_t j = 0; j < A.getColSize(); ++j) {
            A(i, j) = static_cast<T>(dist(rng));
            B(i, j) = static_cast<T>(dist(rng));
        }
    }
}