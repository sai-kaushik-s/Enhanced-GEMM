#include <iostream>
#include <iomanip>
#include <random>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <unistd.h>
#include <immintrin.h>

template<typename T, StorageLayout Layout>
void Matrix<T, Layout>::rebuildDataPointers() {
    if constexpr (IsTransposed) {
        data.resize(colSize);
        for (std::size_t i = 0; i < colSize; ++i)
            data[i] = &(flatData[i * rowSize]);
    } else {
        data.resize(rowSize);
        for (std::size_t i = 0; i < rowSize; ++i)
            data[i] = &(flatData[i * colSize]);
    }
}

template<typename T, StorageLayout Layout>
Matrix<T, Layout>::Matrix(std::size_t rows, std::size_t cols, std::size_t numCores) : 
    rowSize(rows), 
    colSize(cols), 
    packSize(32 / sizeof(T)), 
    numCores(numCores)
{
    flatData.resize(rowSize * colSize); 
    rebuildDataPointers();
    
    long l1dCacheSize = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    double elementSize = sizeof(double);
    tileSize = static_cast<std::size_t>(std::sqrt(l1dCacheSize / (3 * elementSize)));
    tileSize = (tileSize / 4) * 4;
    if (tileSize < 4) { tileSize = 4; }
}

template<typename T, StorageLayout Layout>
Matrix<T, Layout>::Matrix(const Matrix<T, Layout>& other) : 
    rowSize(other.rowSize), 
    colSize(other.colSize), 
    packSize(other.packSize), 
    flatData(other.flatData),
    numCores(other.numCores), 
    tileSize(other.tileSize)
{
    rebuildDataPointers();
}

template<typename T, StorageLayout Layout>
Matrix<T, Layout>& Matrix<T, Layout>::operator=(const Matrix<T, Layout>& other) {
    if (this == &other) { return *this; }

    flatData = other.flatData; 
    
    rowSize = other.rowSize;
    colSize = other.colSize;
    packSize = other.packSize;
    numCores = other.numCores;
    tileSize = other.tileSize;
    
    rebuildDataPointers();

    return *this;
}

template<typename T, StorageLayout Layout>
void Matrix<T, Layout>::randomize() {
    #pragma omp parallel num_threads(getNumCores())
    {
        int tid = omp_get_thread_num();
        std::mt19937_64 rng(12345 + tid * 12345);
        std::normal_distribution<double> dist(0.0, 1.0);
        
        #pragma omp for collapse(2) schedule(static, 1)
        for (std::size_t i = 0; i < rowSize; ++i) {
            for (std::size_t j = 0; j < colSize; ++j) {
                (*this)(i, j) = static_cast<T>(dist(rng));
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
    #pragma omp parallel for num_threads(getNumCores()) reduction(&&:diagonal) schedule(static, 1)
    for (std::size_t i = 0; i < rowSize; ++i) {
        if ((*this)(i, i) != T(1)) {
            diagonal = false;
        }
    }
    
    if (!diagonal) { return false; }

    bool offDiagonal = true;
    #pragma omp parallel for num_threads(getNumCores()) reduction(&&:offDiagonal) collapse(2) schedule(static, 1)
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
    #pragma omp parallel for num_threads(getNumCores()) reduction(&&:allZero) collapse(2) schedule(static, 1)
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
    #pragma omp parallel for num_threads(getNumCores()) reduction(+:checksum) collapse(2) schedule(static, 1)
    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j) {
            checksum += (*this)(i, j);
        }
    }
    return checksum;
}

template<typename T, StorageLayout Layout>
void Matrix<T, Layout>::initializeZero() {
    #pragma omp parallel for num_threads(getNumCores()) collapse(2) schedule(static, 1)
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
std::size_t Matrix<T, Layout>::getTileSize() const { return tileSize; }

template<typename T, StorageLayout Layout>
std::size_t Matrix<T, Layout>::getNumCores() const { return numCores; }

template<typename T, StorageLayout Layout>
T** Matrix<T, Layout>::getData() { return data.data(); }

template<typename T, StorageLayout Layout>
const T* const* Matrix<T, Layout>::getData() const { return data.data(); }

template<typename T, StorageLayout Layout>
T& Matrix<T, Layout>::operator()(std::size_t i, std::size_t j) {
    if constexpr (IsTransposed) { return data[j][i]; } 
    else { return data[i][j]; }
}

template<typename T, StorageLayout Layout>
Matrix<T, (Layout == StorageLayout::RowMajor ? StorageLayout::ColMajor : StorageLayout::RowMajor)> Matrix<T, Layout>::transpose() const {
    Matrix<T, (Layout == StorageLayout::RowMajor ? StorageLayout::ColMajor : StorageLayout::RowMajor)> result(getColSize(), getRowSize(), getNumCores());

    #pragma omp parallel for num_threads(getNumCores()) collapse(2) schedule(static, 1)
    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

template<typename T, StorageLayout Layout>
const T& Matrix<T, Layout>::operator()(std::size_t i, std::size_t j) const {
    if constexpr (IsTransposed) { return data[j][i]; } 
    else { return data[i][j]; }
}

static inline double hsum256_pd(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
    return _mm_cvtsd_f64(s);
}

template<typename T, StorageLayout L_A, StorageLayout L_B, StorageLayout L_C>
void multiply(const Matrix<T, L_A>& A, const Matrix<T, L_B>& B, Matrix<T, L_C>& C) {
    if constexpr (L_A == StorageLayout::RowMajor && L_B == StorageLayout::ColMajor && L_C == StorageLayout::RowMajor) {
        if (A.getColSize() != B.getRowSize()) { throw std::invalid_argument("Dimension mismatch in multiply"); }
        // if (A.isZero() || B.isIdentity()) { C = A; return; }
        // if (B.isZero() || A.isIdentity()) { C = B.transpose(); return; }

        std::size_t tileSize = A.getTileSize();
        std::size_t aRows = A.getRowSize();
        std::size_t bCols = B.getColSize();
        std::size_t aCols = A.getColSize();
        const std::size_t W = 4; // AVX2 256bit / 64bit = 4 doubles

        #pragma omp parallel for num_threads(A.getNumCores()) collapse(2) schedule(static, 1)
        for (std::size_t i = 0; i < aRows; i += tileSize) {
            for (std::size_t j = 0; j < bCols; j += tileSize) {
                for (std::size_t k = 0; k < aCols; k += tileSize) {
                    const std::size_t iiEnd = std::min(i + tileSize, aRows);
                    const std::size_t jjEnd = std::min(j + tileSize, bCols);
                    const std::size_t kkEnd = std::min(k + tileSize, aCols);
                    for (std::size_t ii = i; ii < iiEnd; ++ii) {
                        std::size_t jj = j;
                        const std::size_t jjUnrolledEnd = j + ((jjEnd - j) / 4) * 4;
                        for (; jj < jjUnrolledEnd; jj += 4) {
                            T sum0 = C(ii, jj + 0);
                            T sum1 = C(ii, jj + 1);
                            T sum2 = C(ii, jj + 2);
                            T sum3 = C(ii, jj + 3);
                            __m256d acc0 = _mm256_setzero_pd();
                            __m256d acc1 = _mm256_setzero_pd();
                            __m256d acc2 = _mm256_setzero_pd();
                            __m256d acc3 = _mm256_setzero_pd();

                            std::size_t kk_vec_end = k + ((kkEnd - k) / W) * W;

                            for (std::size_t kk = k; kk < kk_vec_end; kk += W) {
                                __m256d a = _mm256_loadu_pd(&A(ii, kk));

                                __m256d b0 = _mm256_loadu_pd(&B(kk, jj + 0));
                                __m256d b1 = _mm256_loadu_pd(&B(kk, jj + 1));
                                __m256d b2 = _mm256_loadu_pd(&B(kk, jj + 2));
                                __m256d b3 = _mm256_loadu_pd(&B(kk, jj + 3));

                                acc0 = _mm256_fmadd_pd(a, b0, acc0);
                                acc1 = _mm256_fmadd_pd(a, b1, acc1);
                                acc2 = _mm256_fmadd_pd(a, b2, acc2);
                                acc3 = _mm256_fmadd_pd(a, b3, acc3);
                            }
                            sum0 += hsum256_pd(acc0);
                            sum1 += hsum256_pd(acc1);
                            sum2 += hsum256_pd(acc2);
                            sum3 += hsum256_pd(acc3);
                            for (std::size_t kk = kk_vec_end; kk < kkEnd; ++kk) {
                                double a_val = A(ii, kk);
                                sum0 += a_val * B(kk, jj + 0);
                                sum1 += a_val * B(kk, jj + 1);
                                sum2 += a_val * B(kk, jj + 2);
                                sum3 += a_val * B(kk, jj + 3);
                            }
                            C(ii, jj + 0) = sum0;
                            C(ii, jj + 1) = sum1;
                            C(ii, jj + 2) = sum2;
                            C(ii, jj + 3) = sum3;
                        }
                        for (; jj < jjEnd; ++jj) {
                            T sum = C(ii, jj);
                            for (std::size_t kk = k; kk < kkEnd; ++kk) {
                                sum += A(ii, kk) * B(kk, jj);
                            }
                            C(ii, jj) = sum;
                        }
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