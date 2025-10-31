#include <iostream>
#include <iomanip>
#include <random>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <unistd.h>

template<typename T, bool IsTransposed>
void Matrix<T, IsTransposed>::rebuildDataPointers() {
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

template<typename T, bool IsTransposed>
Matrix<T, IsTransposed>::Matrix(std::size_t rows, std::size_t cols, std::size_t numCores)
    : rowSize(rows), colSize(cols), packSize(32 / sizeof(T)), numCores(numCores)
{
    flatData.resize(rowSize * colSize); 
    rebuildDataPointers();
    
    long l1dCacheSize = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    double elementSize = sizeof(double);
    tileSize = static_cast<std::size_t>(std::sqrt(l1dCacheSize / (3 * elementSize)));
    tileSize = (tileSize / 4) * 4;
    if (tileSize < 4) { tileSize = 4; }
}

template<typename T, bool IsTransposed>
Matrix<T, IsTransposed>::Matrix(const Matrix& other)
    : rowSize(other.rowSize), 
      colSize(other.colSize), 
      packSize(other.packSize), 
      flatData(other.flatData),
      numCores(other.numCores), 
      tileSize(other.tileSize)
{
    rebuildDataPointers();
}

template<typename T, bool IsTransposed>
Matrix<T, IsTransposed>& Matrix<T, IsTransposed>::operator=(const Matrix<T, IsTransposed>& other) {
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

template<typename T, bool IsTransposed>
void Matrix<T, IsTransposed>::randomize() {
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (std::size_t i = 0; i < rowSize; ++i)
        for (std::size_t j = 0; j < colSize; ++j)
            (*this)(i, j) = static_cast<T>(dist(rng));
}

template<typename T, bool IsTransposed>
void Matrix<T, IsTransposed>::write(std::ostream& os) const {
    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j)
            os << std::setw(8) << (*this)(i, j) << " ";
        os << '\n';
    }
}

template<typename T, bool IsTransposed>
void Matrix<T, IsTransposed>::print() const {
    write(std::cout);
}

template<typename T, bool IsTransposed>
bool Matrix<T, IsTransposed>::isIdentity() const {
    if (rowSize != colSize) { return false; }

    bool diagonal = true;
    #pragma omp parallel for num_threads(getNumCores()) reduction(&&:diagonal)
    for (std::size_t i = 0; i < rowSize; ++i) {
        if ((*this)(i, i) != T(1)) {
            diagonal = false;
        }
    }
    
    if (!diagonal) { return false; }

    bool offDiagonal = true;
    #pragma omp parallel for num_threads(getNumCores()) reduction(&&:offDiagonal) collapse(2)
    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j) {
            if (i != j && (*this)(i, j) != T(0)) {
                offDiagonal = false;
            }
        }
    }

    return offDiagonal;
}

template<typename T, bool IsTransposed>
bool Matrix<T, IsTransposed>::isZero() const {
    bool allZero = true;
    #pragma omp parallel for num_threads(getNumCores()) reduction(&&:allZero) collapse(2)
    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j) {
            if ((*this)(i, j) != T(0)) {
                allZero = false;
            }
        }
    }
    return allZero;
}

template<typename T, bool IsTransposed>
T Matrix<T, IsTransposed>::getChecksum() const {
    T checksum = T(0);
    #pragma omp parallel for num_threads(getNumCores()) reduction(+:checksum) collapse(2)
    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j) {
            checksum += (*this)(i, j);
        }
    }
    return checksum;
}

template<typename T, bool IsTransposed>
void Matrix<T, IsTransposed>::initializeZero() {
    #pragma omp parallel for num_threads(getNumCores()) collapse(2)
    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j) {
            (*this)(i, j) = T(0);
        }
    }
}

template<typename T, bool IsTransposed>
std::size_t Matrix<T, IsTransposed>::getRowSize() const { return rowSize; }

template<typename T, bool IsTransposed>
std::size_t Matrix<T, IsTransposed>::getColSize() const { return colSize; }

template<typename T, bool IsTransposed>
std::size_t Matrix<T, IsTransposed>::getPackSize() const { return packSize; }

template<typename T, bool IsTransposed>
std::size_t Matrix<T, IsTransposed>::getTileSize() const { return tileSize; }

template<typename T, bool IsTransposed>
std::size_t Matrix<T, IsTransposed>::getNumCores() const { return numCores; }

template<typename T, bool IsTransposed>
T** Matrix<T, IsTransposed>::getData() { return data.data(); }

template<typename T, bool IsTransposed>
const T* const* Matrix<T, IsTransposed>::getData() const { return data.data(); }

template<typename T, bool IsTransposed>
T& Matrix<T, IsTransposed>::operator()(std::size_t i, std::size_t j) {
    if constexpr (IsTransposed) { return data[j][i]; } 
    else { return data[i][j]; }
}

template<typename T, bool IsTransposed>
Matrix<T, !IsTransposed> Matrix<T, IsTransposed>::transpose() const {
    Matrix<T, !IsTransposed> result(getColSize(), getRowSize(), getNumCores());

    #pragma omp parallel for num_threads(getNumCores()) collapse(2)
    for (std::size_t i = 0; i < rowSize; ++i) {
        for (std::size_t j = 0; j < colSize; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

template<typename T, bool IsTransposed>
const T& Matrix<T, IsTransposed>::operator()(std::size_t i, std::size_t j) const {
    if constexpr (IsTransposed) { return data[j][i]; } 
    else { return data[i][j]; }
}

template<typename T, bool T_A, bool T_B, bool T_C>
void multiply(const Matrix<T, T_A>& A, const Matrix<T, T_B>& B, Matrix<T, T_C>& C) {
    if constexpr (!T_A && T_B && !T_C) {
        if (A.getColSize() != B.getRowSize()) { throw std::invalid_argument("Dimension mismatch in multiply"); }
        if (A.isZero() || B.isIdentity()) { C = A; return; }
        if (B.isZero() || A.isIdentity()) { C = B.transpose(); return; }

        std::size_t tileSize = A.getTileSize();
        std::size_t aRows = A.getRowSize();
        std::size_t bCols = B.getColSize();
        std::size_t aCols = A.getColSize();

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
                            for (std::size_t kk = k; kk < kkEnd; ++kk) {
                                T a_val = A(ii, kk);
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
    else if constexpr (!T_A && !T_B && !T_C) {
        multiply(A, B.transpose(), C);
    }
    else {
        static_assert(T_A != T_A, "Unsupported Matrix multiply combination!");
    }
}