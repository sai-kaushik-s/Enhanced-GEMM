#include <iostream>
#include <iomanip>
#include <random>
#include <omp.h>
#include "matrix.h"

template<typename T>
Matrix<T>::Matrix(int rows, int cols, int numCores, bool isTransposed)
    : rowSize(rows), colSize(cols), packSize(32 / sizeof(T)),  numCores(numCores), isTransposed(isTransposed)
{
    if (isTransposed) {
        data = new T*[colSize];
        for (int i = 0; i < colSize; ++i)
            data[i] = new T[rowSize];
    } else {
        data = new T*[rowSize];
        for (int i = 0; i < rowSize; ++i)
            data[i] = new T[colSize];
    }
}

template<typename T>
Matrix<T>::Matrix(const Matrix& other)
    : rowSize(other.rowSize), colSize(other.colSize), numCores(other.numCores), packSize(other.packSize), isTransposed(other.isTransposed)
{
    if (isTransposed) {
        data = new T*[colSize];
        for (int i = 0; i < colSize; ++i)
            data[i] = new T[rowSize];
    } else {
        data = new T*[rowSize];
        for (int i = 0; i < rowSize; ++i)
            data[i] = new T[colSize];
    }
    for (int i = 0; i < rowSize; ++i) {
        for (int j = 0; j < colSize; ++j)
            (*this)(i, j) = other(i, j);
    }
}

template<typename T>
Matrix<T>::~Matrix() {
    int numPtrs = (isTransposed) ? colSize : rowSize;
    for (int i = 0; i < numPtrs; ++i)
        delete[] data[i];
    delete[] data;
}

template<typename T>
void Matrix<T>::randomize() {
    std::random_device rd;
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < rowSize; ++i)
        for (int j = 0; j < colSize; ++j)
            (*this)(i, j) = static_cast<T>(dist(rng));
}

template<typename T>
void Matrix<T>::print() const {
    for (int i = 0; i < rowSize; ++i) {
        for (int j = 0; j < colSize; ++j)
            std::cout << std::setw(8) << (*this)(i, j) << " ";
        std::cout << std::endl;
    }
}

template<typename T>
void Matrix<T>::printAsIs() const {
    for (int i = 0; i < ((isTransposed) ? colSize : rowSize); ++i) {
        for (int j = 0; j < ((isTransposed) ? rowSize : colSize); ++j)
            std::cout << std::setw(8) << data[i][j] << " ";
        std::cout << std::endl;
    }
}


template<typename T>
bool Matrix<T>::isIdentity() const {
    if (rowSize != colSize) return false;

    bool diagonal_ok = true;
    #pragma omp parallel for reduction(&&:diagonal_ok)
    for (size_t i = 0; i < rowSize; ++i) {
        if ((*this)(i, i) != T(1)) {
            diagonal_ok = false;
        }
    }
    
    if (!diagonal_ok) return false;

    bool off_diagonal_ok = true;
    #pragma omp parallel for reduction(&&:off_diagonal_ok) collapse(2)
    for (size_t i = 0; i < rowSize; ++i) {
        for (size_t j = 0; j < colSize; ++j) {
            if (i != j && (*this)(i, j) != T(0)) {
                off_diagonal_ok = false;
            }
        }
    }

    return off_diagonal_ok;
}

template<typename T>
bool Matrix<T>::isZero() const {
    bool all_zero = true;
    #pragma omp parallel for reduction(&&:all_zero) collapse(2)
    for (size_t i = 0; i < rowSize; ++i) {
        for (size_t j = 0; j < colSize; ++j) {
            if ((*this)(i, j) != T(0)) {
                all_zero = false;
            }
        }
    }
    return all_zero;
}


template<typename T>
int Matrix<T>::rows() const { return rowSize; }

template<typename T>
int Matrix<T>::cols() const { return colSize; }

template<typename T>
int Matrix<T>::packs() const { return packSize; }

template<typename T>
bool Matrix<T>::transposed() const { return isTransposed; }

template<typename T>
T** Matrix<T>::getData() const { return data; }

template<typename T>
T& Matrix<T>::operator()(int i, int j) { return (isTransposed) ? data[j][i] : data[i][j]; }

template<typename T>
const T& Matrix<T>::operator()(int i, int j) const { return (isTransposed) ? data[j][i] : data[i][j]; }

template<typename T>
Matrix<T> multiply(const Matrix<T>& A, const Matrix<T>& B, int TILE_SIZE) {
    if (A.cols() != B.rows()) 
        throw std::invalid_argument("Dimension mismatch in multiply");
    if (A.isZero() || B.isIdentity()) return A;
    if (B.isZero() || A.isIdentity()) return B;

    Matrix<T> C(A.rows(), B.cols(), A.numCores);
    
    #pragma omp parallel for num_threads(A.numCores)
    for (int i = 0; i < C.rows(); ++i) {
        for (int j = 0; j < C.cols(); ++j) {
            C(i, j) = T(0);
        }
    }

    #pragma omp parallel for num_threads(A.numCores)
    for (int i = 0; i < A.rows(); i += TILE_SIZE) {
        for (int j = 0; j < B.cols(); j += TILE_SIZE) {
            for (int k = 0; k < A.cols(); k += TILE_SIZE) {
                for (int ii = i; ii < std::min(i + TILE_SIZE, A.rows()); ++ii) {
                    int jj = j;
                    const int jj_tile_end = std::min(j + TILE_SIZE, B.cols());
                    const int jj_unrolled_end = j + ((jj_tile_end - j) / 4) * 4;
                    for (; jj < jj_unrolled_end; jj += 4) {
                        T sum0 = C(ii, jj + 0);
                        T sum1 = C(ii, jj + 1);
                        T sum2 = C(ii, jj + 2);
                        T sum3 = C(ii, jj + 3);
                        for (int kk = k; kk < std::min(k + TILE_SIZE, A.cols()); ++kk) {
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
                    for (; jj < jj_tile_end; ++jj) {
                        T sum = C(ii, jj);
                        for (int kk = k; kk < std::min(k + TILE_SIZE, A.cols()); ++kk) {
                            sum += A(ii, kk) * B(kk, jj);
                        }
                        C(ii, jj) = sum;
                    }
                }
            }
        }
    }
    return C;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) {
    if (this == &other) {
        return *this;
    }

    int numPtrs = (isTransposed) ? colSize : rowSize;
    for (int i = 0; i < numPtrs; ++i)
        delete[] data[i];
    delete[] data;

    rowSize = other.rowSize;
    colSize = other.colSize;
    packSize = other.packSize;
    isTransposed = other.isTransposed;

    if (isTransposed) {
        data = new T*[colSize];
        for (int i = 0; i < colSize; ++i)
            data[i] = new T[rowSize];
    } else {
        data = new T*[rowSize];
        for (int i = 0; i < rowSize; ++i)
            data[i] = new T[colSize];
    }

    for (int i = 0; i < rowSize; ++i) {
        for (int j = 0; j < colSize; ++j)
            (*this)(i, j) = other(i, j);
    }

    return *this;
}

template<typename T>
Matrix<T>::Matrix(Matrix<T>&& other) noexcept
    : rowSize(other.rowSize), colSize(other.colSize), packSize(other.packSize), 
      data(other.data), isTransposed(other.isTransposed) 
{
    other.data = nullptr;
    other.rowSize = 0;
    other.colSize = 0;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    int numPtrs = (isTransposed) ? colSize : rowSize;
    for (int i = 0; i < numPtrs; ++i)
        delete[] data[i];
    delete[] data;

    data = other.data;
    rowSize = other.rowSize;
    colSize = other.colSize;
    packSize = other.packSize;
    isTransposed = other.isTransposed;

    other.data = nullptr;
    other.rowSize = 0;
    other.colSize = 0;

    return *this;
}