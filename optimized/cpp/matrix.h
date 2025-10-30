#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <iomanip>
#include <random>

template<typename T>
class Matrix;

template<typename T>
Matrix<T> multiply(const Matrix<T>& A, const Matrix<T>& B, int TILE_SIZE);


template<typename T>
class Matrix {
private:
    int rowSize;
    int colSize;
    int packSize;
    T** data;
    bool isTransposed;
    int numCores;

public:
    Matrix(int rows, int cols, int numCores, bool transposed = false);
    Matrix(const Matrix& other);
    ~Matrix();

    void randomize();
    void print() const;
    void printAsIs() const;

    int rows() const;
    int cols() const;
    int packs() const;
    bool transposed() const;
    T** getData() const;

    T& operator()(int i, int j);
    const T& operator()(int i, int j) const;

    Matrix<T>& operator=(const Matrix<T>& other);
    Matrix(Matrix<T>&& other) noexcept;
    Matrix<T>& operator=(Matrix<T>&& other) noexcept;

    bool isIdentity() const;
    bool isZero() const;

    friend Matrix<T> multiply<>(const Matrix<T>& A, const Matrix<T>& B, int TILE_SIZE);
};

#include "matrix.tpp"

#endif