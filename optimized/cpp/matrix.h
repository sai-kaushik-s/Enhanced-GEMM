#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <memory>
#include <cstddef>
#include <stdexcept>


template<typename T, bool IsTransposed = false>
class Matrix;

template<typename T, bool T_A, bool T_B, bool T_C>
void multiply(const Matrix<T, T_A>& A, const Matrix<T, T_B>& B, Matrix<T, T_C>& C);

template<typename T, bool IsTransposed>
class Matrix {
private:
    std::size_t rowSize;
    std::size_t colSize;
    std::size_t packSize;
    std::vector<T> flatData;
    std::vector<T*> data;
    std::size_t numCores;
    std::size_t tileSize;

    void rebuildDataPointers();

public:
    Matrix(std::size_t rows, std::size_t cols, std::size_t numCores);
    Matrix(const Matrix& other);
    
    Matrix<T, IsTransposed>& operator=(const Matrix<T, IsTransposed>& other);
    Matrix(Matrix<T, IsTransposed>&& other) noexcept = default;
    Matrix<T, IsTransposed>& operator=(Matrix<T, IsTransposed>&& other) noexcept = default;
    
    ~Matrix() = default;

    void randomize();
    void print() const;
    void write(std::ostream& os) const;
    void initializeZero();

    std::size_t getRowSize() const;
    std::size_t getColSize() const;
    std::size_t getPackSize() const;
    std::size_t getTileSize() const;
    std::size_t getNumCores() const;
    T** getData();
    const T* const* getData() const;

    T& operator()(std::size_t i, std::size_t j);
    const T& operator()(std::size_t i, std::size_t j) const;

    Matrix<T, !IsTransposed> transpose() const;

    bool isIdentity() const;
    bool isZero() const;
    T getChecksum() const;

    friend void multiply<>(const Matrix<T, false>& A, const Matrix<T, true>& B, Matrix<T, false>& C);
    friend void multiply<>(const Matrix<T, false>& A, const Matrix<T, false>& B, Matrix<T, false>& C);
};

#include "matrix.tpp"

#endif