#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <memory>
#include <cstddef>
#include <stdexcept>

enum class StorageLayout {
    RowMajor,
    ColMajor
};

template<typename T, StorageLayout Layout>
class Matrix;

template<typename T, StorageLayout L_A, StorageLayout L_B, StorageLayout L_C>
void multiply(const Matrix<T, L_A>& A, const Matrix<T, L_B>& B, Matrix<T, L_C>& C);

template<typename T, StorageLayout Layout>
class Matrix {
private:
    std::size_t rowSize;
    std::size_t colSize;
    std::size_t packSize;
    std::vector<T> flatData;
    std::size_t numCores;

    std::vector<T> packedData;
    bool isPacked = false;
    std::size_t packedMC, packedKC, packedNC;

    static constexpr bool IsTransposed = (Layout == StorageLayout::ColMajor);

public:
    Matrix(std::size_t rows, std::size_t cols, std::size_t numCores);
    Matrix(const Matrix& other);
    
    Matrix<T, Layout>& operator=(const Matrix<T, Layout>& other);;
    Matrix(Matrix<T, Layout>&& other) noexcept = default;
    Matrix<T, Layout>& operator=(Matrix<T, Layout>&& other) noexcept = default;
    
    ~Matrix() = default;

    void randomize(int seq_stride, int seq_offset);
    void print() const;
    void write(std::ostream& os) const;
    void initializeZero();

    void packMatrix();

    std::size_t getRowSize() const;
    std::size_t getColSize() const;
    std::size_t getPackSize() const;
    std::size_t getNumCores() const;
    bool getIsPacked() const;
    std::size_t getPackedMC() const;
    std::size_t getPackedKC() const;
    std::size_t getPackedNC() const;

    T* getData();
    const T* getData() const;
    const T* getPackedData() const;

    T& operator()(std::size_t i, std::size_t j);
    const T& operator()(std::size_t i, std::size_t j) const;

    Matrix<T, (Layout == StorageLayout::RowMajor ? StorageLayout::ColMajor : StorageLayout::RowMajor)> transpose() const;

    bool isIdentity() const;
    bool isZero() const;
    T getChecksum() const;

    friend void multiply<>(const Matrix<T, StorageLayout::RowMajor>& A, const Matrix<T, StorageLayout::ColMajor>& B, Matrix<T, StorageLayout::RowMajor>& C);
    friend void multiply<>(const Matrix<T, StorageLayout::RowMajor>& A, const Matrix<T, StorageLayout::RowMajor>& B, Matrix<T, StorageLayout::RowMajor>& C);
};

#include "matrix.tpp"

#endif