#pragma once

#include "hopfield/Vector.hpp"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace hopfield {

template <typename T>
class Matrix {
  static_assert(std::is_arithmetic_v<T>, "Matrix<T> requires an arithmetic type");

 public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t columns, T value = T{})
      : rows_(rows), columns_(columns), values_(rows * columns, value) {}

  Matrix(std::initializer_list<std::initializer_list<T>> rows) {
    rows_ = rows.size();
    columns_ = rows_ == 0 ? 0 : rows.begin()->size();
    values_.reserve(rows_ * columns_);

    for (const auto& row : rows) {
      if (row.size() != columns_) {
        throw std::invalid_argument("all matrix rows must have the same length");
      }
      values_.insert(values_.end(), row.begin(), row.end());
    }
  }

  std::size_t rows() const noexcept { return rows_; }
  std::size_t columns() const noexcept { return columns_; }

  T& operator()(std::size_t row, std::size_t column) noexcept {
    return values_[row * columns_ + column];
  }
  const T& operator()(std::size_t row, std::size_t column) const noexcept {
    return values_[row * columns_ + column];
  }

  static Matrix identity(std::size_t size) {
    Matrix result(size, size);
    for (std::size_t i = 0; i < size; ++i) {
      result(i, i) = T{1};
    }
    return result;
  }

  Vector<T> operator*(const Vector<T>& vector) const {
    if (columns_ != vector.size()) {
      throw std::invalid_argument("matrix/vector size mismatch during multiplication");
    }

    Vector<T> result(rows_);
    for (std::size_t row = 0; row < rows_; ++row) {
      for (std::size_t column = 0; column < columns_; ++column) {
        result[row] += (*this)(row, column) * vector[column];
      }
    }
    return result;
  }

  Matrix transpose() const {
    Matrix result(columns_, rows_);
    for (std::size_t row = 0; row < rows_; ++row) {
      for (std::size_t column = 0; column < columns_; ++column) {
        result(column, row) = (*this)(row, column);
      }
    }
    return result;
  }

  bool isSymmetric(T tolerance = static_cast<T>(1e-9)) const {
    if (rows_ != columns_) {
      return false;
    }
    for (std::size_t row = 0; row < rows_; ++row) {
      for (std::size_t column = row + 1; column < columns_; ++column) {
        if (std::abs((*this)(row, column) - (*this)(column, row)) > tolerance) {
          return false;
        }
      }
    }
    return true;
  }

 private:
  std::size_t rows_{0};
  std::size_t columns_{0};
  std::vector<T> values_;
};

template <typename T>
Matrix<T> operator+(const Matrix<T>& left, const Matrix<T>& right) {
  if (left.rows() != right.rows() || left.columns() != right.columns()) {
    throw std::invalid_argument("matrix size mismatch during addition");
  }

  Matrix<T> result(left.rows(), left.columns());
  for (std::size_t row = 0; row < left.rows(); ++row) {
    for (std::size_t column = 0; column < left.columns(); ++column) {
      result(row, column) = left(row, column) + right(row, column);
    }
  }
  return result;
}

}  // namespace hopfield