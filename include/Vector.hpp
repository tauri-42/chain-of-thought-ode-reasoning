#pragma once

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace hopfield {

template <typename T>
class Vector {
  static_assert(std::is_arithmetic_v<T>, "Vector<T> requires an arithmetic type");

 public:
  Vector() = default;
  explicit Vector(std::size_t size, T value = T{}) : values_(size, value) {}
  Vector(std::initializer_list<T> values) : values_(values) {}

  std::size_t size() const noexcept { return values_.size(); }
  bool empty() const noexcept { return values_.empty(); }

  T& operator[](std::size_t index) noexcept { return values_[index]; }
  const T& operator[](std::size_t index) const noexcept { return values_[index]; }

  T& at(std::size_t index) { return values_.at(index); }
  const T& at(std::size_t index) const { return values_.at(index); }

  typename std::vector<T>::iterator begin() noexcept { return values_.begin(); }
  typename std::vector<T>::iterator end() noexcept { return values_.end(); }
  typename std::vector<T>::const_iterator begin() const noexcept {
    return values_.begin();
  }
  typename std::vector<T>::const_iterator end() const noexcept {
    return values_.end();
  }

  T* data() noexcept { return values_.data(); }
  const T* data() const noexcept { return values_.data(); }

  Vector& operator+=(const Vector& other) {
    requireSameSize(other, "addition");
    for (std::size_t i = 0; i < size(); ++i) {
      values_[i] += other[i];
    }
    return *this;
  }

  Vector& operator-=(const Vector& other) {
    requireSameSize(other, "subtraction");
    for (std::size_t i = 0; i < size(); ++i) {
      values_[i] -= other[i];
    }
    return *this;
  }

  Vector& operator*=(T scalar) {
    for (auto& value : values_) {
      value *= scalar;
    }
    return *this;
  }

  Vector& operator/=(T scalar) {
    if (scalar == T{}) {
      throw std::invalid_argument("cannot divide a vector by zero");
    }
    for (auto& value : values_) {
      value /= scalar;
    }
    return *this;
  }

 private:
  void requireSameSize(const Vector& other, const char* operation) const {
    if (size() != other.size()) {
      throw std::invalid_argument("vector size mismatch during " + std::string(operation));
    }
  }

  std::vector<T> values_;
};

template <typename T>
Vector<T> operator+(Vector<T> left, const Vector<T>& right) {
  return left += right;
}

template <typename T>
Vector<T> operator-(Vector<T> left, const Vector<T>& right) {
  return left -= right;
}

template <typename T>
Vector<T> operator-(Vector<T> vector) {
  return vector *= T{-1};
}

template <typename T>
Vector<T> operator*(Vector<T> vector, T scalar) {
  return vector *= scalar;
}

template <typename T>
Vector<T> operator*(T scalar, Vector<T> vector) {
  return vector *= scalar;
}

template <typename T>
Vector<T> operator/(Vector<T> vector, T scalar) {
  return vector /= scalar;
}

template <typename T>
T dot(const Vector<T>& left, const Vector<T>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector size mismatch during dot product");
  }

  T result{};
  for (std::size_t i = 0; i < left.size(); ++i) {
    result += left[i] * right[i];
  }
  return result;
}

template <typename T>
T squaredNorm(const Vector<T>& vector) {
  return dot(vector, vector);
}

template <typename T>
T norm(const Vector<T>& vector) {
  return static_cast<T>(std::sqrt(squaredNorm(vector)));
}

template <typename T>
std::ostream& operator<<(std::ostream& output, const Vector<T>& vector) {
  output << '[';
  for (std::size_t i = 0; i < vector.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << vector[i];
  }
  return output << ']';
}

}  // namespace hopfield