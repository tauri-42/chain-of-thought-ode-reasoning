#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hopfield {

template <typename T>
class Neuron {
  static_assert(std::is_floating_point_v<T>,
                "Neuron<T> should use a floating-point type");

 public:
  using Activation = std::function<T(T)>;

  explicit Neuron(T timeConstant = T{1}, T bias = T{},
                  Activation activation = defaultActivation())
      : time_constant_(timeConstant), bias_(bias), activation_(std::move(activation)) {
    if (time_constant_ <= T{}) {
      throw std::invalid_argument("a neuron's time constant must be positive");
    }
    if (!activation_) {
      throw std::invalid_argument("a neuron's activation function cannot be empty");
    }
  }

  T timeConstant() const noexcept { return time_constant_; }
  T bias() const noexcept { return bias_; }

  void setTimeConstant(T timeConstant) {
    if (timeConstant <= T{}) {
      throw std::invalid_argument("a neuron's time constant must be positive");
    }
    time_constant_ = timeConstant;
  }

  void setBias(T bias) noexcept { bias_ = bias; }

  T output(T input) const { return activation_(input + bias_); }

  // Continuous-time Hopfield dynamics:
  //     tau * dx/dt = -x + activation(input + bias)
  T derivative(T state, T input) const {
    return (-state + output(input)) / time_constant_;
  }

 private:
  static Activation defaultActivation() {
    return [](T value) { return std::tanh(value); };
  }

  T time_constant_;
  T bias_;
  Activation activation_;
};

}  // namespace hopfield