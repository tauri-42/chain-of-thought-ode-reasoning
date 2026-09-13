#pragma once

#include "hopfield/Matrix.hpp"
#include "hopfield/Neuron.hpp"
#include "hopfield/Vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace hopfield {

template <typename T>
class HopfieldNetwork {
  static_assert(std::is_floating_point_v<T>,
                "HopfieldNetwork<T> should use a floating-point type");

 public:
  using Activation = typename Neuron<T>::Activation;

  explicit HopfieldNetwork(std::size_t neuronCount, T timeConstant = T{1},
                           Activation activation = defaultActivation())
      : weights_(neuronCount, neuronCount),
        biases_(neuronCount),
        neurons_(neuronCount, Neuron<T>(timeConstant, T{}, activation)) {}

  std::size_t size() const noexcept { return neurons_.size(); }

  const Matrix<T>& weights() const noexcept { return weights_; }
  const Vector<T>& biases() const noexcept { return biases_; }

  void setWeights(const Matrix<T>& weights) {
    if (weights.rows() != size() || weights.columns() != size()) {
      throw std::invalid_argument("weight matrix must match network size");
    }
    weights_ = weights;
  }

  void setWeight(std::size_t target, std::size_t source, T value) {
    requireNeuronIndex(target);
    requireNeuronIndex(source);
    weights_(target, source) = value;
  }

  void setBiases(const Vector<T>& biases) {
    if (biases.size() != size()) {
      throw std::invalid_argument("bias vector must match network size");
    }
    biases_ = biases;
    for (std::size_t i = 0; i < size(); ++i) {
      neurons_[i].setBias(biases_[i]);
    }
  }

  void setBias(std::size_t index, T value) {
    requireNeuronIndex(index);
    biases_[index] = value;
    neurons_[index].setBias(value);
  }

  void setTimeConstant(T timeConstant) {
    for (auto& neuron : neurons_) {
      neuron.setTimeConstant(timeConstant);
    }
  }

  Vector<T> derivative(T /*time*/, const Vector<T>& state) const {
    requireStateSize(state);
    const Vector<T> input = weights_ * state;
    Vector<T> result(size());
    for (std::size_t i = 0; i < size(); ++i) {
      result[i] = neurons_[i].derivative(state[i], input[i]);
    }
    return result;
  }

  T energy(const Vector<T>& state) const {
    requireStateSize(state);
    // This is the standard Hopfield energy for symmetric weights.
    return static_cast<T>(-0.5) * dot(state, weights_ * state) - dot(biases_, state);
  }

  bool hasSymmetricWeights(T tolerance = static_cast<T>(1e-9)) const {
    return weights_.isSymmetric(tolerance);
  }

  Vector<T> recall(const Vector<T>& initialState, T duration, T stepSize) const {
    if (duration < T{}) {
      throw std::invalid_argument("recall duration must not be negative");
    }
    if (stepSize <= T{}) {
      throw std::invalid_argument("recall step size must be positive");
    }
    Vector<T> state = initialState;
    T time{};
    while (time < duration) {
      const T step = std::min(stepSize, duration - time);
      const Vector<T> slope = derivative(time, state);
      state += step * slope;
      time += step;
    }
    return state;
  }

 private:
  static Activation defaultActivation() {
    return [](T value) { return std::tanh(value); };
  }

  void requireNeuronIndex(std::size_t index) const {
    if (index >= size()) {
      throw std::out_of_range("neuron index is outside the network");
    }
  }

  void requireStateSize(const Vector<T>& state) const {
    if (state.size() != size()) {
      throw std::invalid_argument("state vector must match network size");
    }
  }

  Matrix<T> weights_;
  Vector<T> biases_;
  std::vector<Neuron<T>> neurons_;
};

}  // namespace hopfield