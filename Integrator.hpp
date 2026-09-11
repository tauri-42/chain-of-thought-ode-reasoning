#pragma once

#include "hopfield/Vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hopfield {

template <typename T>
struct Sample {
  T time;
  Vector<T> state;
};

template <typename T>
using Derivative = std::function<Vector<T>(T, const Vector<T>&)>;

template <typename T>
class EulerIntegrator {
 public:
  Vector<T> step(const Derivative<T>& system, T time, const Vector<T>& state,
                 T stepSize) const {
    validateStepSize(stepSize);
    return state + stepSize * system(time, state);
  }

  std::vector<Sample<T>> integrate(const Derivative<T>& system, T startTime,
                                    T endTime, T stepSize,
                                    const Vector<T>& initialState) const {
    validateInterval(startTime, endTime, stepSize);
    std::vector<Sample<T>> trajectory{{startTime, initialState}};
    T time = startTime;
    Vector<T> state = initialState;

    while (time < endTime) {
      const T step = std::min(stepSize, endTime - time);
      state = this->step(system, time, state, step);
      time += step;
      trajectory.push_back({time, state});
    }
    return trajectory;
  }

 private:
  static void validateStepSize(T stepSize) {
    if (stepSize <= T{}) {
      throw std::invalid_argument("integrator step size must be positive");
    }
  }

  static void validateInterval(T startTime, T endTime, T stepSize) {
    validateStepSize(stepSize);
    if (endTime < startTime) {
      throw std::invalid_argument("integration end time must not precede start time");
    }
  }
};

template <typename T>
class RK4Integrator {
 public:
  Vector<T> step(const Derivative<T>& system, T time, const Vector<T>& state,
                 T stepSize) const {
    if (stepSize <= T{}) {
      throw std::invalid_argument("integrator step size must be positive");
    }

    const auto k1 = system(time, state);
    const auto k2 = system(time + stepSize / T{2}, state + (stepSize / T{2}) * k1);
    const auto k3 = system(time + stepSize / T{2}, state + (stepSize / T{2}) * k2);
    const auto k4 = system(time + stepSize, state + stepSize * k3);
    return state + (stepSize / T{6}) * (k1 + T{2} * k2 + T{2} * k3 + k4);
  }

  std::vector<Sample<T>> integrate(const Derivative<T>& system, T startTime,
                                    T endTime, T stepSize,
                                    const Vector<T>& initialState) const {
    if (stepSize <= T{}) {
      throw std::invalid_argument("integrator step size must be positive");
    }
    if (endTime < startTime) {
      throw std::invalid_argument("integration end time must not precede start time");
    }

    std::vector<Sample<T>> trajectory{{startTime, initialState}};
    T time = startTime;
    Vector<T> state = initialState;

    while (time < endTime) {
      const T step = std::min(stepSize, endTime - time);
      state = this->step(system, time, state, step);
      time += step;
      trajectory.push_back({time, state});
    }
    return trajectory;
  }
};

}  // namespace hopfield