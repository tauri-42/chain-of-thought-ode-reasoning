#include "hopfield/HopfieldNetwork.hpp"
#include "hopfield/Integrator.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {

void testVectorAndMatrixMath() {
  using hopfield::Matrix;
  using hopfield::Vector;

  const Vector<double> vector{1.0, 2.0};
  const Matrix<double> matrix{{2.0, 0.0}, {0.0, 3.0}};
  assert((matrix * vector)[0] == 2.0);
  assert((matrix * vector)[1] == 6.0);
  assert(std::abs(hopfield::dot(vector, vector) - 5.0) < 1e-12);
  assert(matrix.isSymmetric());
}

void testNeuronDynamics() {
  hopfield::Neuron<double> neuron(2.0, 0.0, [](double value) { return value; });
  assert(std::abs(neuron.derivative(3.0, 1.0) + 1.0) < 1e-12);
}

void testEulerAndRK4() {
  const hopfield::Derivative<double> exponential = [](double, const hopfield::Vector<double>& state) {
    return state;
  };
  const hopfield::Vector<double> initial{1.0};

  const auto euler = hopfield::EulerIntegrator<double>{}.integrate(
      exponential, 0.0, 1.0, 0.01, initial);
  const auto rk4 = hopfield::RK4Integrator<double>{}.integrate(
      exponential, 0.0, 1.0, 0.01, initial);

  const double expected = std::exp(1.0);
  assert(std::abs(euler.back().state[0] - expected) < 0.02);
  assert(std::abs(rk4.back().state[0] - expected) < 1e-8);
  assert(std::abs(rk4.back().time - 1.0) < 1e-12);
}

void testHopfieldNetwork() {
  hopfield::HopfieldNetwork<double> network(2);
  network.setWeights(hopfield::Matrix<double>{{0.0, 1.0}, {1.0, 0.0}});
  network.setBiases(hopfield::Vector<double>{0.25, -0.25});

  const hopfield::Vector<double> state{0.5, -0.5};
  const auto derivative = network.derivative(0.0, state);
  assert(derivative.size() == 2);
  assert(network.hasSymmetricWeights());
  assert(std::isfinite(network.energy(state)));
}

}  // namespace

int main() {
  testVectorAndMatrixMath();
  testNeuronDynamics();
  testEulerAndRK4();
  testHopfieldNetwork();
  std::cout << "All Hopfield ODE tests passed.\n";
}