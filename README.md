# Hopfield ODE Solver

This is a small, dependency-free C++20 implementation of a continuous-time
Hopfield network. It is organized around the learning milestones from the
project brief:

- `Vector<T>` and `Matrix<T>` provide the basic numerical operations.
- `Neuron<T>` models one continuous-time neuron.
- `HopfieldNetwork<T>` owns neurons, weights, biases, and the Hopfield energy.
- `EulerIntegrator<T>` and `RK4Integrator<T>` integrate any vector ODE with the
  signature `Vector<T>(T time, const Vector<T>& state)`.

The implementation uses standard-library containers so it can compile with the
available compiler immediately. The linear algebra types are intentionally
isolated in `include/hopfield/Vector.hpp` and `include/hopfield/Matrix.hpp`;
they are the only files that need to change when swapping in Eigen later.

## Build and run

```sh
cd cpp
make
./build/hopfield_demo
```

Or with CMake:

```sh
cmake -S cpp -B cpp/build-cmake
cmake --build cpp/build-cmake
ctest --test-dir cpp/build-cmake --output-on-failure
```

The Hopfield equation used by `Neuron<T>` is:

```text
tau * dx/dt = -x + activation(input + bias)
```

The demo stores the bipolar pattern `[+1, -1, +1]`, integrates from a noisy
initial state, and prints the state and energy before and after integration.