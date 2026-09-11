

- `Vector<T>` and `Matrix<T>` provide the basic numerical operations.
- `Neuron<T>` models one continuous-time neuron.
- `HopfieldNetwork<T>` owns neurons, weights, biases, and the Hopfield energy.
- `EulerIntegrator<T>` and `RK4Integrator<T>` integrate any vector ODE with the
  signature `Vector<T>(T time, const Vector<T>& state)`.

## uild and run

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
