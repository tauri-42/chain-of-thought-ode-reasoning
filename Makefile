CXX ?= g++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -Wpedantic -O2
INCLUDES := -Iinclude

.PHONY: all demo test clean

all: demo test

demo: build/hopfield_demo

test: build/test_solver
	./build/test_solver

build:
	mkdir -p build

build/hopfield_demo: examples/hopfield_demo.cpp | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

build/test_solver: tests/test_solver.cpp | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -rf build