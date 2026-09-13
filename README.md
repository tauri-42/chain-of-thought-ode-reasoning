#chain of thought as ode representation

research question: Given some vector a(t) that represents the weights of all attributions prior at some time t, use some continuous time ODE F(x) to model the behavior of a(t) for different reasoning convergences


#part 1: attribution tracking
attribution is tracked through a feedforward transformer using huggingface's ___ model. this is done in google colab and generates large matrix datasets of each attribution step (weights). 

we export this as a h5py file into a c++ ode solver

#part 2 ode solver
ode solver is custom built because i want it to account for hopfield odes, which is what this research is based on. classes present: 

- integrator
- neuron
- vector
- matrix

and analysis classes: 

- jacobian
- fixed point
- stability

these classify each system in some way. 

