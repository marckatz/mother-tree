# Mother Tree Optimization

"A new swarm intelligence algorithm called Mother Tree Optimization (MTO) for solving continuous optimization problems"
Based off Korani et. al. 2019
https://ieeexplore.ieee.org/document/8914049

This algorithm is based off how Douglas Fir trees use fungi (what the authors call "The Mycorrhizal Fungi Network (MFN)") to communicate between each other.
The algorithm has a central "mother tree" which represents the best known node, which gives its data ("nutrients") to the other nodes, which then further pass it on in order

![Figure 1](https://github.com/marckatz/mother-tree/assets/5956327/317f9cf6-e748-433a-b9c7-091603605b84)

From Korani et. al. 2019

## Current Features
Test the algorithm by selecting a certain fitness function (Rastagrin, three-hump camel, sphere, Salomon, normalized Schwefel, Rosenbrock, Griewank)
 - make sure ranges (lines 80-92), fitness function (lines 114-120), and true minimum (lines 251-257) match
Certain parameters can be tweaked:
 - dimensionality (line 71)
 - "root signal" step size (line 74)
 - "MFN" step size (line 77)
 - Population size (line 123)

Running mothertree.py will run the optimization algorithm based on the fitness function and parameters specified. The time taken and accuracy will depend on the fitness functions and parameters. 
The output will show you the best node, the true minimum, and the error. 
