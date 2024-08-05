F.I.N.D.E.R stands for Filtering Informed Newton-like Derivative-free Evolutionary Recursion. 

It is an iterative optimization scheme which builds on the direct computaion of the diagonal terms in the inverse Hessian matrix, approximated using the principles of stochastic filtering. FINDER incorporates an innovative way of utilising these diagonal terms to navigate large dimensional design spaces. It not only ensures rapid advancement in a direction of descent but also explores the neighbourhood in every iteration to find the fittest particle, thereby rendering the algorithm fast and robust.

This repository contains Python codes for a number of optimization problems solved using FINDER and demonstrates its superiority over the Adam optimizer. 

Note : The examples require importing the FINDER_core.py module. So it must be downloaded in the same directory or a path must be specified while importing.
