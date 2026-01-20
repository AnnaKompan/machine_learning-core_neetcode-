"""
Docstring for gradient_descent
This module contains a class Solution with a method get_minimizer that performs gradient descent
to find the minimum of the function f(x) = x^2.

Gradient descent is an optimization algorithm 
used to minimize functions by iteratively moving 
towards the steepest descent as defined by the negative of the gradient.
Link: https://neetcode.io/problems/gradient-descent/question 
"""
class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        for i in range(iterations):
            curr_d = 2 * init
            init = init - curr_d * learning_rate
        return round(init,5)