"""
Docstring for linear_regression_training
This module contains a class Solution with methods to perform training
in a linear regression model, using NumPy for matrix operations:
- get_derivative: Computes the derivative of the loss with respect to a specific weight.
- get_model_prediction: Computes predictions based on input features and weights.
- train_model: Trains the model by updating weights using gradient descent.
"""

import numpy as np # for numeric operations
from numpy.typing import NDArray # for type hinting NumPy arrays (float64)


class Solution:
    # N - number of samples(len(X))
    # X - Nx3 NumPy array of input features
    # desired_weight - which weight we are calculating the derivative for (0, 1, or 2)

    def get_derivative(
            self, model_prediction: NDArray[np.float64], 
            ground_truth: NDArray[np.float64], N: int,
            X: NDArray[np.float64], desired_weight: int
    ) -> float:
        # MSE (Mean Squared Error) Loss function derivative with respect to one weight
        # ground_truth - model_prediction gives us the error for each sample
        # X[:, desired_weight] where X[row, column] takes all rows for the specified column (x1, x2, or x3 all weights)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64],
                            weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # np.matmul() matrix multiplication(x1*w1 + x2*w2 + x3*w3 for each sample)
        # squeeze() to convert Nx1 array to 1D array of length N
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        # you will need to call get_derivative() for each weight
        for _ in range(num_iterations):
            model_prediction = self.get_model_prediction(X, initial_weights)
            # calculate the derivative for each weight(compute gradient)
            d1 = self.get_derivative(model_prediction, Y, len(X), X, 0)
            d2 = self.get_derivative(model_prediction, Y, len(X), X, 1)
            d3 = self.get_derivative(model_prediction, Y, len(X), X, 2)

        # and update each one separately based on the learning rate!
        # new_weight = old_weight - learning_rate * derivative
            initial_weights[0] = initial_weights[0] - d1 * self.learning_rate
            initial_weights[1] = initial_weights[1] - d2 * self.learning_rate
            initial_weights[2] = initial_weights[2] - d3 * self.learning_rate

        # return np.round(your_answer, 5)
        return np.round(initial_weights, 5)