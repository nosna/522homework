import numpy as np


class LinearRegression:
    """
    w: np.ndarray
    b: float
    """

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Closed form solution for Linear Regression.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.
        """
        w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.b = w[0]
        self.w = w[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication with X and w and add bias term b.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Gradient Descent solution for Linear Regression.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.
            lr: The learning rate.
            epochs: The number of epochs.
        """
        N, D = X.shape
        X = np.hstack((np.ones((N, 1)), X))
        self.w = np.zeros((D + 1,))

        for _ in range(epochs):
            self.w -= lr * self.compute_gradient(self.w, X, y)

    def compute_gradient(self, w, X, y):
        n = len(y)
        return -2 / n * X.T @ (y - X @ w)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        N, D = X.shape
        X = np.hstack((np.ones((N, 1)), X))
        return X @ self.w
