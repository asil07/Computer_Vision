import numpy as np


class Perceptron:
    def __init__(self, N, alpha=0.1):
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):

        X = np.c_[X, np.ones((X.shape[0]))]
        # print(f'X = {X}')
        for epoch in np.arange(0, epochs):
            # loops over each individual data point
            for (x, target) in zip(X, y):
                # print(f'(x, target) = {(x, target)}')
                # takes dot product between the input features and
                # the weight matrix, then passes this value through the step function
                # to obtain prediction
                p = self.step(np.dot(x, self.W))
                # print(f'p = {p}')
                if p != target:
                    error = p - target
                    # print(f'error = p - target = {error}')
                    self.W += -self.alpha * error * x
                    # print("W += -alpha * error * x = {}".format(self.W))

    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)

        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.W))




