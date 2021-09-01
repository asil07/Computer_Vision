import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))
            # the last two layers are a special case where the input
            # connections need a bias term but the output does not
            w = np.random.randn(layers[-2] + 1, layers[-1])

    def __repr__(self):
        # this represents network architecture and returns string
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        """ sigmoid funksiyasi algoritmi: return 1.0 / (1 + np.exp(-x)) """
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        """algoritm: return x * (1 - x)"""
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        """X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print(f'[INFO] epoch={epoch + 1}, loss={loss}')"""
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print(f'[INFO] epoch={epoch + 1}, loss={loss}')

    def fit_partial(self, x, y):
        """Bu Algoritm har bir layer ning activatsika funksiyasi va weight orasidagi dot
        productni oladi va uni nonlinear bolgan bolgan aktivatsiya funkisyasi sigmoid
        ga otkazadi va keyingi layerga o'tkazadi"""
        A = [np.atleast_2d(x)]
        for layer in np.arange(0, len(self.W)):

            net = A[layer].dot(self.W[layer])

            out = self.sigmoid(net)

            A.append(out)

