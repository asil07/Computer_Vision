from pyimagesearch.nn.perceptron import Perceptron
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print("[info] training perceptron ...")
p = Perceptron(X.shape[1], alpha=0.001)
p.fit(X, y, epochs=20)

print("[info] testing ...")
for (x, target) in zip(X, y):
    pred = p.predict(x)
    print(f'[INFO] data={x}, ground-truth={target[0]}, pred={pred}')


