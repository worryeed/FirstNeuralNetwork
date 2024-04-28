import numpy as np


def relu(t):
    return np.maximum(0, t)


def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)


def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z


INPUT_DIM = 4
H_DIM = 5
OUTPUT_DIM = 3

x = np.random.randn(INPUT_DIM)

W1 = np.random.randn(INPUT_DIM, H_DIM)
b1 = np.random.randn(H_DIM)
W2 = np.random.randn(H_DIM, OUTPUT_DIM)
b2 = np.random.randn(OUTPUT_DIM)

probs = predict(x)
pred_class = np.argmax(probs)
class_names = ['Setosa', 'Versicolor', 'Virginica']
print('Predicted class: ', class_names[pred_class])