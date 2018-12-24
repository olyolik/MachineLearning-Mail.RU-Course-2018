from struct import unpack
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from scipy.optimize import minimize

def load_dataset(images_filename, labels_filename):
    with open(images_filename, 'rb') as im:
        images = im.read()
    with open(labels_filename, 'rb') as l:
        labels = l.read()
    images_magic, *images_shape = unpack('>IIII', images[:16])
    labels_magic, labels_shape = unpack('>II', labels[:8])
    assert images_magic == 2051
    assert labels_magic == 2049

    images_raw = np.frombuffer(images[16:], dtype=np.uint8).reshape(images_shape)
    labels_raw = np.frombuffer(labels[8:], dtype=np.uint8).reshape(labels_shape)
    return images_raw, labels_raw

# Thanks Maria Tarasevich for a nice idea to use PCA
def generate_pca_basis(images, labels, k=40):
    numbers = 1. * images
    avg = np.mean(numbers, axis=(1, 2))
    numbers -= avg.reshape(-1, 1, 1)
    U, Sigma, V = svds(numbers.reshape(-1, 28**2), k=k)
    return V.reshape(-1, 28, 28)

def generate_moments(images, basis):
    moments = []
    for k in range(len(basis)):
        value = np.einsum('ijk,jk', images, basis[k]) / 28**2
        moments.append(value)
    for i in range(len(basis)):
        for j in range(i+1):
            moments.append(moments[i] * moments[j])
    ans = np.transpose(moments)
    return ans

def normes(X_orig):
    return np.amax(X_orig, axis=0)

def add_bias(X_orig, norms):
    X_orig = X_orig / norms
    bias = np.ones((X_orig.shape[0], 1), dtype=np.double)
    return np.hstack((bias, X_orig))

def splitting(X,y):
    n = X.shape[0]
    perm = np.random.RandomState(seed=42).permutation(n)
    X = X[perm]
    y = y[perm]
    n_train = int(0.75 * n)
    n_valid = n

    X_train = X[:n_train, :]
    X_valid = X[n_train:, :]

    y_train = y[:n_train]
    y_valid = y[n_train:]

    return X_train, y_train, X_valid, y_valid

def comp_f1(y_true, y_pred):
    C = np.zeros((10, 10), dtype=np.int)
    for i in range(10):
        for j in range(10):
            C[i, j] = np.dot(np.int_(y_true==i), np.int_(y_pred==j))

    f1_all = []
    for i in range(10):
        prec = C[i, i] / (np.sum(C[i, :]) + 1e-20)
        recall = C[i, i] / (np.sum(C[:, i]) + 1e-20)
        f1_all.append(2 * prec * recall / (prec + recall + 1e-20))
    return np.mean(f1_all)

class Softmax:
    def __init__(self, X, y, labels=None):
        if labels is not None:
            num_classes = len(labels)
        else:
            num_classes = np.max(y) + 1
            labels = list(range(num_classes))
        Y = np.empty((len(y), num_classes), dtype=int)
        for i in range(num_classes):
            Y[:, i] = y == labels[i]
        self.X = X
        self.Y = Y


    def model(self, W, X=None):
        if X is None:
            X = self.X
        Z = np.exp(np.dot(X, W))
        return Z/Z.sum(axis=1).reshape(-1, 1)


    def predict(self, X, W):
        P = self.model(W, X=X)
        return P.argmax(axis=1)


    def loss(self, W, lambda_coef):
        logP = np.log(self.model(W))
        return -np.sum(self.Y * logP)/logP.shape[0] + lambda_coef * np.linalg.norm(W) **2


    def gradient_analitic(self, W, lam):
        P = self.model(W)
        return np.dot(self.X.T, P-self.Y)/P.shape[0] + 2 * lam * W


    def descent(self, lambda_coef, alpha = None, delta=1e-6, max_it=400):
        if alpha is None:
            alpha = self.fast_step
        grad_function = self.gradient_analitic
        W_old = np.zeros((self.X.shape[1], self.Y.shape[1]))
        for i in range(max_it):
            G = grad_function(W_old, lambda_coef)
            if isinstance(alpha, (int, float)):
                W_new = W_old - alpha*G
            else:
                W_new = W_old - alpha(W_old, lambda_coef, G)*G
            J_new = self.loss(W_new, lambda_coef)
            # print(i, J_new)
            W_diff = np.linalg.norm(W_new - W_old)
            if W_diff <= delta:
                break
            W_old = W_new
        return W_new


    def fast_step(self, W, lambda_coef, G):
        def alpha_loss(alpha):
            return self.loss(W - alpha[0]*G, lambda_coef)
        alpha_opt = minimize(alpha_loss, [0]).x
        return alpha_opt

    def min_loss(self, lambda_coef, w_approx=None, method='Newton-CG'):
        W_shape = (self.X.shape[1], self.Y.shape[1])
        def wrapped_loss(w):
            lv = self.loss(w.reshape(W_shape), lambda_coef)
            return lv
        def wrapped_jac(w):
            return self.gradient_analitic(w.reshape(W_shape), lambda_coef).reshape(-1)
        if w_approx is None:
            w_approx = np.zeros(np.prod(W_shape))
        if lambda_coef > 3.5e-5:
            w_opt = minimize(wrapped_loss,
                             x0=w_approx,
                             jac=wrapped_jac,
                             method=method,
                             options={'disp': False, 'maxiter': 2000}
                            ).x
        else:
            w_opt = self.descent(lambda_coef, max_it=1000)

        return w_opt.reshape(W_shape)

def predict_all(X, wall):
    a = np.argmax(X.dot(wall), axis=1)
    return a