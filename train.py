import numpy as np
from utils import *
import pickle
import argparse
from os.path import join
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--x_train_dir', default='.', help='Directory with train images')
parser.add_argument('--y_train_dir', default='.', help='Directory with train labels')
parser.add_argument('--model_output_dir', default='.', help='Directory with trained model')

args = parser.parse_args()

np.seterr(divide='raise', invalid='raise', over='raise', under='ignore')

images, y = load_dataset(join(args.x_train_dir, 'train-images.idx3-ubyte'),
                         join(args.y_train_dir, 'train-labels.idx1-ubyte'))

size = 60000
im_train, y_train, im_valid, y_valid = splitting(images[:size], y[:size])

basis = generate_pca_basis(im_train, y_train, k=30)

X_train_m = generate_moments(im_train, basis)
X_valid_m = generate_moments(im_valid, basis)

norms = normes(X_train_m)

X_train = add_bias(X_train_m, norms)
X_valid = add_bias(X_valid_m, norms)

sm = Softmax(X_train, y_train)
f1_best = -1
w_approx=None

for lamda_coef in reversed(np.logspace(-4, -2, 5)):
    w = sm.min_loss(lamda_coef, w_approx=w_approx, method='Newton-CG')
    w_approx = w
    y_pred_val = sm.predict(X_valid, w)
    f1 = 100*comp_f1(y_valid, y_pred_val)
    y_pred = sm.predict(X_train, w)
    f1_train = 100*comp_f1(y_train, y_pred)
    if f1 > f1_best:
        w_opt = w
    print(lamda_coef, f1, f1_train)

data = {
        'w_opt': w_opt,
        'basis': basis,
        'norms': norms
}

with open(join(args.model_output_dir, 'model.pickle'), 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

X_t_m = generate_moments(images, basis)
X_t = add_bias(X_t_m, norms)


y_pred = predict_all(X_t, w_opt)
print(classification_report(y, y_pred, digits=3))
