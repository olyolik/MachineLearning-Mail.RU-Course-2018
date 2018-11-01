import numpy as np
from utils import *
import pickle
import argparse
from sklearn.metrics import classification_report
from os.path import join

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--x_test_dir', default='.', help='Directory with test images')
parser.add_argument('--y_test_dir', default='.', help='Directory with test labels')
parser.add_argument('--model_input_dir', default='.', help='Directory with trained model')

args = parser.parse_args()

images_test, y_test = load_dataset(join(args.x_test_dir, 't10k-images.idx3-ubyte'),
                                   join(args.y_test_dir, 't10k-labels.idx1-ubyte'))

with open(join(args.model_input_dir, 'model.pickle'), 'rb') as f:
    data = pickle.load(f)

w_opt = data['w_opt']
basis = data['basis']
norms = data['norms']


X_test_m = generate_moments(images_test, basis)
X_test = add_bias(X_test_m, norms)


y_pred = predict_all(X_test, w_opt)

print(classification_report(y_test, y_pred, digits=3))
