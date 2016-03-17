#!/usr/bin/env python

import numpy as np
import PIL
import theano
import theano.tensor as T
from PIL import Image
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import get_output
from lasagne.nonlinearities import softmax, rectify


class Model:
    def __init__(self, input_shape, output_shape):
        self.weights_file = 'weights.pickle'  # TODO
        self.input_shape = input_shape
        l_in = InputLayer(
            input_shape, name='in')
        l_out = DenseLayer(
            l_in, num_units=output_shape, nonlinearity=softmax, name='out')

        self.layers = {l_in.name: l_in, l_out.name: l_out}

        self.x = T.tensor4('x')
        self.y = T.ivector('y')
        self.x_batch = T.tensor4('x_batch')
        self.y_batch = T.ivector('y_batch')

        self.infer_func = self.build_infer_func()

    def build_infer_func(self):
        y_hat = get_output(self.layers['out'], self.x, deterministic=True)
        infer_func = theano.function(
            inputs=[theano.In(self.x_batch)],
            outputs=y_hat,
            givens={
                self.x: self.x_batch,
            },
        )

        return infer_func

    def get_class_scores_filenames(self, filenames):
        num_samples = len(filenames)
        X = np.empty((num_samples,) + self.input_shape[1:], dtype=np.float32)
        _, c, h, w = self.input_shape
        for i, fname in enumerate(filenames):
            img = Image.open(fname)
            img = np.asarray(
                img.resize((w, h), PIL.Image.ANTIALIAS), dtype=np.float32)
            if c == 3:
                img = img.transpose(1, 2, 0)
            X[i] = img / 255.

        return self.get_class_scores(X)

    def get_class_scores(self, X):
        return self.infer_func(X)


def main():
    # test code for development
    from os import listdir
    from os.path import join
    data_dir = 'test'
    all_files = listdir(data_dir)[:128]
    print('found %d files' % (len(all_files)))
    filenames = [join(data_dir, f) for f in all_files[:128]]

    print('initializing model')
    np.set_printoptions(threshold=np.nan)
    model = Model((128, 1, 95, 95), 10)
    print('getting class scores for %d images' % (len(filenames)))
    y_hat = model.get_class_scores_filenames(filenames)
    print(y_hat)


if __name__ == '__main__':
    main()
