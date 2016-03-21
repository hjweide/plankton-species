#!/usr/bin/env python

import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import PIL
import theano
import theano.tensor as T
from PIL import Image
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import DenseLayer
from lasagne.layers import FeaturePoolLayer
from lasagne.layers import get_output
from lasagne.layers import get_all_params
from lasagne.layers import get_all_param_values
from lasagne.layers import get_all_layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import LeakyRectify
from lasagne.init import Orthogonal
from os.path import join
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from time import strftime
from time import time


class Model:
    def __init__(self, input_shape, classes, verbose=True):
        self.verbose = verbose
        self.input_shape = input_shape

        conv = {
            'pad': 'same',
            'filter_size': (3, 3),
            'stride': (1, 1),
            'nonlinearity': LeakyRectify(leakiness=1. / 3),
            'W': Orthogonal()
        }

        pool = {
            'pool_size': (3, 3),
            'stride': (2, 2),
            'mode': 'max',
        }

        dense = {
            'num_units': 1024,
            'W': Orthogonal(),
            'nonlinearity': LeakyRectify(leakiness=1. / 3),
        }

        # TODO: need to move the layer construction, this is unnecessary
        # if the layers are loaded from disk
        l_in = InputLayer(input_shape, name='in')

        l_conv1a = Conv2DLayer(l_in, name='c1a', num_filters=32, **conv)
        l_conv1b = Conv2DLayer(l_conv1a, name='c1b', num_filters=16, **conv)
        l_pool1 = Pool2DLayer(l_conv1b, name='p1', **pool)

        l_conv2a = Conv2DLayer(l_pool1, name='c2a', num_filters=64, **conv)
        l_conv2b = Conv2DLayer(l_conv2a, name='c2b', num_filters=32, **conv)
        l_pool2 = Pool2DLayer(l_conv2b, name='p2', **pool)

        l_conv3a = Conv2DLayer(l_pool2, name='c3a', num_filters=128, **conv)
        l_conv3b = Conv2DLayer(l_conv3a, name='c3b', num_filters=128, **conv)
        l_conv3c = Conv2DLayer(l_conv3b, name='c3c', num_filters=64, **conv)
        l_pool3 = Pool2DLayer(l_conv3c, name='p3', **pool)

        l_conv4a = Conv2DLayer(l_pool3, name='c4a', num_filters=256, **conv)
        l_conv4b = Conv2DLayer(l_conv4a, name='c4b', num_filters=256, **conv)
        l_conv4c = Conv2DLayer(l_conv4b, name='c4c', num_filters=128, **conv)
        l_pool4 = Pool2DLayer(l_conv4c, name='p4', **pool)

        l_drop5 = DropoutLayer(l_pool4, name='do5', p=0.5)
        l_dense5 = DenseLayer(l_drop5, name='d5', **dense)
        l_maxout5 = FeaturePoolLayer(l_dense5, name='mo5', pool_size=2)

        l_drop6 = DropoutLayer(l_maxout5, name='do5', p=0.5)
        l_dense6 = DenseLayer(l_drop6, name='d6', **dense)
        l_maxout6 = FeaturePoolLayer(l_dense6, name='mo6', pool_size=2)

        l_dropout7 = DropoutLayer(l_maxout6, name='do7', p=0.5)
        l_dense7 = DenseLayer(
            l_dropout7, name='out',
            num_units=len(classes), nonlinearity=softmax,
        )

        model_layers = get_all_layers(l_dense7)
        self.layers = {layer.name: layer for layer in model_layers}
        self.best_weights = get_all_param_values(self.layers['out'])

        self.encoder = LabelEncoder()
        self.encoder.fit(classes)

        self.x = T.tensor4('x')
        self.y = T.ivector('y')
        self.x_batch = T.tensor4('x_batch')
        self.y_batch = T.ivector('y_batch')

        self.history = {
            'start_time': '',
            'num_epochs': 0,
            'epoch_durations': [-1],
            'train_losses': [np.inf],
            'train_accuracies': [0.],
            'valid_losses': [np.inf],
            'valid_accuracies': [0.],
            'best_train_loss_epoch': 0,
            'best_valid_loss_epoch': 0,
            'best_train_accuracy_epoch': 0,
            'best_valid_accuracy_epoch': 0,
        }

    def save(self, filename):
        model_dict = {
            'layers': self.layers,
            'best_weights': self.best_weights,
            'train_mean': self.train_mean,
            'train_std': self.train_std,
            'encoder': self.encoder,
            'history': self.history,
        }

        if self.verbose:
            print('saving model to %s' % (filename))
        with open(filename, 'wb') as f:
            pickle.dump(model_dict, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        if self.verbose:
            print('loading model from %s' % (filename))
        with open(filename, 'rb') as f:
            model_dict = pickle.load(f)

        self.layers = model_dict['layers']
        self.best_weights = model_dict['best_weights']
        self.train_mean = model_dict['train_mean']
        self.train_std = model_dict['train_std']
        self.encoder = model_dict['encoder']
        self.history = model_dict['history']

        # TODO: do we want to initialize to the old or the best weights?
        src_params = model_dict['best_weights']
        dst_params = get_all_params(self.layers['out'])
        for src, dst in zip(src_params, dst_params):
            dst.set_value(src)

    def initialize_inference(self):
        self.infer_func = self.build_infer_func()

    def build_train_func(self, lr=0.01, mntm=0.9):
        y_hat = get_output(self.layers['out'], self.x, deterministic=False)
        train_loss = T.mean(
            T.nnet.categorical_crossentropy(y_hat, self.y)
        )
        train_acc = T.mean(
            T.eq(y_hat.argmax(axis=1), self.y)
        )

        all_params = get_all_params(self.layers['out'], trainable=True)
        updates = nesterov_momentum(
            train_loss, all_params, lr, mntm)

        train_func = theano.function(
            inputs=[theano.In(self.x_batch), theano.In(self.y_batch)],
            outputs=[train_loss, train_acc],
            updates=updates,
            givens={
                self.x: self.x_batch,
                self.y: self.y_batch,
            },
        )

        return train_func

    def initialize_training(self, lr, mntm, batch_size, max_epochs):
        if self.verbose:
            print('initializing training with:')
            print(' lr = %.2e, mntm=%.2e, batch_size = %d, max_epochs = %d' % (
                lr, mntm, batch_size, max_epochs))

        self.lr = lr
        self.mntm = mntm
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        if self.verbose:
            print('compiling theano function for training')
        self.train_func = self.build_train_func(lr, mntm)

        if self.verbose:
            print('compiling theano function for validation')

        self.valid_func = self.build_valid_func()

    def load_image(self, filename):
        img = Image.open(filename)
        img = img.resize(
            (self.input_shape[3], self.input_shape[2]), PIL.Image.ANTIALIAS)
        img = np.asarray(img, dtype=np.float32)

        if self.input_shape[1] == 3:
            img = img.transpose(1, 2, 0)

        return img

    def prepare_data(self, cache_files=None, filenames=None, labels=None):
        assert cache_files is not None or (filenames is not None and labels is not None), 'need either cache files or filenames and labels'

        # load the image files from disk
        if filenames is not None and labels is not None:
            targets = self.encoder.transform(labels)

            filenames, targets = shuffle(filenames, targets, random_state=42)

            skf = StratifiedKFold(targets, n_folds=(1. / 0.2))
            train_idx, valid_idx = next(iter(skf))

            X_train = np.empty(
                ((train_idx.shape[0],) + self.input_shape[1:]), dtype=np.float32)
            y_train = np.empty(
                train_idx.shape[0], dtype=np.int32)
            X_valid = np.empty(
                ((valid_idx.shape[0],) + self.input_shape[1:]), dtype=np.float32)
            y_valid = np.empty(
                valid_idx.shape[0], dtype=np.int32)

            for i, idx in enumerate(train_idx):
                X_train[i] = self.load_image(filenames[idx])
                y_train[i] = targets[idx]

            for i, idx in enumerate(valid_idx):
                X_valid[i] = self.load_image(filenames[idx])
                y_valid[i] = targets[idx]

            # write the numpy arrays to disk as a tuple
            if cache_files is not None:
                train_file, valid_file = cache_files
                with open(train_file, 'wb') as f:
                    pickle.dump((X_train, y_train), f, pickle.HIGHEST_PROTOCOL)
                with open(valid_file, 'wb') as f:
                    pickle.dump((X_valid, y_valid), f, pickle.HIGHEST_PROTOCOL)

        # load the image files directly from disk
        else:
            train_file, valid_file = cache_files
            with open(train_file, 'rb') as f:
                X_train, y_train = pickle.load(f)
            with open(valid_file, 'rb') as f:
                X_valid, y_valid = pickle.load(f)

        return X_train, y_train, X_valid, y_valid

    def transform(self, X, y):
        X_copy = X.copy()
        y_copy = y.copy()

        v_idx = np.random.choice([True, False], replace=True, size=X.shape[0])
        h_idx = np.random.choice([True, False], replace=True, size=X.shape[0])

        X_copy[v_idx] = X_copy[v_idx, :, ::-1, :]
        X_copy[h_idx] = X_copy[h_idx, :, :, ::-1]

        return X_copy, y_copy

    def start_training(
            self, X_train, y_train, X_valid, y_valid, filename=None):
        if self.verbose:
            print('Data shapes:')
            print(' X_train.shape = %r, X_valid.shape = %r' % (
                X_train.shape, X_valid.shape))
            print(' y_train.shape = %r, y_valid.shape = %r' % (
                y_train.shape, y_valid.shape))
            print('Before normalization:')
            print(' X_train.min() = %.2f, X_train.max() = %.2f' % (
                X_train.min(), X_train.max()))
            print(' X_valid.min() = %.2f, X_valid.max() = %.2f' % (
                X_valid.min(), X_valid.max()))

        X_train = X_train.astype(np.float32) / 255.
        X_valid = X_valid.astype(np.float32) / 255.
        y_train = y_train.astype(np.int32)
        y_valid = y_valid.astype(np.int32)

        self.train_mean = np.mean(X_train, axis=0)
        X_train -= self.train_mean
        X_valid -= self.train_mean

        self.train_std = np.std(X_train, axis=0)
        X_train /= self.train_std
        X_valid /= self.train_std

        if self.verbose:
            print('After normalization:')
            print(' X_train.min() = %.2f, '
                  ' X_train.max() = %.2f' % (
                      X_train.min(),
                      X_train.max()))
            print(' X_valid.min() = %.2f, '
                  ' X_valid.max() = %.2f' % (
                      X_valid.min(),
                      X_valid.max()))
            print(' np.abs(X_train.mean(axis=0).max()) = %.6f, '
                  ' np.abs(X_valid.mean(axis=0)).max() = %.6f' % (
                      np.abs(X_train.mean(axis=0)).max(),
                      np.abs(X_valid.mean(axis=0)).max()))

        num_batches_train = (
            X_train.shape[0] + self.batch_size - 1) / self.batch_size
        num_batches_valid = (
            X_valid.shape[0] + self.batch_size - 1) / self.batch_size

        self.history['start_time'] = '%s' % (strftime('%Y-%m-%d_%H:%M:%S'))
        if self.verbose:
            print('max_epochs = %d' % (self.max_epochs))
            print('num_batches_train = %d' % (num_batches_train))
            print('num_batches_valid = %d' % (num_batches_valid))

            print('starting training at %s' % (self.history['start_time']))

        try:
            for epoch in range(1, self.max_epochs + 1):
                t_epoch_start = time()
                batch_train_losses, batch_train_accuracies = [], []
                for i in xrange(num_batches_train):
                    train_idx = slice(
                        i * self.batch_size, (i + 1) * self.batch_size)
                    X_train_batch = X_train[train_idx]
                    y_train_batch = y_train[train_idx]

                    X_train_batch_transformed, y_train_batch_transformed =\
                        self.transform(X_train_batch, y_train_batch)

                    batch_train_loss, batch_train_acc = self.train_func(
                        X_train_batch_transformed, y_train_batch_transformed)

                    batch_train_losses.append(batch_train_loss)
                    batch_train_accuracies.append(batch_train_acc)

                batch_train_losses_mean = np.mean(batch_train_losses)
                batch_train_accuracies_mean = np.mean(batch_train_accuracies)
                self.history['train_losses'].append(
                    batch_train_losses_mean)
                self.history['train_accuracies'].append(
                    batch_train_accuracies_mean)

                batch_valid_losses, batch_valid_accuracies = [], []
                for i in xrange(num_batches_valid):
                    valid_idx = slice(
                        i * self.batch_size, (i + 1) * self.batch_size)
                    X_valid_batch = X_valid[valid_idx]
                    y_valid_batch = y_valid[valid_idx]

                    batch_valid_loss, batch_valid_acc = self.valid_func(
                        X_valid_batch, y_valid_batch)

                    batch_valid_losses.append(batch_valid_loss)
                    batch_valid_accuracies.append(batch_valid_acc)

                t_epoch_end = time()

                batch_valid_losses_mean = np.mean(batch_valid_losses)
                batch_valid_accuracies_mean = np.mean(batch_valid_accuracies)
                self.history['valid_losses'].append(
                    batch_valid_losses_mean)
                self.history['valid_accuracies'].append(
                    batch_valid_accuracies_mean)

                self.history['num_epochs'] = epoch
                self.history['epoch_durations'].append(t_epoch_end - t_epoch_start)

                # TODO: do something about these long lines
                best_train_loss_epoch = self.history['best_train_loss_epoch']
                best_valid_loss_epoch = self.history['best_valid_loss_epoch']
                best_train_accuracy_epoch = self.history['best_train_accuracy_epoch']
                best_valid_accuracy_epoch = self.history['best_valid_accuracy_epoch']

                current_train_loss = self.history['train_losses'][-1]
                best_train_loss = self.history['train_losses'][best_train_loss_epoch]

                current_train_accuracy = self.history['train_accuracies'][-1]
                best_train_accuracy = self.history['train_accuracies'][best_train_accuracy_epoch]

                current_valid_loss = self.history['valid_losses'][-1]
                best_valid_loss = self.history['valid_losses'][best_valid_loss_epoch]

                current_valid_accuracy = self.history['valid_accuracies'][-1]
                best_valid_accuracy = self.history['valid_accuracies'][best_valid_accuracy_epoch]

                if current_train_loss < best_train_loss:
                    self.history['best_train_loss_epoch'] = epoch
                if current_valid_loss < best_valid_loss:
                    self.history['best_valid_loss_epoch'] = epoch
                    self.best_weights = get_all_param_values(
                        self.layers['out'])

                if current_train_accuracy > best_train_accuracy:
                    self.history['best_train_accuracy_epoch'] = epoch
                if current_valid_accuracy > best_valid_accuracy:
                    self.history['best_valid_accuracy_epoch'] = epoch

                self.print_epoch_info()

        except KeyboardInterrupt:
            print('caught ctrl-c... stopped training.')
            self.print_training_summary()
            if filename is None:
                filename = '%s' % (strftime('%Y-%m-%d_%H:%M:%S'))
            self.save(filename)

    def print_epoch_info(self):
        current_epoch = self.history['num_epochs']
        train_loss = self.history['train_losses'][-1]
        valid_loss = self.history['valid_losses'][-1]
        train_accuracy = self.history['train_accuracies'][-1]
        valid_accuracy = self.history['valid_accuracies'][-1]
        duration = self.history['epoch_durations'][-1]

        if self.history['best_train_loss_epoch'] == current_epoch:
            train_epoch_color = ('\033[32m', '\033[0m')
        else:
            train_epoch_color = ('', '')

        if self.history['best_valid_loss_epoch'] == current_epoch:
            valid_epoch_color = ('\033[32m', '\033[0m')
        else:
            valid_epoch_color = ('', '')

        print('{:>4} | {}{:>10.6f}{} | {}{:>10.6f}{} | {:>3.2f}% | {:>3.2f}% |'
              ' {:>4.2f}s |'.format(
                  current_epoch,
                  train_epoch_color[0], train_loss, train_epoch_color[1],
                  valid_epoch_color[0], valid_loss, valid_epoch_color[1],
                  100 * train_accuracy, 100 * valid_accuracy, duration))

    def print_training_summary(self):
        train_losses = self.history['train_losses']
        valid_losses = self.history['valid_losses']
        train_accuracies = self.history['train_accuracies']
        valid_accuracies = self.history['valid_accuracies']

        best_train_loss_epoch = self.history['best_train_loss_epoch']
        best_valid_loss_epoch = self.history['best_valid_loss_epoch']
        best_train_accuracy_epoch = self.history['best_train_accuracy_epoch']
        best_valid_accuracy_epoch = self.history['best_valid_accuracy_epoch']
        print('training summary:')
        print('best training loss: %.6f at epoch %d' % (
            train_losses[best_train_loss_epoch], best_train_loss_epoch))
        print('best validation loss: %.6f at epoch %d' % (
            valid_losses[best_valid_loss_epoch], best_valid_loss_epoch))
        print('best training accuracy: %.2f%% at epoch %d' % (
            100 * train_accuracies[best_train_accuracy_epoch], best_train_accuracy_epoch))
        print('best validation accuracy: %.2f%% at epoch %d' % (
            100 * valid_accuracies[best_valid_accuracy_epoch], best_valid_accuracy_epoch))

        num_epochs = self.history['num_epochs']
        fig, ax1 = plt.subplots()

        # the 0th epoch is the dummy infinity loss
        epochs = range(1, num_epochs + 1)
        ax1.plot(epochs, train_losses[1:], 'b')
        ax1.plot(epochs, valid_losses[1:], 'g')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.set_xlim((1, num_epochs))

        ax2 = ax1.twinx()
        ax2.set_xlim((1, num_epochs))
        ax2.plot(epochs, train_accuracies[1:], 'b--')
        ax2.plot(epochs, valid_accuracies[1:], 'g--')
        ax2.set_ylabel('accuracy')

        ax1.legend(('training', 'validation'), loc='center right')

        train_val_log = join('logs', '%s.png' % self.history['start_time'] )
        plt.savefig(train_val_log, bbox_inches='tight')

    def build_valid_func(self):
        y_hat = get_output(self.layers['out'], self.x, deterministic=True)
        valid_loss = T.mean(
            T.nnet.categorical_crossentropy(y_hat, self.y)
        )
        valid_acc = T.mean(
            T.eq(y_hat.argmax(axis=1), self.y)
        )

        valid_func = theano.function(
            inputs=[theano.In(self.x_batch), theano.In(self.y_batch)],
            outputs=[valid_loss, valid_acc],
            givens={
                self.x: self.x_batch,
                self.y: self.y_batch,
            },
        )

        return valid_func

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

    def get_class_scores_filenames(self, filenames, species):
        num_samples = len(filenames)
        X = np.empty((num_samples,) + self.input_shape[1:], dtype=np.float32)
        for i, fname in enumerate(filenames):
            X[i] = self.load_image(fname)

        # swap the columns so they match what's expected
        # TODO: find a better way to do this...
        fixed_idx = self.encoder.transform(species)
        return self.get_class_scores(X)[:, fixed_idx]

    def get_class_scores(self, X):
        X_test = X.astype(np.float32) / 255.
        X_test -= self.train_mean
        X_test /= self.train_std
        return self.infer_func(X_test)


def test_initialization():

    # test code for development
    from os import listdir
    from os.path import join

    print('initializing model')
    model = Model((None, 1, 95, 95), 121)
    model.initialize_weights(join('models', 'weights_augmentation_maxout.pickle'))

    data_dir = 'test'
    all_files = listdir(data_dir)
    print('found %d files' % (len(all_files)))
    filenames = [join(data_dir, f) for f in all_files[:128]]

    np.set_printoptions(threshold=np.nan)
    print('getting class scores for %d images' % (len(filenames)))
    y_hat = model.get_class_scores_filenames(filenames)

    #print(y_hat)

    print('done')


def test_training():
    n_train, n_valid = 800, 200
    X_train = np.random.randint(0, 256, size=(n_train, 1, 95, 95))
    X_valid = np.random.randint(0, 256, size=(n_valid, 1, 95, 95))
    y_train = np.random.randint(0, 121, size=n_train)
    y_valid = np.random.randint(0, 121, size=n_valid)

    print('initializing model')
    model = Model((None, 1, 95, 95), 121)
    model.initialize_training(0.01, 0.9, 128, 10000)

    print('starting training')
    model.start_training(X_train, y_train, X_valid, y_valid)


if __name__ == '__main__':
    #test_initialization()
    test_training()
