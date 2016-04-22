import daug

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import get_all_layers
from lasagne.layers import get_all_param_values
from lasagne.layers import get_all_params
from lasagne.layers import get_output
from lasagne.updates import nesterov_momentum
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from time import strftime, time

from os.path import isfile, join


class TaxonomicClassifier:
    def __init__(self):
        pass

    def compute_joint_probabilities(self, X):
        pass

    def compute_independent_probabilities(self, X):
        pass


class RankClassifier:
    def __init__(self, input_shape, classes, verbose=True):
        self.verbose = verbose
        self.input_shape = input_shape
        self.encoder = LabelEncoder()
        self.classes = classes
        self.encoder.fit(classes)

        if self.verbose:
            print('initializing model with input shape %r and %d outputs' % (
                input_shape, len(classes)))

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

    def assemble(self, assemble_func):
        self.x = T.tensor4('x')
        self.y = T.ivector('y')
        self.x_batch = T.tensor4('x_batch')
        self.y_batch = T.ivector('y_batch')

        l_out = assemble_func(self.input_shape, len(self.classes))
        model_layers = get_all_layers(l_out)
        self.layers = {layer.name: layer for layer in model_layers}
        self.best_weights = get_all_param_values(self.layers['out'])

    def _initialize_training_and_validation(self, lr, mntm):
        if self.verbose:
            print('initializing training with:')
            print(' lr = %.2e, mntm=%.2e' % (lr, mntm))

        # compile the theano function for one training batch
        if self.verbose:
            print('compiling theano function for training')

        y_hat_train = get_output(
            self.layers['out'], self.x, deterministic=False)
        train_loss = T.mean(
            T.nnet.categorical_crossentropy(y_hat_train, self.y)
        )
        train_acc = T.mean(
            T.eq(y_hat_train.argmax(axis=1), self.y)
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

        if self.verbose:
            print('successfully compiled theano function for training')
        # compile the theano function for one validation batch
        if self.verbose:
            print('compiling theano function for validation')

        y_hat_valid = get_output(
            self.layers['out'], self.x, deterministic=True)
        valid_loss = T.mean(
            T.nnet.categorical_crossentropy(y_hat_valid, self.y)
        )
        valid_acc = T.mean(
            T.eq(y_hat_valid.argmax(axis=1), self.y)
        )

        valid_func = theano.function(
            inputs=[theano.In(self.x_batch), theano.In(self.y_batch)],
            outputs=[valid_loss, valid_acc],
            givens={
                self.x: self.x_batch,
                self.y: self.y_batch,
            },
        )

        if self.verbose:
            print('successfully compiled theano function for validation')

        return train_func, valid_func

    def _initialize_inference(self):
        if self.verbose:
            print('compiling theano function for inference')

        y_hat_infer = get_output(
            self.layers['out'], self.x, deterministic=True)
        infer_func = theano.function(
            inputs=[theano.In(self.x_batch)],
            outputs=y_hat_infer,
            givens={
                self.x: self.x_batch,
            },
        )

        if self.verbose:
            print('successfully compiled theano function for inference')
        return infer_func

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

    def train(self, filenames, labels,
              batch_size=128, lr=0.01, mntm=0.9, max_epochs=10,
              weightsfile=None, cachefile=None):

        if not hasattr(self, 'infer_func'):
            self.train_func, self.valid_func =\
                self._initialize_training_and_validation(lr, mntm)

        if cachefile is not None and isfile(cachefile):
            if self.verbose:
                print('loading data from cache: %s' % (cachefile))
            with open(cachefile, 'rb') as f:
                data_dict = pickle.load(f)
            X_train = data_dict['X_train']
            y_train = data_dict['y_train']
            X_valid = data_dict['X_valid']
            y_valid = data_dict['y_valid']
        else:
            if self.verbose:
                print('loading data into memory')

            targets = self.encoder.transform(labels)
            filenames, targets = shuffle(filenames, targets, random_state=42)
            skf = StratifiedKFold(targets, n_folds=(1. / 0.2))
            train_idx, valid_idx = next(iter(skf))

            X_train = np.empty(
                ((train_idx.shape[0],) + self.input_shape[1:]),
                dtype=np.float32)
            y_train = np.empty(
                train_idx.shape[0], dtype=np.int32)

            X_valid = np.empty(
                ((valid_idx.shape[0],) + self.input_shape[1:]),
                dtype=np.float32)
            y_valid = np.empty(
                valid_idx.shape[0], dtype=np.int32)

            for i, idx in enumerate(train_idx):
                img = daug.load_image(
                    filenames[idx], resize_to=self.input_shape[2:4])
                X_train[i] = img.transpose(2, 0, 1)
                y_train[i] = targets[idx]

            for i, idx in enumerate(valid_idx):
                img = daug.load_image(
                    filenames[idx], resize_to=self.input_shape[2:4])
                X_valid[i] = img.transpose(2, 0, 1)
                y_valid[i] = targets[idx]

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

        if cachefile is not None and not isfile(cachefile):
            if self.verbose:
                print('saving data to cache: %s' % (cachefile))
            with open(cachefile, 'wb') as f:
                data_dict = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_valid': X_valid,
                    'y_valid': y_valid,
                }

                pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)

        num_batches_train = (
            X_train.shape[0] + batch_size - 1) / batch_size
        num_batches_valid = (
            X_valid.shape[0] + batch_size - 1) / batch_size
        self.history['start_time'] = '%s' % (strftime('%Y-%m-%d_%H:%M:%S'))

        if self.verbose:
            print('max_epochs = %d' % (max_epochs))
            print('num_batches_train = %d' % (num_batches_train))
            print('num_batches_valid = %d' % (num_batches_valid))

            print('starting training at %s' % (self.history['start_time']))

        try:
            for epoch in range(1, max_epochs + 1):
                t_epoch_start = time()
                batch_train_losses, batch_train_accuracies = [], []
                for i in xrange(num_batches_train):
                    train_idx = slice(
                        i * batch_size, (i + 1) * batch_size)
                    X_train_batch = X_train[train_idx]
                    y_train_batch = y_train[train_idx]

                    X_train_batch_transformed, y_train_batch_transformed =\
                        daug.transform(X_train_batch, y_train_batch)

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
                        i * batch_size, (i + 1) * batch_size)
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
                self.history['epoch_durations'].append(
                    t_epoch_end - t_epoch_start)

                best_train_loss_epoch =\
                    self.history['best_train_loss_epoch']
                best_valid_loss_epoch =\
                    self.history['best_valid_loss_epoch']
                best_train_accuracy_epoch =\
                    self.history['best_train_accuracy_epoch']
                best_valid_accuracy_epoch =\
                    self.history['best_valid_accuracy_epoch']

                current_train_loss =\
                    self.history['train_losses'][-1]
                best_train_loss =\
                    self.history['train_losses'][best_train_loss_epoch]

                current_train_accuracy =\
                    self.history['train_accuracies'][-1]
                best_train_accuracy =\
                    self.history['train_accuracies'][best_train_accuracy_epoch]

                current_valid_loss =\
                    self.history['valid_losses'][-1]
                best_valid_loss =\
                    self.history['valid_losses'][best_valid_loss_epoch]

                current_valid_accuracy =\
                    self.history['valid_accuracies'][-1]
                best_valid_accuracy =\
                    self.history['valid_accuracies'][best_valid_accuracy_epoch]

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

                self._print_epoch_info()

        except KeyboardInterrupt:
            print('caught ctrl-c... stopped training.')
        self._print_training_summary()
        if weightsfile is None:
            weightsfile = join(
                'models', '%s' % (strftime('%Y-%m-%d_%H:%M:%S')))
        self.save(weightsfile)

    def _print_epoch_info(self):
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

    def _print_training_summary(self):
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
            100 * train_accuracies[best_train_accuracy_epoch],
            best_train_accuracy_epoch))
        print('best validation accuracy: %.2f%% at epoch %d' % (
            100 * valid_accuracies[best_valid_accuracy_epoch],
            best_valid_accuracy_epoch))

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

    def predict(self, X, batch_size=128):
        assert X.shape[1:] == self.input_shape[1:], (
            'X.shape[1:] = %r does not match expected self.input_shape[1:]' % (
                X.shape, self.input_shape[1:]))
        assert hasattr(self, 'train_mean') and hasattr(self, 'train_std'), (
            'this model does not have a normalizing mean and std deviation')
        assert hasattr(self, 'layers'), (
            'the model needs to be assembled before predicting')

        if not X.dtype == np.float32:
            print('casting input data from %s to np.float32' % (
                X.dtype))
            X = X.astype(np.float32)

        if not hasattr(self, 'infer_func'):
            self.infer_func = self._initialize_inference()

        X -= self.train_mean
        X /= self.train_std

        num_batches_test = (
            X.shape[0] + batch_size - 1) / batch_size

        y_hat_batch_list = []
        for i in xrange(num_batches_test):
            test_idx = slice(
                i * batch_size, (i + 1) * batch_size)
            X_batch = X[test_idx]

            y_hat_batch = self.infer_func(X_batch)
            y_hat_batch_list.append(y_hat_batch)

        y_hat = np.vstack(y_hat_batch_list)

        return y_hat

    def predict_from_filenames(self, filenames, batch_size=128):
        X = np.empty(
            ((len(filenames),) + self.input_shape[1:]),
            dtype=np.float32)

        for i, fname in enumerate(filenames):
            img = daug.load_image(fname, resize_to=self.input_shape[2:4])
            X[i] = img.transpose(2, 0, 1)

        return self.predict(X, batch_size=batch_size)
