from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import DenseLayer
from lasagne.layers import FeaturePoolLayer
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import LeakyRectify
from lasagne.init import Orthogonal


def build(input_shape, num_outputs):
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
        num_units=num_outputs, nonlinearity=softmax,
    )

    return l_dense7
