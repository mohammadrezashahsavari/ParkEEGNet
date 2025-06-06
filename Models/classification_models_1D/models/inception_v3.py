"""Inception V3 1D model for Keras.

Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).

# Reference

- [Rethinking the Inception Architecture for Computer Vision](
    http://arxiv.org/abs/1512.00567) (CVPR 2016)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from .. import get_submodules_from_kwargs
from keras_applications import imagenet_utils

backend = None
layers = None
models = None
keras_utils = None

POOLING_SIZE = 3*3


def conv1d_bn(x,
              filters,
              kernel_size,
              padding='same',
              strides=1,
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 2

    kernel_size = kernel_size * kernel_size
    x = layers.Conv1D(
        filters, kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x


def InceptionV3(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        stride_size=4,
        kernel_size=9,
        **kwargs
):
    """Instantiates the Inception v3 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 2

    x = conv1d_bn(img_input, 32, 3, strides=stride_size, padding='valid')
    x = conv1d_bn(x, 32, 3, padding='valid')
    x = conv1d_bn(x, 64, 3)
    x = layers.MaxPooling1D(POOLING_SIZE, strides=stride_size)(x)

    x = conv1d_bn(x, 80, 1, padding='valid')
    x = conv1d_bn(x, 192, 3, padding='valid')
    x = layers.MaxPooling1D(POOLING_SIZE, strides=stride_size)(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv1d_bn(x, 64, 1)

    branch5x5 = conv1d_bn(x, 48, 1)
    branch5x5 = conv1d_bn(branch5x5, 64, 5)

    branch3x3dbl = conv1d_bn(x, 64, 1)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3)

    branch_pool = layers.AveragePooling1D(POOLING_SIZE,
                                          strides=1,
                                          padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 32, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv1d_bn(x, 64, 1)

    branch5x5 = conv1d_bn(x, 48, 1)
    branch5x5 = conv1d_bn(branch5x5, 64, 5)

    branch3x3dbl = conv1d_bn(x, 64, 1)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3)

    branch_pool = layers.AveragePooling1D(POOLING_SIZE,
                                          strides=1,
                                          padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 64, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv1d_bn(x, 64, 1)

    branch5x5 = conv1d_bn(x, 48, 1)
    branch5x5 = conv1d_bn(branch5x5, 64, 5)

    branch3x3dbl = conv1d_bn(x, 64, 1)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3)

    branch_pool = layers.AveragePooling1D(POOLING_SIZE,
                                          strides=1,
                                          padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 64, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv1d_bn(x, 384, 3, strides=stride_size, padding='valid')

    branch3x3dbl = conv1d_bn(x, 64, 1)
    branch3x3dbl = conv1d_bn(branch3x3dbl, 96, 3)
    branch3x3dbl = conv1d_bn(
        branch3x3dbl, 96, 3, strides=stride_size, padding='valid')

    branch_pool = layers.MaxPooling1D(POOLING_SIZE, strides=stride_size)(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv1d_bn(x, 192, 1)

    branch7x7 = conv1d_bn(x, 128, 1)
    branch7x7 = conv1d_bn(branch7x7, 128, 1)
    branch7x7 = conv1d_bn(branch7x7, 192, 7)

    branch7x7dbl = conv1d_bn(x, 128, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 128, 7)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 128, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 128, 7)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 1)

    branch_pool = layers.AveragePooling1D(POOLING_SIZE,
                                          strides=1,
                                          padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 192, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv1d_bn(x, 192, 1)

        branch7x7 = conv1d_bn(x, 160, 1)
        branch7x7 = conv1d_bn(branch7x7, 160, 1)
        branch7x7 = conv1d_bn(branch7x7, 192, 7)

        branch7x7dbl = conv1d_bn(x, 160, 1)
        branch7x7dbl = conv1d_bn(branch7x7dbl, 160, 7)
        branch7x7dbl = conv1d_bn(branch7x7dbl, 160, 1)
        branch7x7dbl = conv1d_bn(branch7x7dbl, 160, 7)
        branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 1)

        branch_pool = layers.AveragePooling1D(
            POOLING_SIZE, strides=1, padding='same')(x)
        branch_pool = conv1d_bn(branch_pool, 192, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv1d_bn(x, 192, 1)

    branch7x7 = conv1d_bn(x, 192, 1)
    branch7x7 = conv1d_bn(branch7x7, 192, 1)
    branch7x7 = conv1d_bn(branch7x7, 192, 7)

    branch7x7dbl = conv1d_bn(x, 192, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 7)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 1)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 7)
    branch7x7dbl = conv1d_bn(branch7x7dbl, 192, 11)

    branch_pool = layers.AveragePooling1D(POOLING_SIZE,
                                          strides=1,
                                          padding='same')(x)
    branch_pool = conv1d_bn(branch_pool, 192, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv1d_bn(x, 192, 1)
    branch3x3 = conv1d_bn(branch3x3, 320, 3,
                          strides=stride_size, padding='valid')

    branch7x7x3 = conv1d_bn(x, 192, 1)
    branch7x7x3 = conv1d_bn(branch7x7x3, 192, 1)
    branch7x7x3 = conv1d_bn(branch7x7x3, 192, 7)
    branch7x7x3 = conv1d_bn(
        branch7x7x3, 192, 3, strides=stride_size, padding='valid')

    branch_pool = layers.MaxPooling1D(POOLING_SIZE, strides=stride_size)(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv1d_bn(x, 320, 1)

        branch3x3 = conv1d_bn(x, 384, 1)
        branch3x3_1 = conv1d_bn(branch3x3, 384, 1)
        branch3x3_2 = conv1d_bn(branch3x3, 384, 3)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv1d_bn(x, 448, 1)
        branch3x3dbl = conv1d_bn(branch3x3dbl, 384, 3)
        branch3x3dbl_1 = conv1d_bn(branch3x3dbl, 384, 1)
        branch3x3dbl_2 = conv1d_bn(branch3x3dbl, 384, 3)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling1D(
            POOLING_SIZE, strides=1, padding='same')(x)
        branch_pool = conv1d_bn(branch_pool, 192, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='sigmoid', name='act_sigmoid')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='inception_v3')

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)
