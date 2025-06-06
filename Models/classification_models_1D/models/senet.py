import os
import collections

from keras_applications import imagenet_utils

from Models.classification_models_1D import get_submodules_from_kwargs
from ._common_blocks import GroupConv1D, ChannelSE
from ..weights import load_model_weights

backend = None
layers = None
models = None
keras_utils = None

ModelParams = collections.namedtuple(
    'ModelParams',
    ['model_name', 'residual_block', 'groups', 'reduction', 'input_3x3', 'dropout']
)

# -------------------------------------------------------------------------
#   Helpers functions
# -------------------------------------------------------------------------

def get_bn_params(**params):
    axis = 2 if backend.image_data_format() == 'channels_last' else 1
    default_bn_params = {
        'axis': axis,
        'epsilon': 9.999999747378752e-06,
    }
    default_bn_params.update(params)
    return default_bn_params


def get_num_channels(tensor):
    channels_axis = 2 if backend.image_data_format() == 'channels_last' else 1
    return backend.int_shape(tensor)[channels_axis]


# -------------------------------------------------------------------------
#   Residual blocks
# -------------------------------------------------------------------------

def SEResNetBottleneck(
        filters,
        reduction=16,
        strides=1,
        kernel_size=9,
        **kwargs
):
    bn_params = get_bn_params()

    def layer(input_tensor):
        x = input_tensor
        residual = input_tensor

        # bottleneck
        x = layers.Conv1D(filters // 4, 1, kernel_initializer='he_uniform',
                          strides=strides, use_bias=False)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.ZeroPadding1D(kernel_size // 2)(x)
        x = layers.Conv1D(filters // 4, kernel_size,
                          kernel_initializer='he_uniform', use_bias=False)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv1D(filters, 1, kernel_initializer='he_uniform', use_bias=False)(x)
        x = layers.BatchNormalization(**bn_params)(x)

        #  if number of filters or spatial dimensions changed
        #  make same manipulations with residual connection
        x_channels = get_num_channels(x)
        r_channels = get_num_channels(residual)

        if strides != 1 or x_channels != r_channels:
            residual = layers.Conv1D(x_channels, 1, strides=strides,
                                     kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = layers.BatchNormalization(**bn_params)(residual)

        # apply attention module
        x = ChannelSE(reduction=reduction, **kwargs)(x)

        # add residual connection
        x = layers.Add()([x, residual])

        x = layers.Activation('relu')(x)

        return x

    return layer


def SEResNeXtBottleneck(
        filters,
        reduction=16,
        strides=1,
        kernel_size=9,
        groups=32,
        base_width=4,
        **kwargs
):
    bn_params = get_bn_params()

    def layer(input_tensor):
        x = input_tensor
        residual = input_tensor

        width = (filters // 4) * base_width * groups // 64

        # bottleneck
        x = layers.Conv1D(width, 1, kernel_initializer='he_uniform', use_bias=False)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.ZeroPadding1D(kernel_size // 2)(x)
        x = GroupConv1D(width, kernel_size, strides=strides, groups=groups,
                        kernel_initializer='he_uniform', use_bias=False, **kwargs)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv1D(filters, 1, kernel_initializer='he_uniform', use_bias=False)(x)
        x = layers.BatchNormalization(**bn_params)(x)

        #  if number of filters or spatial dimensions changed
        #  make same manipulations with residual connection
        x_channels = get_num_channels(x)
        r_channels = get_num_channels(residual)

        if strides != 1 or x_channels != r_channels:
            residual = layers.Conv1D(x_channels, 1, strides=strides,
                                     kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = layers.BatchNormalization(**bn_params)(residual)

        # apply attention module
        x = ChannelSE(reduction=reduction, **kwargs)(x)

        # add residual connection
        x = layers.Add()([x, residual])

        x = layers.Activation('relu')(x)

        return x

    return layer


def SEBottleneck(
        filters,
        reduction=16,
        strides=1,
        kernel_size=9,
        groups=64,
        is_first=False,
        **kwargs
):
    bn_params = get_bn_params()
    modules_kwargs = ({k: v for k, v in kwargs.items()
                       if k in ('backend', 'layers', 'models', 'utils')})

    if is_first:
        downsample_kernel_size = 1
        padding = False
    else:
        downsample_kernel_size = kernel_size
        padding = True

    def layer(input_tensor):

        x = input_tensor
        residual = input_tensor

        # bottleneck
        x = layers.Conv1D(filters // 2, 1, kernel_initializer='he_uniform', use_bias=False)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.ZeroPadding1D(kernel_size // 2)(x)
        x = GroupConv1D(filters, kernel_size, strides=strides, groups=groups,
                        kernel_initializer='he_uniform', use_bias=False, **kwargs)(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv1D(filters, 1, kernel_initializer='he_uniform', use_bias=False)(x)
        x = layers.BatchNormalization(**bn_params)(x)

        #  if number of filters or spatial dimensions changed
        #  make same manipulations with residual connection
        x_channels = get_num_channels(x)
        r_channels = get_num_channels(residual)

        if strides != 1 or x_channels != r_channels:
            if padding:
                residual = layers.ZeroPadding1D(downsample_kernel_size // 2)(residual)
            residual = layers.Conv1D(x_channels, downsample_kernel_size, strides=strides,
                                     kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = layers.BatchNormalization(**bn_params)(residual)

        # apply attention module
        x = ChannelSE(reduction=reduction, **kwargs)(x)

        # add residual connection
        x = layers.Add()([x, residual])

        x = layers.Activation('relu')(x)

        return x

    return layer


# -------------------------------------------------------------------------
#   SeNet builder
# -------------------------------------------------------------------------


def SENet(
        model_params,
        input_tensor=None,
        input_shape=None,
        include_top=True,
        classes=1000,
        weights='imagenet',
        stride_size=4,
        kernel_size=9,
        init_filters=64,
        first_kernel_size=49,
        repetitions=None,
        **kwargs
):
    """Instantiates the ResNet, SEResNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Args:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    residual_block = model_params.residual_block
    bn_params = get_bn_params()

    # if stride_size is scalar make it tuple of length 5
    if type(stride_size) not in (tuple, list):
        stride_size = (stride_size, stride_size, stride_size, stride_size, stride_size)

    if len(stride_size) < 3:
        print('Error: stride_size length must be 3 or more')
        return None

    if len(stride_size) - 1 != len(repetitions):
        print('Error: stride_size length must be equal to repetitions length - 1')
        return None

    # define input
    if input_tensor is None:
        input = layers.Input(shape=input_shape, name='input')
    else:
        if not backend.is_keras_tensor(input_tensor):
            input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            input = input_tensor

    x = input

    if model_params.input_3x3:

        x = layers.ZeroPadding1D(kernel_size // 2)(x)
        x = layers.Conv1D(init_filters, kernel_size, strides=stride_size[0],
                          use_bias=False, kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.ZeroPadding1D(kernel_size // 2)(x)
        x = layers.Conv1D(init_filters, kernel_size, use_bias=False,
                          kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

        x = layers.ZeroPadding1D(kernel_size // 2)(x)
        x = layers.Conv1D(init_filters * 2, kernel_size, use_bias=False,
                          kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

    else:
        x = layers.ZeroPadding1D(first_kernel_size // 2)(x)
        x = layers.Conv1D(init_filters, first_kernel_size, strides=stride_size[0], use_bias=False,
                          kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization(**bn_params)(x)
        x = layers.Activation('relu')(x)

    x = layers.ZeroPadding1D((stride_size[1] + 1) // 2)(x)
    x = layers.MaxPooling1D((stride_size[1] + 1), strides=stride_size[1])(x)

    # body of resnet
    filters = init_filters * 2
    stride_count = 2
    for i, stage in enumerate(repetitions):

        # increase number of filters with each stage
        filters *= 2

        for j in range(stage):

            # decrease spatial dimensions for each stage (except first, because we have maxpool before)
            if i == 0 and j == 0:
                x = residual_block(
                    filters,
                    reduction=model_params.reduction,
                    strides=1,
                    kernel_size=kernel_size,
                    groups=model_params.groups,
                    is_first=True,
                    **kwargs
                )(x)

            elif i != 0 and j == 0:
                x = residual_block(
                    filters,
                    reduction=model_params.reduction,
                    strides=stride_size[stride_count],
                    kernel_size=kernel_size,
                    groups=model_params.groups,
                    **kwargs
                )(x)
                stride_count += 1
            else:
                x = residual_block(
                    filters,
                    reduction=model_params.reduction,
                    strides=1,
                    kernel_size=kernel_size,
                    groups=model_params.groups,
                    **kwargs
                )(x)

    if include_top:
        x = layers.GlobalAveragePooling1D()(x)
        if model_params.dropout is not None:
            x = layers.Dropout(model_params.dropout)(x)
        x = layers.Dense(classes)(x)
        x = layers.Activation('softmax', name='output')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = input

    model = models.Model(inputs, x)

    if weights:
        if type(weights) == str and os.path.exists(weights):
            model.load_weights(weights)
        else:
            load_model_weights(
                model,
                model_params.model_name,
                weights,
                classes,
                include_top,
                kernel_size,
                input_shape[-1],
                **kwargs
            )

    return model


# -------------------------------------------------------------------------
#   SE Residual Models
# -------------------------------------------------------------------------

MODELS_PARAMS = {
    'seresnet50': ModelParams(
        'seresnet50',
        residual_block=SEResNetBottleneck,
        groups=1,
        reduction=16,
        input_3x3=False,
        dropout=None,
    ),

    'seresnet101': ModelParams(
        'seresnet101',
        residual_block=SEResNetBottleneck,
        groups=1,
        reduction=16,
        input_3x3=False,
        dropout=None,
    ),

    'seresnet152': ModelParams(
        'seresnet152',
        residual_block=SEResNetBottleneck,
        groups=1,
        reduction=16,
        input_3x3=False,
        dropout=None,
    ),

    'seresnext50': ModelParams(
        'seresnext50',
        residual_block=SEResNeXtBottleneck,
        groups=32,
        reduction=16,
        input_3x3=False,
        dropout=None,
    ),

    'seresnext101': ModelParams(
        'seresnext101',
        residual_block=SEResNeXtBottleneck,
        groups=32,
        reduction=16,
        input_3x3=False,
        dropout=None,
    ),

    'senet154': ModelParams(
        'senet154',
        residual_block=SEBottleneck,
        groups=64,
        reduction=16,
        input_3x3=True,
        dropout=0.2,
    ),
}


def SEResNet50(
        input_shape=None,
        input_tensor=None,
        weights=None,
        classes=1000,
        include_top=True,
        stride_size=4,
        kernel_size=9,
        init_filters=64,
        repetitions=(3, 4, 6, 3),
        **kwargs
):
    return SENet(
        MODELS_PARAMS['seresnet50'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        stride_size=stride_size,
        kernel_size=kernel_size,
        init_filters=init_filters,
        repetitions=repetitions,
        **kwargs
    )


def SEResNet101(
        input_shape=None,
        input_tensor=None,
        weights=None,
        classes=1000,
        include_top=True,
        stride_size=4,
        kernel_size=9,
        init_filters=64,
        repetitions=(3, 4, 23, 3),
        **kwargs
):
    return SENet(
        MODELS_PARAMS['seresnet101'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        stride_size=stride_size,
        kernel_size=kernel_size,
        init_filters=init_filters,
        repetitions=repetitions,
        **kwargs
    )


def SEResNet152(
        input_shape=None,
        input_tensor=None,
        weights=None,
        classes=1000,
        include_top=True,
        stride_size=4,
        kernel_size=9,
        init_filters=64,
        repetitions=(3, 8, 36, 3),
        **kwargs
):
    return SENet(
        MODELS_PARAMS['seresnet152'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        stride_size=stride_size,
        kernel_size=kernel_size,
        init_filters=init_filters,
        repetitions=repetitions,
        **kwargs
    )


def SEResNeXt50(
        input_shape=None,
        input_tensor=None,
        weights=None,
        classes=1000,
        include_top=True,
        stride_size=4,
        kernel_size=9,
        init_filters=64,
        repetitions=(3, 4, 6, 3),
        **kwargs
):
    return SENet(
        MODELS_PARAMS['seresnext50'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        stride_size=stride_size,
        kernel_size=kernel_size,
        init_filters=init_filters,
        repetitions=repetitions,
        **kwargs
    )


def SEResNeXt101(
        input_shape=None,
        input_tensor=None,
        weights=None,
        classes=1000,
        include_top=True,
        stride_size=4,
        kernel_size=9,
        init_filters=64,
        repetitions=(3, 4, 23, 3),
        **kwargs
):
    return SENet(
        MODELS_PARAMS['seresnext101'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        stride_size=stride_size,
        kernel_size=kernel_size,
        init_filters=init_filters,
        repetitions=repetitions,
        **kwargs
    )


def SENet154(
        input_shape=None,
        input_tensor=None,
        weights=None,
        classes=1000,
        include_top=True,
        stride_size=4,
        kernel_size=9,
        init_filters=64,
        repetitions=(3, 8, 36, 3),
        **kwargs
):
    return SENet(
        MODELS_PARAMS['senet154'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        stride_size=stride_size,
        kernel_size=kernel_size,
        init_filters=init_filters,
        repetitions=repetitions,
        **kwargs
    )


def preprocess_input(x, **kwargs):
    return imagenet_utils.preprocess_input(x, mode='torch', **kwargs)


setattr(SEResNet50, '__doc__', SENet.__doc__)
setattr(SEResNet101, '__doc__', SENet.__doc__)
setattr(SEResNet152, '__doc__', SENet.__doc__)
setattr(SEResNeXt50, '__doc__', SENet.__doc__)
setattr(SEResNeXt101, '__doc__', SENet.__doc__)
setattr(SENet154, '__doc__', SENet.__doc__)
