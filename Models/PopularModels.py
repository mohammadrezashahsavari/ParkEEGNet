from .classification_models_1D.tfkeras import Classifiers


'''
def Resnet18_Model(num_blocks_list=[2, 2, 2, 2], input_shape=(500, 12), n_output_nodes = 1):
    inputs = Input(shape=input_shape)
    num_filters = 64
    
    t = BatchNormalization()(inputs)
    t = Conv1D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block1d(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    t = AveragePooling1D(4)(t)

    t = Flatten()(t)

    t = Dense(256, activation='relu')(t)
    t = Dense(256, activation='relu')(t)
    t = Dense(n_output_nodes, activation='sigmoid' if n_output_nodes == 1 else 'softmax')(t)

    return Model(inputs, t)

def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block1d(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv1D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv1D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv1D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out



# MobileNet block
def mobilnet_block (t, filters, strides):
    t = DepthwiseConv1D(kernel_size = 3, strides = strides, padding = 'same')(t)
    t = BatchNormalization()(t)
    t = ReLU()(t)
    
    t = Conv1D(filters = filters, kernel_size = 1, strides = 1)(t)
    t = BatchNormalization()(t)
    t = ReLU()(t)
    
    return t

#stem of the model
def MobilNet_Model(input_shape, n_output_nodes=1):
    input = Input(shape = input_shape)
    t = Conv1D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(input)
    t = BatchNormalization()(t)
    t = ReLU()(t)
    # main part of the model
    t = mobilnet_block(t, filters = 64, strides = 1)
    t = mobilnet_block(t, filters = 128, strides = 2)
    t = mobilnet_block(t, filters = 128, strides = 1)
    t = mobilnet_block(t, filters = 256, strides = 2)
    t = mobilnet_block(t, filters = 256, strides = 1)
    t = mobilnet_block(t, filters = 512, strides = 2)
    for _ in range (5):
        t = mobilnet_block(t, filters = 512, strides = 1)

    t = mobilnet_block(t, filters = 1024, strides = 2)
    t = mobilnet_block(t, filters = 1024, strides = 1)
    t = AveragePooling1D(pool_size = 7, strides = 1, data_format='channels_first')(t)
    t = Flatten()(t)
    output = Dense (n_output_nodes, activation = 'sigmoid')(t)

    return Model(inputs=input, outputs=output)
'''

def VGG16_Model(input_shape, n_output_nodes=1):
    VGG16, preprocess_input = Classifiers.get('vgg16')
    model = VGG16(
        input_shape = input_shape,
        classes = n_output_nodes,
        include_top=True,
        weights = None,
        classifier_activation = 'sigmoid',
        stride_size=2
    )
    return model


def ResNet18_Model(input_shape, n_output_nodes=1):
    Resnet18, preprocess_input = Classifiers.get('resnet18')
    model = Resnet18(
        input_shape = input_shape,
        classes = n_output_nodes,
        include_top=True,
        weights = None,
        classifier_activation = 'sigmoid',
        stride_size=2
    )
    return model


def EfficientNet_Model(input_shape, n_output_nodes=1):
    EfficientNet, preprocess_input = Classifiers.get('EfficientNetB0')
    model = EfficientNet(
        input_shape = input_shape,
        classes = n_output_nodes,
        include_top=True,
        weights = None,
        classifier_activation = 'sigmoid',
        stride_size=2
    )
    return model

def MobileNet_Model(input_shape, n_output_nodes=1):
    MobileNet, preprocess_input = Classifiers.get('mobilenet')
    model = MobileNet(
        input_shape = input_shape,
        classes = n_output_nodes,
        include_top=True,
        weights = None,
        classifier_activation = 'sigmoid',
        stride_size=2
    )
    return model

def Inceptionv3_Model(input_shape, n_output_nodes=1):
    Inceptionv3, preprocess_input = Classifiers.get('inceptionv3')
    model = Inceptionv3(
        input_shape = input_shape,
        classes = n_output_nodes,
        include_top=True,
        weights = None,
        classifier_activation = 'sigmoid',
        stride_size=2
    )
    return model

def DenseNet_Model(input_shape, n_output_nodes=1):
    DenseNet, preprocess_input = Classifiers.get('densenet121')
    model = DenseNet(
        input_shape = input_shape,
        classes = n_output_nodes,
        include_top=True,
        weights = None,
        classifier_activation = 'sigmoid',
        stride_size=2
    )
    return model
