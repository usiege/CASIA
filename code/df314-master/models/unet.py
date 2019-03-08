from keras import Model
from keras.layers import BatchNormalization, Input, Add, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, concatenate, Dropout, Activation, UpSampling2D


def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def Conv_Block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = BatchActivate(x)
    return x


def Res_Block(block, filters=16, activation=True):
    x = BatchActivate(block)
    x = Conv_Block(x, filters, (3, 3))
    x = Conv_Block(x, filters, (3, 3), activation=False)
    x = Add()([x, block])
    if activation:
        x = BatchActivate(x)
    return x


def build_model(input_layer, start_neurons, num_classes=1):
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding='same')(input_layer)
    conv1 = Res_Block(conv1, start_neurons * 1)
    conv1 = Res_Block(conv1, start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding='same')(pool1)
    conv2 = Res_Block(conv2, start_neurons * 2)
    conv2 = Res_Block(conv2, start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding='same')(pool2)
    conv3 = Res_Block(conv3, start_neurons * 4)
    conv3 = Res_Block(conv3, start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding='same')(pool3)
    conv4 = Res_Block(conv4, start_neurons * 8)
    conv4 = Res_Block(conv4, start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding='same')(pool4)
    conv5 = Res_Block(conv5, start_neurons * 16)
    conv5 = Res_Block(conv5, start_neurons * 16, True)
    pool5 = MaxPooling2D((2, 2))(conv5)

    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding='same')(pool5)
    convm = Res_Block(convm, start_neurons * 32)
    convm = Res_Block(convm, start_neurons * 32, True)

    deconv5 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv5 = concatenate([deconv5, conv5])
    uconv5 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding='same')(uconv5)
    uconv5 = Res_Block(uconv5, start_neurons * 16)
    uconv5 = Res_Block(uconv5, start_neurons * 16, True)

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv5)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding='same')(uconv4)
    uconv4 = Res_Block(uconv4, start_neurons * 8)
    uconv4 = Res_Block(uconv4, start_neurons * 8, True)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding='same')(uconv3)
    uconv3 = Res_Block(uconv3, start_neurons * 4)
    uconv3 = Res_Block(uconv3, start_neurons * 4, True)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding='same')(uconv2)
    uconv2 = Res_Block(uconv2, start_neurons * 2)
    uconv2 = Res_Block(uconv2, start_neurons * 2, True)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding='same')(uconv1)
    uconv1 = Res_Block(uconv1, start_neurons * 1)
    uconv1 = Res_Block(uconv1, start_neurons * 1, True)

    output_layer = Conv2D(num_classes, (1, 1), padding="same", activation=None)(uconv1)
    if num_classes is not None and num_classes > 1:
        output_layer = Activation('softmax')(output_layer)
    else:
        output_layer = Activation('sigmoid')(output_layer)

    return output_layer


def create_model(w, h, channel, num_classes=1):
    input_layer = Input((w, h, channel))
    output_layer = build_model(input_layer, 16, num_classes=num_classes)
    return Model(input_layer, output_layer)
