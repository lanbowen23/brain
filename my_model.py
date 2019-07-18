from keras.layers import Input, Lambda, Activation, concatenate, add
from keras.layers.convolutional import MaxPooling2D, Conv3D, Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Permute
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from keras.losses import categorical_crossentropy

K.set_image_dim_ordering("th")

channel_axis_5d = 2


def timeDist_DeepEM3D_Net(input, scale=True, size=256):
    # Input Shape is 256 x 256 x 3 (tf) or 3 x 256 x 256 (th)
    
    # make sure image size is multiple of 16
    size_m = K.cast(K.round(size / 16 + 0.5) * 16, 'int32')
    input = Permute([1, 3, 4, 2])(input)
    input = TimeDistributed(Lambda(lambda image: 
                K.tf.image.resize_images(image, (size_m, size_m))))(input)
    input = Permute([1, 4, 2, 3])(input)

    x1, z1 = time_inception_resnet_stem_3Dconv(input)
    x1 = time_inception_resnet_v2_A(x1, scale_residual=scale)
    x2 = time_reduction_A(x1, k=256, l=256, m=384, n=384)

    x2 = time_inception_resnet_v2_B(x2, scale_residual=scale)
    x3 = time_reduction_resnet_v2_B(x2)

    x3 = time_inception_resnet_v2_C(x3, scale_residual=scale)

    u1 = TimeDistributed(Conv2DTranspose(2, (5, 5), strides=(2, 2), padding='same'))(z1)
    u1 = BatchNormalization(axis=2)(u1)

    u2 = TimeDistributed(Conv2DTranspose(2, (9, 9), strides=(4, 4), padding='same'))(x1)
    u2 = BatchNormalization(axis=2)(u2)

    u3 = TimeDistributed(Conv2DTranspose(2, (17, 17), strides=(8, 8), padding='same'))(x2)
    u3 = BatchNormalization(axis=2)(u3)

    u4 = TimeDistributed(Conv2DTranspose(2, (33, 33), strides=(16, 16), padding='same'))(x3)
    u4 = BatchNormalization(axis=2)(u4)

    merged = add([u1, u2, u3, u4])
    out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(merged)

    out = Permute([1, 3, 4, 2])(out)
    out = TimeDistributed(Lambda(lambda image: 
                K.tf.image.resize_images(image, (size, size))))(out)
    out = Permute([1, 4, 2, 3])(out)
    return out


def time_inception_resnet_stem_3Dconv(input):
    # Input Shape is 320 x 320 x 3 (tf) or 3 x 320 x 320 (th)
    # for original our biomedical paper, we used 2 layer of 3D convolution followed by 2D conv
    # here we do 2 layer 3D conv, but keep the times (original #slices in Z direction) unchanged.

    # First, permute time 5d time to the order : [ch,height,width,time(Z)] for 3D convolution
    c = Permute([2, 3, 4, 1])(input)
    # [h, w, time, c] for tensorflow
    # c = Permute([3, 4, 1, 2])(input)
    c = Conv3D(32, (3, 3, 3), activation='relu', strides=(2, 2, 1), padding='same')(c)
    # next_size= 160#input[1]/2
    c = Conv3D(64, (3, 3, 3), activation='relu', strides=(1, 1, 1), padding='same')(c)
    # Now, for rest of time distributed 2D convs, we need Permute data back to dimension order [time,channel,height,
    # width]
    c = Permute([4, 1, 2, 3])(c)

    c = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(c)
    # --------------  left branch -------------------------------------------
    c1 = TimeDistributed(Conv2D(64, (1, 1), activation='relu', padding='same'))(c)
    c1 = TimeDistributed(Conv2D(96, (3, 3), activation='relu', padding='same'))(c1)
    # --------------- right branch ------------------------------------------
    c2 = TimeDistributed(Conv2D(64, (1, 1), activation='relu', padding='same'))(c)
    c2 = TimeDistributed(Conv2D(64, (7, 1), activation='relu', padding='same'))(c2)
    c2 = TimeDistributed(Conv2D(64, (1, 7), activation='relu', padding='same'))(c2)
    c2 = TimeDistributed(Conv2D(96, (3, 3), activation='relu', padding='same'))(c2)
    m1 = concatenate([c1, c2], axis=channel_axis_5d)
    # m_pad =ZeroPadding2D((1,1))(m1)
    p1 = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))(m1)
    p2 = TimeDistributed(Conv2D(192, (3, 3), activation='relu', strides=(2, 2),
                                padding='same'))(m1)
    m2 = concatenate([p1, p2], axis=channel_axis_5d)
    m2 = BatchNormalization(axis=2)(m2)  # channel axis
    m2 = Activation('relu')(m2)
    return m2, m1


def time_inception_resnet_v2_A(input, scale_residual=True):
    # Input is relu activation
    init = input

    ir1 = TimeDistributed(Conv2D(32, (1, 1), activation='relu', padding='same'))(input)

    ir2 = TimeDistributed(Conv2D(32, (1, 1), activation='relu', padding='same'))(input)
    ir2 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(ir2)

    ir3 = TimeDistributed(Conv2D(32, (1, 1), activation='relu', padding='same'))(input)
    ir3 = TimeDistributed(Conv2D(48, (3, 3), activation='relu', padding='same'))(ir3)
    ir3 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(ir3)

    ir_merge = concatenate([ir1, ir2, ir3], axis=channel_axis_5d)

    ir_conv = TimeDistributed(Conv2D(384, (1, 1), activation='linear', padding='same'))(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = BatchNormalization(axis=2)(out)
    out = Activation("relu")(out)
    return out


def time_reduction_A(input, k=192, l=224, m=256, n=384):
    # r1 = TimeDistributed(Conv2D(384, 3, 3, activation='relu', strides=(2,2),padding='same'))(input)
    r1 = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))(input)
    r2 = TimeDistributed(Conv2D(n, (3, 3), activation='relu', strides=(2, 2), padding='same'))(input)

    r3 = TimeDistributed(Conv2D(k, (1, 1), activation='relu', padding='same'))(input)
    r3 = TimeDistributed(Conv2D(l, (3, 3), activation='relu', padding='same'))(r3)
    # m_pad =ZeroPadding2D((1,1))(r3)
    r3 = TimeDistributed(Conv2D(m, (3, 3), activation='relu', strides=(2, 2), padding='same'))(r3)

    m = concatenate([r1, r2, r3], axis=channel_axis_5d)
    m = BatchNormalization(axis=2)(m)
    m = Activation('relu')(m)
    return m


def time_inception_resnet_v2_B(input, scale_residual=True):
    # Input is relu activation
    init = input

    ir1 = TimeDistributed(Conv2D(192, (1, 1), activation='relu', padding='same'))(input)

    ir2 = TimeDistributed(Conv2D(128, (1, 1), activation='relu', padding='same'))(input)
    ir2 = TimeDistributed(Conv2D(160, (1, 7), activation='relu', padding='same'))(ir2)
    ir2 = TimeDistributed(Conv2D(192, (7, 1), activation='relu', padding='same'))(ir2)

    ir_merge = concatenate([ir1, ir2], axis=channel_axis_5d)

    ir_conv = TimeDistributed(Conv2D(1152, (1, 1), activation='linear', padding='same'))(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = BatchNormalization(axis=2)(out)
    out = Activation("relu")(out)
    return out


def time_reduction_resnet_v2_B(input):
    r1 = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))(input)
    # r1 = TimeDistributed(Conv2D(1152, 3, 3, activation='relu', strides=(2,2),padding='same'))(input)

    r2 = TimeDistributed(Conv2D(256, (1, 1), activation='relu', padding='same'))(input)
    r2 = TimeDistributed(Conv2D(384, (3, 3), activation='relu', strides=(2, 2), padding='same'))(r2)

    r3 = TimeDistributed(Conv2D(256, (1, 1), activation='relu', padding='same'))(input)
    r3 = TimeDistributed(Conv2D(288, (3, 3), activation='relu', strides=(2, 2), padding='same'))(r3)

    r4 = TimeDistributed(Conv2D(256, (1, 1), activation='relu', padding='same'))(input)
    r4 = TimeDistributed(Conv2D(288, (3, 3), activation='relu', padding='same'))(r4)
    r4 = TimeDistributed(Conv2D(320, (3, 3), activation='relu', strides=(2, 2), padding='same'))(r4)

    m = concatenate([r1, r2, r3, r4], axis=channel_axis_5d)
    m = BatchNormalization(axis=2)(m)
    m = Activation('relu')(m)
    return m


def time_inception_resnet_v2_C(input, scale_residual=True):
    # Input is relu activation
    init = input

    ir1 = TimeDistributed(Conv2D(192, (1, 1), activation='relu', padding='same'))(input)

    ir2 = TimeDistributed(Conv2D(192, (1, 1), activation='relu', padding='same'))(input)
    ir2 = TimeDistributed(Conv2D(224, (1, 3), activation='relu', padding='same'))(ir2)
    ir2 = TimeDistributed(Conv2D(256, (3, 1), activation='relu', padding='same'))(ir2)

    ir_merge = concatenate([ir1, ir2], axis=channel_axis_5d)

    ir_conv = TimeDistributed(Conv2D(2144, (1, 1), activation='linear', padding='same'))(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = BatchNormalization(axis=2)(out)
    out = Activation("relu")(out)
    return out


def time_dist_softmax(x):
    assert K.ndim(x) == 5  # K as backend
    # e = K.exp(x - K.max(x, axis=2, keepdims=True))
    e = K.exp(x)
    # it seems here already consider neuron-boundary ratio
    s = K.sum(e, axis=2, keepdims=True)
    return e / s


def time_dist_softmax_out_shape(input_shape):
    shape = list(input_shape)
    return tuple(shape)
