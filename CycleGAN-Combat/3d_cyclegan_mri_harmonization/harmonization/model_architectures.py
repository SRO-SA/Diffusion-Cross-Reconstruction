from tensorflow.keras.layers import Layer, Conv3D, Conv3DTranspose, LeakyReLU, ReLU, Concatenate, Activation, Lambda, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.math import reduce_mean, abs as tf_abs, tanh as tf_tanh
from tensorflow import where as tf_where, pad as tf_pad


class InstanceNormalization(Layer):
    
    def __init__(self, epsilon=1e-3):
        super().__init__(dtype='float32')
        self.epsilon = epsilon
        
    def build(self, batch_input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=batch_input_shape[-1:],
            initializer="ones",
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=batch_input_shape[-1:],
            initializer="zeros",
            trainable=True
        )
        self.axis = range(1, len(batch_input_shape)-1)
        super().build(batch_input_shape)
        
    def call(self, x):
        mean = reduce_mean(x, axis=self.axis, keepdims=True)
        dev = x-mean
        mean_abs_dev = reduce_mean(tf_abs(dev), axis=self.axis, keepdims=True)
        normalized = dev / (mean_abs_dev+self.epsilon)
        return self.scale*normalized + self.offset
    
    
def Generator(lrelu_slope=0.2, kernel_initializer='glorot_uniform'):
    
    def Downsample(n_filters, padding="same", inst_norm=True):
        model = Sequential([
            Conv3D(filters=n_filters, kernel_size=4, strides=2, padding=padding, use_bias=False, kernel_initializer=kernel_initializer)
        ])
        if inst_norm:
            model.add(InstanceNormalization())
        model.add(LeakyReLU(lrelu_slope))
        return model
    
    def Upsample(n_filters):
        return Sequential([
            Conv3DTranspose(filters=n_filters, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=kernel_initializer),
            InstanceNormalization(),
            ReLU()
        ])
    
    down_stack = [
        Downsample(64, padding='valid', inst_norm=False), # (bs, 96, 96, 96, 64)
        Downsample(128), # (bs, 48, 48, 48, 128)
        Downsample(256), # (bs, 24, 24, 24, 256)
        Downsample(512), # (bs, 12, 12, 12, 512)
        Downsample(512), # (bs, 6, 6, 6, 512),
        Downsample(512) # (bs, 3, 3, 3, 512)
    ]
    
    up_stack = [
        Upsample(512), # (bs, 6, 6, 6, 512)
        Upsample(512), # (bs, 12, 12, 12, 512)
        Upsample(256), # (bs, 24, 24, 24, 256)
        Upsample(128), # (bs, 48, 48, 48, 128)
        Upsample(64) # (bs, 96, 96, 96, 64)
    ]
    last = Sequential([
        Conv3DTranspose(filters=1, kernel_size=4, strides=2, padding="same", use_bias=True, kernel_initializer=kernel_initializer),
        Activation(lambda x: tf_where(x<0, tf_tanh(x), x), dtype="float32")
    ])
    
    inputs = Input(shape=(192,192,192,1))
    x = Lambda(lambda x: tf_pad(x, ((0,0),(1,1),(1,1),(1,1),(0,0)), 'CONSTANT', constant_values=-1))(inputs)

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])
        
    return Model(inputs=inputs, outputs=last(x))

def Discriminator(lrelu_slope=0.2, kernel_initializer='glorot_uniform'):
    
    def Downsample(n_filters, padding='same', inst_norm=True):
        model = Sequential([
            Conv3D(filters=n_filters, kernel_size=4, strides=2, padding=padding, use_bias=False, kernel_initializer=kernel_initializer)
        ])
        if inst_norm:
            model.add(InstanceNormalization())
        model.add(LeakyReLU(lrelu_slope))
        return model
    
    return Sequential([
        Lambda(lambda x: tf_pad(x, ((0,0),(1,1),(1,1),(1,1),(0,0)), 'CONSTANT', constant_values=-1), input_shape=(192,192,192,1)),
        Downsample(64, padding='valid', inst_norm=False),
        Downsample(128),
        Downsample(256),
        Conv3D(filters=1, kernel_size=3, strides=1, padding="same", use_bias=True, kernel_initializer=kernel_initializer),
        Activation('linear', dtype='float32')
    ])