import tensorflow as tf
from tensorflow.keras.layers import Activation, AveragePooling2D, Conv2D, MaxPooling2D, Dropout, Input, Flatten, Dense, BatchNormalization
from tensorflow.keras.initializers import random_uniform, glorot_uniform
from tensorflow.keras.models import Model


class Model1():
    def __init__(self) -> None:
        pass

    def ann_block(self, X, nodes=1, drop_rate=0):
        X = Dense(nodes, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dropout(drop_rate)(X)
        return X

    def model(self, input_shape):
        X_input = Input(input_shape)

        X = self.ann_block(X_input, 256, drop_rate=.3)
        X = self.ann_block(X, 512, drop_rate=.4)
        X = self.ann_block(X, 1024, drop_rate=.5)
        X = self.ann_block(X, 256, drop_rate=.4)
        X = self.ann_block(X, 128, drop_rate=.2)
        X = self.ann_block(X, 256, drop_rate=.3)
        X = self.ann_block(X, 32, drop_rate=.1)
        Y = self.ann_block(X, 1, drop_rate=0)

        model = Model(inputs=X_input, outputs=Y)

        return model