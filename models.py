import tensorflow as tf
from tensorflow.keras.layers import Activation, AveragePooling2D, Conv2D, MaxPooling2D, Input, Flatten, Dense, BatchNormalization
from tensorflow.keras.initializers import random_uniform, glorot_uniform
from tensorflow.keras.models import Model


class Model1():
    def __init__(self) -> None:
        pass

    def __conv2d_block(self, X, filters, kernel_size, strides, training=True, initializer=glorot_uniform):
        X = Conv2D(filters, kernel_size, strides, kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization(axis=3)(X, training=training)
        X = Activation('relu')(X)
        return X
    
    def model(self, input_shape=(165, 150, 3), classes=4):
        X_input = Input(input_shape)
        X = self.__conv2d_block(X_input, 5, 3, (1,1))
        # X = self.__conv2d_block(X, 4,3,(1,1))
        X = self.__conv2d_block(X, 8,3,(1,1))
        # X = self.__conv2d_block(X, 8,3,(2,2))
        X = self.__conv2d_block(X, 16,5,(3,3))
        X = self.__conv2d_block(X, 16,5,(2,2))
        X = AveragePooling2D(pool_size=(2,2))(X)
        X = Flatten()(X)
        X = Dense(256, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)
        X = Dense(16, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)
        Y = Dense(1, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)

        model = Model(inputs=X_input, outputs=Y)

        return model