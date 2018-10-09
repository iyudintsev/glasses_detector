from keras.layers import Input, Conv2D, BatchNormalization, \
    MaxPooling2D, Flatten, Dense, Dropout, ZeroPadding2D
from keras.models import Model
from keras.optimizers import Adam


def create_conv_block(num_channels, add_conv_block=False):
    def conv_block(net):
        net = ZeroPadding2D()(net)
        net = Conv2D(num_channels, (3,3), activation='relu')(net)
        net = BatchNormalization()(net)
        net = ZeroPadding2D()(net)
        net = Conv2D(num_channels, (3,3), activation='relu')(net)
        net = BatchNormalization()(net)
        if add_conv_block:
            net = ZeroPadding2D()(net)
            net = Conv2D(num_channels, (3,3), activation='relu')(net)
            net = BatchNormalization()(net)
        net = MaxPooling2D((2,2))(net)
        return net
    return conv_block


def create_fully_connected_block(neurons):
    def fully_connected_block(net):
        net = Dense(neurons, activation='relu')(net)
        net = Dropout(0.5)(net)
        return net
    return fully_connected_block


def create_convnet(input_shape):
    input_image = Input(shape=input_shape)
    net = create_conv_block(8)(input_image)
    net = create_conv_block(16)(net)
    net = create_conv_block(32, add_conv_block=True)(net)
    net = create_conv_block(64, add_conv_block=True)(net)
    net = Flatten()(net)
    net = create_fully_connected_block(64)(net)
    net = create_fully_connected_block(16)(net)
    net = create_fully_connected_block(16)(net)
    output = Dense(1, activation='sigmoid')(net)
    model = Model(input_image, output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(5e-3, decay=5e-4), metrics=['accuracy'])
    return model


def main():
    model = create_convnet((150, 150, 3))
    model.summary()


if __name__ == "__main__":
    main()
