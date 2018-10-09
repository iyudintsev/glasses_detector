import os
import click
import numpy as np
from utils import get_files_and_target
from model import create_convnet
from PIL import Image
from keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime


def get_data():
    path_to_images = './data/images/'
    if not os.path.exists(path_to_images):
        print("please prepare data using 'prepare_data.py'")
        return
    images = os.listdir(path_to_images)
    images.sort()
    list_attr_celeba = './data/list_attr_celeba.txt'
    if not os.path.exists(list_attr_celeba):
        print("please download the file 'list_attr_celeba.txt' into folder './data'")

    _, files, target = get_files_and_target(list_attr_celeba)
    mapping = dict(zip(files, target))
    target = [mapping[filename] for filename in images]
    target = [1 if y == 1 else 0 for y in target]
    images = [os.path.join(path_to_images, im) for im in images]
    np.random.seed(42)
    data = list(zip(images, target))
    np.random.shuffle(data)
    images, target = zip(*data)
    return len(images), images, target


def get_image(path):
    return np.array(Image.open(path))


def create_generator(X, y, steps, batch_size):
    while True:
        for i in range(steps):
            X_batch = X[i*batch_size: (i+1)*batch_size]
            X_batch = list(map(get_image, X_batch))
            y_batch = y[i*batch_size: (i+1)*batch_size]
            yield np.array(X_batch), np.array(y_batch)


@click.command()
@click.option('--option', help="available options: train, validate")
def main(option):
    if option is None:
        option = "train"
    n, images, target = get_data()
    n_train = 4 * n // 5
    n_test = n - n_train
    X_train, X_test = images[:n_train], images[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]
    input_shape = (150, 150, 3)
    model = create_convnet(input_shape)
    batch_size = 16
    steps_per_epoch = int(round(n_train / batch_size))
    validation_steps = int(round(n_test / batch_size))
    generator = create_generator(X_train, y_train, steps_per_epoch, batch_size)
    validation_data = create_generator(X_test, y_test, validation_steps, batch_size)
    tensorboard = TensorBoard("./log/%s" % str(datetime.now()).split('.')[0])
    modelcheckpoint = ModelCheckpoint('./models/convnet.h5', monitor='val_acc', verbose=0,
                                      save_best_only=True, save_weights_only=True, mode='max')
    if option == "train":
        model.load_weights('./models/convnet-0.h5')
        model.fit_generator(generator, 
                            steps_per_epoch=steps_per_epoch,
                            epochs=20,
                            callbacks=[tensorboard, modelcheckpoint],
                            validation_data=validation_data,
                            validation_steps=validation_steps,)
    else:
        model.load_weights('./models/convnet.h5')
        loss, acc = model.evaluate_generator(validation_data, steps=validation_steps)
        print("loss: {0:.5f}".format(loss))
        print("accuracy: {0:.5f}".format(acc))


if __name__ == "__main__":
    main()
