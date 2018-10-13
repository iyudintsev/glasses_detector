import os
import click
import numpy as np
from model import create_convnet
from utils import detect_face


@click.command()
@click.option('--path_to_images', help="path to images")
@click.option('--cnn_detector', help="available values: True, False", default=False, type=bool)
def main(path_to_images, cnn_detector):
    if not os.path.exists(path_to_images):
        print('incorrect path to images')
        return
    people_with_glasses = []

    input_shape = (150, 150, 3)
    model = create_convnet(input_shape)
    model.load_weights('./models/convnet.h5')
    print('loading model...')
    print('start detection')
    for f in os.listdir(path_to_images):
        f_full = os.path.join(path_to_images, f)
        image = detect_face(f_full, cnn_detector)
        if image is None:
            print("detector couldn't find a face: %s" % f)
            continue
        result = model.predict(np.array([image]), batch_size=16)
        if result[0] > 0.5:
            people_with_glasses.append(f_full)
    print('done!')
    print("detected %d image (people with glasses)" % len(people_with_glasses))
    people_with_glasses.sort()
    if people_with_glasses:
        print('\n'.join(people_with_glasses))
    with open('results.txt', 'w') as f:
        for path in people_with_glasses:
            f.write(path + '\n')


if __name__ == "__main__":
    main()
