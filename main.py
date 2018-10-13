import os
import sys
import time
import numpy as np
from model import create_convnet
from utils import detect_face


def main():
    if len(sys.argv) != 2:
        print('incorrect format of execution')
        return
    path_to_images = sys.argv[1]
    if not os.path.exists(path_to_images):
        print('incorrect path to images')
        return
    people_with_glasses = []

    input_shape = (150, 150, 3)
    model = create_convnet(input_shape)
    model.load_weights('./models/convnet.h5')
    print('loading model...')
    print('start detection')
    t1 = time.time()
    for f in os.listdir(path_to_images):
        f_full = os.path.join(path_to_images, f)
        image = detect_face(f_full)
        if image is None:
            print("detector couldn't find a face: %s" % f)
            continue
        result = model.predict(np.array([image]), batch_size=16)
        if result[0] > 0.5:
            people_with_glasses.append(f_full)
    t2 = time.time()
    print('done!')
    print('time: {}'.format(t2 - t1))
    print("detected %d image (people with glasses)" % len(people_with_glasses))
    people_with_glasses.sort()
    if people_with_glasses:
        print('\n'.join(people_with_glasses))
    with open('results.txt', 'w') as f:
        for path in people_with_glasses:
            f.write(path + '\n')


if __name__ == "__main__":
    main()
