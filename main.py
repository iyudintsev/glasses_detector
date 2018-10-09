import os
import sys
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
    for f in sorted(os.listdir(path_to_images)):
        f = os.path.join(path_to_images, f)
        image = detect_face(f)
        if image is None:
            continue
        result = model.predict(np.array([image]), batch_size=16)
        if result[0] > 0.5:
            people_with_glasses.append(f)
    print("detected %d images" % len(people_with_glasses))
    if people_with_glasses:
        print('\n'.join(people_with_glasses))


if __name__ == "__main__":
    main()
