import os
from utils import detect_face, get_files_and_target
from PIL import Image


def detect_face_and_save(input_folder, output_folder, filename):
    output_path = os.path.join(output_folder, filename)
    if os.path.exists(output_path):
        return
    path_to_image = os.path.join(input_folder, filename)
    image_array = detect_face(path_to_image)
    if image_array is None:
        return 0
    image = Image.fromarray(image_array)
    image.save(output_path)
    return 1


def main():
    input_folder = './data/img_align_celeba/'
    if not os.path.exists(input_folder):
        print("please download CelebA into folder '%s'" % input_folder)
        return
    list_attr_celeba = './data/list_attr_celeba.txt'
    if not os.path.exists(list_attr_celeba):
        print("please download the file 'list_attr_celeba.txt' into folder './data'")
    output_folder = './data/images/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    n, files, target = get_files_and_target(list_attr_celeba)
    images_with_glasses = [files[i] for i in range(n) if target[i] == 1]
    images_without_glasses = [files[i] for i in range(n) if target[i] == -1]

    print('prepare images with glasses...')
    num_with = len(images_with_glasses)
    counter = 0
    for i, f in enumerate(images_with_glasses):
        result = detect_face_and_save(input_folder, output_folder, f)
        if result == 0:
            num_with -= 1
        else:
            counter += 1
        print ("%d/%d" % (counter, num_with), end="\r")
    print('number of images: %d' % num_with)
    print('done.')

    balance = 1
    num_without = num_with * balance
    counter = 0
    i = 0
    print('prepare images without glasses...')
    while counter < num_without:
        f = images_without_glasses[i]
        result = detect_face_and_save(input_folder, output_folder, f)
        counter += result
        print ("%d/%d" % (counter, num_without), end="\r")
        i += 1
    print('number of images: %d' % num_without)
    print('done.')
    print('total number of images: %d' % (num_without + num_with))


if __name__ == "__main__":
    main()
