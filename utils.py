import os
import dlib


predictor_path = './models/shape_predictor_5_face_landmarks.dat'


def detect_face(path_to_image):
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    img = dlib.load_rgb_image(path_to_image)
    dets = detector(img, 1)
    num_faces = len(dets)
    if num_faces == 0:
        return None
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img, detection))
    image = dlib.get_face_chip(img, faces[0])
    return image


def get_files_and_target(list_attr_celeba):
    with open(list_attr_celeba) as f:
        n = int(f.readline().strip())
        headers = f.readline().strip().split()
        index = headers.index('Eyeglasses') + 1
        files = []
        target = []
        for line in f:
            line = line.strip().split()
            files.append(line[0])
            target.append(int(line[index]))
    return n, files, target
