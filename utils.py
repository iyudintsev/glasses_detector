import os
import dlib


predictor_path = './models/shape_predictor_5_face_landmarks.dat'
cnn_detector_path = "./models/mmod_human_face_detector.dat"
detector = dlib.get_frontal_face_detector()
cnn_detector = None
sp = dlib.shape_predictor(predictor_path)


def get_dets(img, cnn_det):
    dets = detector(img, 1)
    if len(dets) == 0:
        if not cnn_det:
            return None
        else:
            global cnn_detector
            if cnn_detector is None:
                cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
            dets = cnn_detector(img, 1)
            if len(dets) == 0:
                return None
            rects = dlib.rectangles()
            rects.extend([d.rect for d in dets])
            dets = rects
    return dets


def detect_face(path_to_image, cnn_det=False):
    img = dlib.load_rgb_image(path_to_image)
    dets = get_dets(img, cnn_det)
    if dets is None:
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
