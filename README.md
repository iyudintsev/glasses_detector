# Description

This deep learning model allows recognizing whether a person in the photo has glasses. 

## The model

As a baseline we could use [the model](https://www.researchgate.net/publication/320964354_Shallow_convolutional_neural_network_for_eyeglasses_detection_in_facial_images) based on [GoogleNet](https://arxiv.org/abs/1409.4842). However, we used the model that relies on [VGG16](https://arxiv.org/abs/1409.1556). This model was simplified for CPU computations.

## Data

At first, we use the model from `dlib` that allows obtaining landmarks. After that, we alighn and extract a face from a photo using the HOG and `dlib` as well.

[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) could be chosen as the data for this task. We developed a script for obtaining a balanced dataset containing persons with and without glasses (about 25000 photos). 

# How to use this code

1) In the folder `models`, you should run the script `get_shape_predictor.sh` to get the model from `dlib` for obtaining landmarks on a face.

** You can use the trained model directly. The detail is on the step 5**

2) Next, you should make the folder `data` in the main directory of the project (`mkdir data`). Download two files [`list_attr_celeba.txt`](https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs) and 
[`img_aligh_celeba.zip`](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg) into the `data` directory. Unzip the archive `unzip img_aligh_celeba.zip`.

3) After that, execute `python3 prepare_data.py` for obtaining data for training. This process can take some time. The result of this is training and test dataset in the folder `./data/images`.

4) At this step, you should execute `python3 run_model.py` for the model training.

5) If you want to use the trained model then you should execute `python3 main.py path/to/images`. The input is the folder with images and the output is a file `results.txt` with paths to the images containing persons with glasses.
