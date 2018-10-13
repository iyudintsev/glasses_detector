import os
import json
import requests
import telebot
import wget
import numpy as np
from model import create_convnet
from PIL import Image
import tensorflow as tf
from utils import detect_face


graph = tf.get_default_graph()
input_shape = (150, 150, 3)
print("loading model...")
model = create_convnet(input_shape)
model.load_weights('./models/convnet.h5')
print("done!")


counter = len(os.listdir('./photo'))
with open('token.txt') as f:
    token = f.readline().strip()

prefix = "https://api.telegram.org/bot%s/" % token
prefix_file = "https://api.telegram.org/file/bot%s/" % token


bot = telebot.TeleBot(token)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Hello! Please, upload your photo.")


def get_image(file_id):
    global counter
    url = prefix + 'getFile?file_id={}'.format(file_id)
    r = requests.get(url)
    if r.status_code != 200:
        return None
    file_path = json.loads(r.content)['result']['file_path']
    url = prefix_file + file_path
    path = "./photo/%d.jpg" % counter
    counter += 1
    wget.download(url, path, bar=False)
    return path


@bot.message_handler(content_types=['photo'])
def process_photo(message):
    file_id = message.photo[-1].file_id
    path_to_image = get_image(file_id)
    if path_to_image is None:
        bot.reply_to(message, "Please, try again later.")
        return
    image = detect_face(path_to_image)
    if image is None:
        bot.reply_to(message, "I couldn't find a face.")
        print("request, status -1")
        return
    with graph.as_default():
        result = model.predict(np.array([image]), batch_size=16)
    if result > 0.5:
        bot.reply_to(message, "I see glasses :)")
        status = 1
    else:
        bot.reply_to(message, "There are no glasses in the photo.")
        status = 0
    print("request, status %d" % status)


if __name__ == "__main__":
    print("the application is running")
    bot.polling(none_stop=True)
