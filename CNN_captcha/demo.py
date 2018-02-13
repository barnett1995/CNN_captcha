import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from src.gen_model import create_layer,convert2gray,x_input
from src.config import MAX_CAPTCHA, CHAR_SET_LEN, VALIDATE_STRING
from src.gen_image import array_to_text
import matplotlib.pyplot as plt

keep_prob=tf.placeholder(tf.float32)


captcha_image = Image.open("./out.png")
captcha_source = np.array(captcha_image)
image = convert2gray(captcha_source)
#image = image.flatten() / 255
X = tf.cast(image, tf.float32)
Y = create_layer(X, keep_prob)
max_y = tf.argmax(tf.reshape(Y, [MAX_CAPTCHA, CHAR_SET_LEN]), 1)


def crick():
    saver=tf.train.Saver()
    with tf.Session() as sess:

        #ckpt=tf.train.get_checkpoint_state("./model/break.cpkt-30900")
        saver.restore(sess,"./model/break.ckpt-31700")

        rst = sess.run(max_y, feed_dict={keep_prob: 1})
        print(rst)
        result = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        n=0
        for i in range(MAX_CAPTCHA):
            result[n * CHAR_SET_LEN + i] = 1
            i += 1
        result = array_to_text(result)
        print(result)


        #print(" 预测: {}".format(predict_text))

crick()