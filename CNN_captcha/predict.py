"""
专门做预测的
"""
import time

import numpy as np
import tensorflow as tf

from src.config import MAX_CAPTCHA, CHAR_SET_LEN
from src.gen_model import x_input,keep_prob,create_layer
from src.gen_image import gen_random_captcha_image,convert2gray,array_to_text



def hack_function(sess, predict, captcha_image):
    """
    装载完成识别内容后，
    :param sess:
    :param predict:
    :param captcha_image:
    :return:
    """
    text_list = sess.run(predict, feed_dict={x_input: [captcha_image], keep_prob: 1})

    text = text_list[0].tolist()
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    return array_to_text(vector)


def batch_hack_captcha():
    """
    批量生成验证码，然后再批量进行识别
    :return:
    """

    # 定义预测计算图
    output = create_layer(x_input, keep_prob)
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]),1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # saver = tf.train.import_meta_graph(save_model + ".meta")
        saver.restore(sess,"./model/break.ckpt-31700")

        stime = time.time()

        task_cnt = 1000
        right_cnt = 0
        print(task_cnt)

        for i in range(1000):
            text, image = gen_random_captcha_image()
            print(text)
            image = convert2gray(image)
            image = image.flatten() / 255
            predict_text = hack_function(sess, predict, image)
            if text == predict_text:
                right_cnt += 1
            else:
                print("标记: {}  预测: {}".format(text, predict_text))
                pass
                # print("标记: {}  预测: {}".format(text, predict_text))




if __name__ == '__main__':
    batch_hack_captcha()
    print('end...')
