from cnn_train import crack_captcha_cnn
from cnn_train import convert2gray
from cnn_train import vec2text
from cnn_train import MAX_CAPTCHA
from cnn_train import CHAR_SET_LEN
from cnn_train import keep_prob
from generated_captcha import cpt_image

import tensorflow as tf
import numpy as np

def crack_captcha(captcha_image):
	output = crack_captcha_cnn()

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint('.'))

		predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
		text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

		text = text_list[0].tolist()
		vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
		i = 0
		for n in text:
				vector[i*CHAR_SET_LEN + n] = 1
				i += 1
		return vec2text(vector)

if __name__ == '__main__':

	text, image = cpt_image()
	image = convert2gray(image) #生成一张新图
	image = image.flatten() / 255 # 将图片一维化
	predict_text = crack_captcha(image) #导入模型识别
	print("正确: {}  预测: {}".format(text, predict_text))
	#train_crack_captcha_cnn()