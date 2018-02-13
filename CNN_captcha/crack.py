import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from src.gen_model import create_layer,convert2gray
from src.config import MAX_CAPTCHA, CHAR_SET_LEN, VALIDATE_STRING

'''
'''
#FLAGS=tf.flags.FLAGS
#tf.flags.DEFINE_string("input","","./out.png")

keep_prob=tf.placeholder(tf.float32)
X=cv2.imread("1.png")
print(X)
#captcha_image = Image.open(X)
#captcha_source = np.array(captcha_image)
X=convert2gray(X)
X=tf.cast(X,tf.float32)
Y=create_layer(X,keep_prob)
max_y=tf.argmax(tf.reshape(Y,[MAX_CAPTCHA,CHAR_SET_LEN]),1)

with tf.Session() as sess:
    saver=tf.train.Saver()
    #ckpt=tf.train.get_checkpoint_state("./model/break.cpkt-31700")
    saver.restore(sess,"./model/break.ckpt-31700")
    rst=sess.run(max_y,feed_dict={keep_prob:1})
    print(rst)
    # 把36个数值对应到10个数字与26个字母中

    result=[]
    for i in range(MAX_CAPTCHA):
        result.append(VALIDATE_STRING[rst[MAX_CAPTCHA-(MAX_CAPTCHA-i)]])
    print(result)

