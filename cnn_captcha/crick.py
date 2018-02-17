import tensorflow as tf
import numpy as np
from PIL import Image
from gen_model import create_layer,convert2gray
from config import MAX_CAPTCHA, CHAR_SET_LEN, VALIDATE_STRING,IMAGE_HEIGHT,IMAGE_WIDTH


keep_prob=tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = create_layer(X,keep_prob)
max_y=tf.argmax(tf.reshape(Y,[MAX_CAPTCHA,CHAR_SET_LEN]),1)

def crick_test(path):
    with tf.Session() as sess:
        saver=tf.train.Saver()
        #ckpt=tf.train.get_checkpoint_state("./model/break.cpkt-31700")
        saver.restore(sess, "/tmp/model/break.ckpt-33600")
        captcha_image = Image.open(path)
        captcha_source = np.array(captcha_image)
        image = convert2gray(captcha_source)
        image = image.flatten() / 255
        rst=sess.run(max_y,feed_dict={X:[image],keep_prob:1})
        # 把36个数值对应到10个数字与26个字母中
        result=[]
        for i in range(MAX_CAPTCHA):
            result.append(VALIDATE_STRING[rst[MAX_CAPTCHA-(MAX_CAPTCHA-i)]])
        print(result)
    #return (result)

if __name__ == '__main__':
    path ="/home/gxm/Documents/Git/CNN/gen_captcha/Image/1.png"
    crick_test(path)

