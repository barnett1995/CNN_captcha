import tensorflow as tf
import numpy as np
from PIL import Image
from cnn_captcha.gen_model import create_layer,convert2gray
from cnn_captcha.config import MAX_CAPTCHA, CHAR_SET_LEN, VALIDATE_STRING,IMAGE_HEIGHT,IMAGE_WIDTH



#利用模型识别验证码图片
def crack_test():
    #设置keep_prob参数
    keep_prob = tf.placeholder(tf.float32)
    #申请占位副
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    Y = create_layer(X, keep_prob)
    max_y = tf.argmax(tf.reshape(Y, [MAX_CAPTCHA, CHAR_SET_LEN]), 1)
    with tf.Session() as sess:
        #读取模型
        saver=tf.train.Saver()
        #ckpt=tf.train.get_checkpoint_state("./model/break.cpkt-31700")
        saver.restore(sess, "/home/gxm/model/break.ckpt-33600")
        #获取验证码图片
        captcha_image = Image.open("/home/gxm/Documents/Git/CNN/WebSite/test_img/1.png")
        #图片转二维数组
        captcha_source = np.array(captcha_image)
        #降维转灰度图像
        image = convert2gray(captcha_source)

        image = image.flatten() / 255
        #将图片利用模型进行运算
        rst=sess.run(max_y,feed_dict={X:[image],keep_prob:1})
        # 把36个数值对应到10个数字与26个字母中
        result=[]
        for i in range(MAX_CAPTCHA):
            result.append(VALIDATE_STRING[rst[MAX_CAPTCHA-(MAX_CAPTCHA-i)]])
            value = "".join(result)
        #print(value)
        return (value)


#单元测试模块
if __name__ == '__main__':

    crack_test()

