from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
import cnn_captcha.config as config
from PIL import Image
import random

char_dict = {}
number_dict = {}


# 生成随机的指定的字符串
def __gen_random_captcha_text(char_set=config.VALIDATE_STRING, size=None):
    # 验证是否为字符串
    if not char_set or not isinstance(char_set, str):
        raise ValueError('get the empty char_set')

    #random生成随机字符串
    result = list(char_set)
    random.shuffle(result)

    # 返回字符串
    return ''.join(result[0:size])

#随机生成验证码图像
def gen_random_captcha_image():
    #定义图片大小
    image = ImageCaptcha(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT,font_sizes=[config.FONT_SIZE])
    #获取随机生成的字符串
    text = __gen_random_captcha_text(size=config.MAX_CAPTCHA)
    #将字符串加入图片
    captcha = image.generate(text)
    captcha_image = Image.open(captcha)
    #生成验证码源
    captcha_source = np.array(captcha_image)
    return text, captcha_source


# 检查生成的图像宽度,高度是否符合预设值
def gen_require_captcha_image():
    while True:
        text, image = gen_random_captcha_image()
        if image.shape == (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3):
            #返回字符串,图片
            return text, image


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用,对于抽取特征也没啥用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


#准备字符串索引
def prepare_char_dict():
    if char_dict:
        return char_dict

    for index, val in enumerate(config.VALIDATE_STRING):
        char_dict[val] = index

    return char_dict

#准备数字索引
def prepare_number_dict():
    if number_dict:
        return number_dict

    for index, val in enumerate(config.VALIDATE_STRING):
        number_dict[index] = val

    return number_dict

#文本转向量
def text_to_array(text):
    char_dict_tmp = prepare_char_dict()

    arr = np.zeros(config.MAX_CAPTCHA * config.CHAR_SET_LEN, dtype=np.int8)
    for i, p in enumerate(text):
        key_index = char_dict_tmp[p]
        index = i * config.CHAR_SET_LEN + key_index
        arr[index] = 1

    return arr

#向量转文本
def array_to_text(arr):
    num_dict_tmp = prepare_number_dict()
    text = []
    char_pos = arr.nonzero()[0]
    for index, val in enumerate(char_pos):
        if index == 0:
            index = 1
        key_index = val % (index * config.CHAR_SET_LEN)
        text.append(num_dict_tmp[key_index])
    return ''.join(text)

#显示图片信息
def show_image_text():
    text, image = gen_random_captcha_image()

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)

    plt.show()

def creat_image():
    image = ImageCaptcha(width=160, height=60, font_sizes=[35])
    # 获取随机生成的字符串
    text = __gen_random_captcha_text(size=config.MAX_CAPTCHA)
    # 将字符串加入图片

    image.write(text,"../1.png")
#单元测试模块
if __name__ == '__main__':
    # __do_image_text()
    # arr = text_to_array('0142')
    # print '==========='
    # print array_to_text(arr)
    #show_image_text()
    #prepare_number_dict()
    creat_image()
