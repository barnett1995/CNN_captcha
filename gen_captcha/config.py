#创建验证码使用的数据源
NUMBER = '0123456789'
CHAR_SMALL = 'abcdefghijklmnopqrstuvwxyz'
CHAR_BIG = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

#验证码最大位数
MAX_CAPTCHA = 6
#验证码数据集
VALIDATE_STRING = NUMBER + CHAR_SMALL #+ CHAR_BIG
CHAR_SET_LEN = len(VALIDATE_STRING) # 数字字母总长度，即所有可能的选项的数量

#图片大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
FONT_SIZE = 62

#准确率
MAX_ACCURACY = 0.99
