#coding:utf-8

from captcha.image import ImageCaptcha
#from captcha.image import ImageCaptcha
from PIL import Image			#图像处理标准库
import time,random,os			#random随机数字生成器
import matplotlib.pyplot as plt		#matplotlib绘图库
import numpy as np			#python numpy数学库

#验证码大小写
SMALL_LETTER = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
CAPITAL_LETTER = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
NUMBER = ['0','1','2','3','4','5','6','7','8','9']

#随机生成验证码数组
def cpt_text():
	#设置验证码数字字母集
	captcha_set = CAPITAL_LETTER + SMALL_LETTER + NUMBER
	#验证码长度
	captcha_size = 4
	#生成验证码的空字符串
	captcha_text = []
	#循环,乱序填入captcha_text
	for i in range(captcha_size):
		#random()函数:随机选择 
		#choice() 方法返回一个列表，元组或字符串的随机项
		a = random.choice(captcha_set)
		#向captcha_text字符串添加
		captcha_text.append(a)
	return captcha_text

#生成验证码图片
def cpt_image():
	#将cpt_text()返回值赋值
	captcha_text = cpt_text()
	#字符串连接生成新的字符串
	captcha_text = ''.join(captcha_text)
	#captcha_text.append(captcha_text)
	
	image = ImageCaptcha()
	captcha = image.generate(captcha_text)
	#image.open方法:打开图片
	captcha_image = Image.open(captcha)
	#np.array生成数组
	captcha_image = np.array(captcha_image)
	return captcha_text, captcha_image

	
#单元测试
if __name__=='__main__':
	#获取cpy_image()返回值
	image,text=cpt_image()
	#time.ctime()生成时间戳
	print (time.ctime())
	# matplotlib.pyplot.figure创建图
	fi = plt.figure()
	#matplotlib.pyplot.figure.add_subplot 添加子图
	#将画布分割成1行1列，图像画在从左到右从上到下的第1块
	ax = fi.add_subplot(111)

	#添加文本,figtext()
	#坐标x=0.2,y=0.8  ha,va:et_verticalalignment(align)设置垂直对齐
	ax.text(0.2, 0.8,text, ha='center', va='center')
	#显示图片
	plt.imshow(image)
	plt.show()

