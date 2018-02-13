# 利用TensorFlow搭建CNN模型识别验证码
## 一.各文件,目录作用
1.src/model/       :存储训练模型
2.src/config.py    :配置参数
3.src/gen_image.py :生成验证码,图片转换
4.src/gen_model.py :cnn模型
5.train.py	   :调用cnn模型进行训练
6.test_iamge_generator.py :生成并保存一张测试图片
7.crack.py 	   :对生成图片进行测试

## 二.测试环境
### 1.硬件,OS
cpu:Intel Xeon E5 2640 V4 2.4GHZ 单核
Ram:1G
OS:ubuntu 16.04 LTS Server

### 2.软件
pyhon:3.5.2
#### python包版本
captcha:0.2.1  
numpy:1.13.3  
matplotlib:2.1.1  
tensorflow:1.4.1  
pilow:4.0  
opencv-python:3.4.0.12  
tensorbaoard:0.4.0rc3  
 
## 三.运行
1.运行生成模型  
    python3 train.py  
2.生成图片  
    python3 test_iamge_generator.py  
3.对图片进行识别  
    python3 track.py  

