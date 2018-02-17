# 利用TensorFlow搭建CNN模型识别验证码
## 一.各文件,目录作用
1.Website:是用于展示验证码识别效果的web应用,基于Django web框架开发
2.cnn_captcha: CNN模型与测试程序
3.gen_captcha: 生成测试图片

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
### Website
1.部署网站
2.使用gen_captcha目录中的程序生成验证码
3.访问网站进行测试(只可以用gen_captcha生成的验证码)
### cnn_captcha
1.运行生成模型  
```
python3 gen_model.py
```
2.生成图片  
```
python3 test_iamge_generator.py
```
3.对图片进行识别  
```
python3 track.py  
```

## 才疏学浅,结构,代码有很多不完善的地方,还望各位大佬批评指点.
