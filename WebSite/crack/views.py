import tensorflow as tf
from django.shortcuts import render
from django.shortcuts import HttpResponse
from cnn_captcha.test import *
from cnn_captcha.config import *
from cnn_captcha.gen_model import create_layer,convert2gray
import numpy as np
from PIL import Image


import os

# Create your views here.
va = []


# 申请占位副


def index(request):

    va = []
    # 请求方法为POST时，进行处理
    if request.method == "POST":
        # 获取上传的文件，如果没有文件，则默认为None
        myFile = request.FILES.get("file", None)
        name = myFile.name
        if not myFile:
            return HttpResponse("没有文件")
        destination = open(os.path.join("./test_img/", name), 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():  # 分块写入文件
            destination.write(chunk)
        destination.close()

        #
        #value = "".join(result)
        #va["value"] = value
        #result = crack_test()
        #

        vc = crack_test(name,keep_prob,X,Y,max_y)
        va.append(vc)
        #va='123'
    return render(request, "index.html",{"data":va})




