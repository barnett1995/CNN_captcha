
from django.shortcuts import render
from django.shortcuts import HttpResponse
from cnn_captcha.test import *

from PIL import Image
import os




# Create your views here.

va = []


'''
def index(request):
    if request.method == "POST":  # 请求方法为POST时，进行处理
        myFile = request.FILES.get("file", None)  # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return HttpResponse("没有文件")
        destination = open(os.path.join("./test_img/", myFile.name), 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():  # 分块写入文件
            destination.write(chunk)
        destination.close()

        #
        #value = "".join(result)
        #va["value"] = value
        #result = crack_test()
        va.append(vc)
    return render(request, "index.html",{"data":va})
'''
def index(request):

    va = []

    if request.method == "POST":  # 请求方法为POST时，进行处理
        myFile = request.FILES.get("file", None)  # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return HttpResponse("没有文件")
        destination = open(os.path.join("./test_img/", myFile.name), 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():  # 分块写入文件
            destination.write(chunk)
        destination.close()

        #
        #value = "".join(result)
        #va["value"] = value
        #result = crack_test()
        #

        vc = crack_test()
        va.append(vc)
    return render(request, "index.html",{"data":va})




