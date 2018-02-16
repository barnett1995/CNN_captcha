
from django.shortcuts import render
from django.shortcuts import HttpResponse

from PIL import Image
import os

# Create your views here.
va = {}



def index(request):
    if request.method == "POST":  # 请求方法为POST时，进行处理
        myFile = request.FILES.get("file", None)  # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return HttpResponse("没有文件")
        destination = open(os.path.join("./test_img/", myFile.name), 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():  # 分块写入文件
            destination.write(chunk)
        destination.close()
        path = './test_img/1.png'
        result = crick_test(path)
        value = "".join(result)
        va["value"] = value

    return render(request, "index.html", va)
