from captcha.image import ImageCaptcha
from gen_image import __gen_random_captcha_text
from config import MAX_CAPTCHA

def creat_image():
    a = 0
    while a < 100:
        i = str(a)
        image = ImageCaptcha(width=160, height=60, font_sizes=[35])
        text = __gen_random_captcha_text(size=MAX_CAPTCHA)
        image.write(text,"./image/"+ i +".png")
        a += 1

if __name__ == '__main__':
    creat_image()
