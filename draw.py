import cv2
import numpy as np
from paddleocr import PaddleOCR
import re

point_start = {'x':0,'y':0}
point_end = {'x':0,'y':0}
count = 10 #马赛克模糊程度
boxs = []
lst = []
re_s = []
ip_re = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
domain_re = r"(?<=http[s]://)(((((\d{1,2})|(1\d{2})|(2[0-4]\d)|(25[0-5]))\.){3}((\d{1,2})|(1\d{2})|(2[0-4]\d)|(25[0-5]))(:\d{,8})?)|([.\w-]*))((?=/)|(?!/))"
re_s.append(ip_re)
re_s.append(domain_re)

def mosaci(box,img):
    point_start['x'] = int(box[0][0])
    point_start['y'] = int(box[2][0])
    point_end['x'] = int(box[0][1])
    point_end['y'] = int(box[2][1])
    quan = img[point_end['x']:point_end['y'], point_start['x']: point_start['y']]
    c1 = quan.shape[0]
    c2 = quan.shape[1]
    quan = quan[::count,::count]#按count的数值来取像素
    quan = np.repeat(quan,count ,axis=0)#x轴放大
    quan = np.repeat(quan,count,axis=1)#y轴放大
    img[point_end['x']:point_end['y'], point_start['x']: point_start['y']] = quan[:c1,:c2]
    return img

def findstr(fulltext,ts):
    text = fulltext[1][0]
    for t in ts:
        if str(t) in str(text):
            box = fulltext[0]
            return box
        else:
            pass
if __name__ == '__main__':
    ocr = PaddleOCR(use_angle_cls=False, lang="ch")
    img_path = '0880c08b0110b1b49619.png'
    result = ocr.ocr(img_path, cls=False)
    img = cv2.imread(img_path)
    for fulltext in result:
        for i in re_s:
            match = re.compile(i).search(fulltext[1][0])
            if match is not None:
                lst.append(match[0])
            else:
                pass
        box = findstr(fulltext,lst)
        if box is None:
            pass
        else:
            boxs.append(box)
    if len(boxs) == 0:
        img_out = img
    else:
        for box in boxs:
            img_out = mosaci(box,img)
    cv2.imshow("img", img_out)
    cv2.imwrite('result.jpg',img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()