import numpy as np
import cv2

writepath = '/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/logitect/result/'
def showMat(img,scale,name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    height, width = img.shape[:2]
    imS = cv2.resize(img, (int(width/2.0), int(height/2.0)))
    cv2.imshow(name, imS)

def saveimage(img,name):
    cv2.imwrite(writepath+name+'.png',img)



