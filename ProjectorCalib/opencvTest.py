import numpy as np
import cv2
from model import Model
from CornerDetector import CornerDetector
from Ulities import showMat
from digitsRecognizer import DigitsRecognizer


def main():
    # Load an color image in grayscale
    img = cv2.imread(
        '/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/IMG_0795.JPG',
        1)

    chessboardConfigPath = "/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/calibImageParam.json"

    modelPath = '/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/logs/model-573000.tar'
    digitsRec = DigitsRecognizer(modelPath)
    detector = CornerDetector(chessboardConfigPath,digitsRec)
    detector.detect(img)



    # imgR = imS[:,:,2]
    # imgG = imS[:,:,1]
    # imgB = imS[:,:,0]
    # cv2.namedWindow('imgR', cv2.WINDOW_NORMAL)
    # cv2.imshow('imgR', imgR)
    # cv2.namedWindow('imgG', cv2.WINDOW_NORMAL)
    # cv2.imshow('imgG', imgG)
    # cv2.namedWindow('imgB', cv2.WINDOW_NORMAL)
    # cv2.imshow('imgB', imgB)



    # testmat = np.zeros((256,512,3),dtype="uint8") #bgr
    #
    # for i in range(512):
    #     for j in range(256):
    #         testmat[j,i,0] = 0
    #         testmat[j,i,1] = 0
    #         testmat[j, i, 2] = (i+j)%255
    # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    # cv2.imshow('test', testmat)

    testsss = np.array([[1,2,3],[4,5,6]])
    print(testsss)

    print(testsss[1,2])

    print ('Done')


if __name__ == '__main__':
    main()