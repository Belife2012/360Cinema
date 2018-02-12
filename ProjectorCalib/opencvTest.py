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


    # digitsRec = DigitsRecognizer(modelPath)
    # detector = CornerDetector(chessboardConfigPath,digitsRec)
    # detector.detect(img)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    img[res[:, 1], res[:, 0]] = [0, 0, 255]
    img[res[:, 3], res[:, 2]] = [0, 255, 0]

    cv2.imwrite('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/subpixel5.png', img)



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