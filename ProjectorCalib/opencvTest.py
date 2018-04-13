import numpy as np
import cv2
import json
from model import Model
from CornerDetector import CornerDetector
from Ulities import showMat
from digitsRecognizer import DigitsRecognizer
from ChessBoardArea import ChessBoardArea
import math
import glob, os


def colorEquilibrium(img):
    iBlueGray = np.zeros(shape=(256,1),dtype = np.uint32)
    iGreenGray = np.zeros(shape=(256,1),dtype = np.uint32)
    iRedGray = np.zeros(shape=(256,1),dtype = np.uint32)
    fBlueGray = np.zeros(shape=(256,1),dtype=float)
    fGreenGray = np.zeros(shape=(256,1),dtype=float)
    fRedGray = np.zeros(shape=(256,1),dtype=float)

    nr,nc = img.shape[:2]
    imgRed = img[:,:,2]
    imgGreen = img[:,:,1]
    imgBlue = img[:,:,0]
    for i in range(nr):
        for j in range(nc):
            iBlueGray[imgBlue[i,j]] += 1
            iGreenGray[imgGreen[i,j]] += 1
            iRedGray[imgRed[i,j]] += 1

    iMinBlueGray =iMinGreenGray=iMinRedGray = 0
    iMaxBlueGray =iMaxGreenGray = iMaxRedGray = 255
    fBlueGrayPDF = fGreenGrayPDF = fRedGrayPDF = 0.0
    fEquationRatio = 0.005


    for i in range(256):
        fBlueGray[i] = float(iBlueGray[i])/float(nc*nr)
        fGreenGray[i] = float(iGreenGray[i])/float(nc*nr)
        fRedGray[i] = float(iRedGray[i])/float(nc*nr)


        fBlueGrayPDF += fBlueGray[i]
        if (fBlueGrayPDF < fEquationRatio):
            iMinBlueGray = i
        elif (fBlueGrayPDF > (1-fEquationRatio)):
            if (fBlueGrayPDF - fBlueGray[i]) < (1 - fEquationRatio):
                iMaxBlueGray = i

        fGreenGrayPDF += fGreenGray[i]
        if (fGreenGrayPDF < fEquationRatio):
            iMinGreenGray = i
        elif (fGreenGrayPDF > (1-fEquationRatio)):
            if (fGreenGrayPDF - fGreenGray[i]) < (1 - fEquationRatio):
                iMaxGreenGray = i

        fRedGrayPDF += fRedGray[i]
        if (fRedGrayPDF < fEquationRatio):
            iMinRedGray = i
        elif (fRedGrayPDF > (1-fEquationRatio)):
            if (fRedGrayPDF - fRedGray[i]) < (1 - fEquationRatio):
                iMaxRedGray = i

    fEquaParakb=255.0/float(iMaxBlueGray-iMinBlueGray)
    fEquaParadb=255.0*iMinBlueGray/float(iMinBlueGray-iMaxBlueGray)
    fEquaParakg=255.0/float(iMaxGreenGray-iMinGreenGray)
    fEquaParadg=255.0*iMinGreenGray/float(iMinGreenGray-iMaxGreenGray)
    fEquaParakr=255.0/float(iMaxRedGray-iMinRedGray)
    fEquaParadr=255.0*iMinRedGray/float(iMinRedGray-iMaxRedGray)

    for i in range(nr):
        for j in range(nc):
            iTempBlueGray = imgBlue[i,j]
            iTempBlueGray = np.int32(float(iTempBlueGray)*fEquaParakb+fEquaParadb+0.5)
            if iTempBlueGray < 0:
                iTempBlueGray = 0
            elif iTempBlueGray > 255:
                iTempBlueGray = 255
            img[i,j,0] = iTempBlueGray

            iTempGreenGray = imgGreen[i,j]
            iTempGreenGray = np.int32(float(iTempGreenGray)*fEquaParakg+fEquaParadg+0.5)
            if iTempGreenGray < 0:
                iTempGreenGray = 0
            elif iTempGreenGray > 255:
                iTempGreenGray = 255
            img[i,j,1] = iTempGreenGray

            iTempRedGray = imgRed[i,j]
            iTempRedGray = np.int32(float(iTempRedGray)*fEquaParakr+fEquaParadr+0.5)
            if iTempRedGray < 0:
                iTempRedGray = 0
            elif iTempRedGray > 255:
                iTempRedGray = 255
            img[i,j,2] = iTempRedGray
def removeshortlines(edge):
    # get all connected components in the image with their stats (including their size, in pixel)
    nb_edges, output, stats, _ = cv2.connectedComponentsWithStats(edge, connectivity=8)
    # output is an image where every component has a different value
    size = stats[1:, -1]  # extracting the size from the statistics

    # selecting bigger components
    result = edge.copy()
    for e in range(0, nb_edges - 1):
        # replace this line depending on your application, here I chose to keep
        # all components above the mean size of components in the image
        if size[e] >= 1000:
            th_up = e + 2
            th_do = th_up

            # masking to keep only the components which meet the condition
            mask = cv2.inRange(output, th_do, th_up)
            result = cv2.bitwise_xor(result, mask)
    return result
def isnearPt(target,pt):
    if math.sqrt(float(target[0]-pt[0])*float(target[0]-pt[0])+float(target[1]-pt[1])*float(target[1]-pt[1])) < 1.5:
        return True
    else:
        return False






def colordistoRed(color):
    # color is BGR
    return np.sqrt(np.square(float(color[0]))+np.square(float(color[1]))+np.square(float(color[2])-255.0))

def main():



    # Load an color image in grayscale
    chessboardConfigPath = "/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/calibImageParam.json"
    modelPath = '/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/logs/model-573000.tar'
    rootpath = "/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/test/"
    os.chdir(rootpath)
    for file in glob.glob("*.png"):

        name =  rootpath+file
        img = cv2.imread(name,1)


        digitsRec = DigitsRecognizer(modelPath)
        detector = CornerDetector(chessboardConfigPath,digitsRec)
        chessboardAreaGroup = detector.detect(img)
        print('detectd image',file)
        grouplist = []
        data = {}

        for i in range(len(chessboardAreaGroup)):
            groupdict = {}
            groupdict['index'] = chessboardAreaGroup[i].index
            ptlist = []
            for key in chessboardAreaGroup[i].cornersDict.keys():
                ptdict = {}
                ptdict['order'] = key
                ptdict['pos'] = [np.asscalar(chessboardAreaGroup[i].cornersDict[key][0]),np.asscalar(chessboardAreaGroup[i].cornersDict[key][1])]
                ptlist.append(ptdict)
            groupdict['pts'] = ptlist
            grouplist.append(groupdict)
        data["grouplist"] = grouplist
        savename = rootpath+file.split('.')[0]+'.json'
        with open(savename,'w') as outfile:
            json.dump(data,outfile)
            print('save json',savename)

    # aa = np.array([12,13])
    # t = aa[0]
    # ss = [aa[1]]
    # ss.append(aa[0])
    # ss.append(t)




    #
    #
    # # groupdict = {"index":32, "pts": [{"0":[1, 2]},{"1": [3, 4]}, {"2":[5, 6]} ]}



    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.bilateralFilter(img,9,75,75)
    # edge = cv2.Canny(img, 90, 180)
    # cv2.imshow('edge',edge)
    # cv2.imwrite('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/logitect/edge.png', edge)
    #
    # imgR = img[:, :, 2]
    # imgG = img[:, :, 1]
    # imgB = img[:, :, 0]
    # edgeR = cv2.Canny(imgR, 90, 180)
    # cv2.imshow('edge',edgeR)
    # cv2.imwrite('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/logitect/edgeR.png', edgeR)
    # imgG = cv2.bilateralFilter(imgG,9,75,75)
    # edgeG = cv2.Canny(imgG, 90, 180)
    # cv2.imshow('edge',edgeG)
    # cv2.imwrite('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/logitect/edgeG.png', edgeG)
    # edgeB = cv2.Canny(imgB, 90, 180)
    # cv2.imshow('edge',edgeB)
    # cv2.imwrite('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/logitect/edgeB.png', edgeB)
    # # find Harris corners
    # gray = np.float32(gray)
    # dst = cv2.cornerHarris(gray, 40, 15, 0.04)
    # dst = cv2.dilate(dst, None)
    # ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    # dst = np.uint8(dst)
    #
    # # find centroids
    # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    #
    # # define the criteria to stop and refine the corners
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    #
    # # Now draw them
    # res = np.hstack((centroids, corners))
    # res = np.int0(res)
    # img[res[:, 1], res[:, 0]] = [0, 0, 255]
    # img[res[:, 3], res[:, 2]] = [0, 255, 0]
    # cv2.imshow('sss',img)
    # cv2.imwrite('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/subpixel5.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

    # testsss = np.array([[1,2,3],[4,5,6]])
    # print(testsss)
    #
    # print(testsss[1,2])

    print ('Done')


if __name__ == '__main__':
    main()