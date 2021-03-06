# coding=utf-8
from Ulities import saveimage
import numpy as np
import cv2
import json
from digitsRecognizer import DigitsRecognizer
import math
from PatchArea import PatchArea
from ChessBoardArea import ChessBoardArea
import random
import copy
import torch
from torch.autograd import Variable
import sys
sys.setrecursionlimit(10000)

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
def enhanceColor(img):
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img2
def findnearpt(pt,img):
    height, width = img.shape[:2]
    nearpt = []
    if pt[0] > 0:
        if img[pt[0]-1,pt[1]] == 255:
            nearpt.append([pt[0]-1,pt[1]])
            img[pt[0]-1,pt[1]] = 0

    if pt[0] < height-1:
        if img[pt[0]+1,pt[1]] == 255:
            nearpt.append([pt[0]+1,pt[1]])
            img[pt[0]+1,pt[1]] = 0

    if pt[0]>0 and pt[1]>0:
        if img[pt[0] - 1, pt[1]-1] == 255:
            nearpt.append([pt[0] - 1, pt[1]-1])
            img[pt[0] - 1, pt[1]-1] = 0

    if pt[0] > 0 and pt[1] < width-1:
        if img[pt[0] - 1, pt[1]+1] == 255:
            nearpt.append([pt[0] - 1, pt[1]+1])
            img[pt[0] - 1, pt[1]+1] = 0

    if pt[1] > 0 :
        if img[pt[0], pt[1]-1] == 255:
            nearpt.append([pt[0], pt[1]-1])
            img[pt[0], pt[1]-1] = 0

    if pt[1] < width -1:
        if img[pt[0], pt[1]+1] == 255:
            nearpt.append([pt[0] , pt[1]+1])
            img[pt[0] , pt[1]+1] = 0

    if pt[0] < height -1 and pt[1] > 0:
        if img[pt[0] + 1, pt[1]-1] == 255:
            nearpt.append([pt[0] + 1, pt[1]-1])
            img[pt[0] + 1, pt[1]-1] = 0

    if pt[0] < height-1 and pt[1] < width -1:
        if img[pt[0] + 1, pt[1]+1] == 255:
            nearpt.append([pt[0] + 1, pt[1]+1])
            img[pt[0] + 1, pt[1]+1] = 0

    for i in range(len(nearpt)):
        nearp = findnearpt(nearpt[i],img)
        nearpt = nearpt+nearp
    return nearpt

def removeShortelements(img):
    dotset = {}
    index = 0
    height, width = img.shape[:2]
    im = img.copy()
    seglist = []

    for i in range(height):
        for j in range(width):
            if im[i,j] == 255:
                segset = []
                segset.append([i,j])
                im[i,j]=0
                segs = findnearpt([i,j],im)
                segset = segset+segs
                seglist.append(segset)

    for i in range(len(seglist)):
        if len(seglist[i]) < 40:
            for j in range(len(seglist[i])):
                img[seglist[i][j][0],seglist[i][j][1]] = 0

def computeSolidity(cnt):
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    return float(area) / hull_area

def distofTwoPoints(pnt1,pnt2):
    return np.sqrt(np.square(pnt1[0]-pnt2[0])+np.square(pnt1[1]-pnt2[1]))

def findmidPoint(pnt1,pnt2):
    return np.array([(pnt1[0]+pnt2[0])/2.0,(pnt1[1]+pnt2[1])/2.0])
def doRANSAC(datax,datay):
    maxcount = 0
    sigma = 0.03
    iterNum = 1000
    isVertical = False
    beta = 0
    alpha = 0
    InnerX = []
    InnerY = []
    OuterX = []
    OuterY = []
    w = np.max(datax)-np.min(datax)
    h = np.max(datay)-np.min(datay)

    topcount = 10
    firstsearchingX= []
    firstsearchingY = []

    secondsearchingX = []
    secondsearchingY = []
    if h > w:
        indexarr = findTopNindex(datay,topcount)
        for i in range(len(indexarr)):
            firstsearchingX.append(datax[indexarr[i]])
            firstsearchingY.append(datay[indexarr[i]])
        neglist = datay*-1
        indexarr = findTopNindex(neglist,topcount)
        for i in range(len(indexarr)):
            secondsearchingX.append(datax[indexarr[i]])
            secondsearchingY.append(datay[indexarr[i]])


    elif h < w:
        indexarr = findTopNindex(datax,topcount)
        for i in range(len(indexarr)):
            firstsearchingX.append(datax[indexarr[i]])
            firstsearchingY.append(datay[indexarr[i]])
        neglist = datax*-1
        indexarr = findTopNindex(neglist,topcount)
        for i in range(len(indexarr)):
            secondsearchingX.append(datax[indexarr[i]])
            secondsearchingY.append(datay[indexarr[i]])

    else:
        firstsearchingX = datax
        firstsearchingY = datay

        secondsearchingX = datax
        secondsearchingY = datay

        iterNum = 1000000


    while(iterNum):
        iterNum-=iterNum
        #random.seed(calendar.timegm(time.gmtime()))
        first = random.randint(0,len(firstsearchingX)-1)
        second = random.randint(0,len(secondsearchingX)-1)
        firstpoint = [firstsearchingX[first],firstsearchingY[first]]
        secondpoint = [secondsearchingX[second],secondsearchingY[second]]
        _innersX = []
        _innersY = []
        _outersX = []
        _outersY = []
        _maxcount = 0
        # if firstpoint[1] != secondpoint[1] and (float(secondpoint[1]-firstpoint[1])/float(secondpoint[0]-firstpoint[0])) > 1:
        #     #vertical line and beta > 1 reverse it
        #
        #     _betaReverse = float(secondpoint[0] - firstpoint[0])/float(secondpoint[1] - firstpoint[1])
        #     _alphaReverse = (float(secondpoint[1]*firstpoint[0])-float(firstpoint[1]*secondpoint[0]))/float(secondpoint[1] - firstpoint[1])
        #
        #     for i in range(len(datax)):
        #         if i != first and i != second:
        #             if math.fabs(_betaReverse*datay[i] - datax[i] + _alphaReverse)/math.sqrt(_betaReverse*_betaReverse + 1) <= sigma:
        #                 _maxcount += 1
        #                 _innersX.append(datax[i])
        #                 _innersY.append(datay[i])
        #             else:
        #                 _outersX.append(datax[i])
        #                 _outersY.append(datay[i])
        #     if _maxcount > maxcount:
        #         maxcount = _maxcount
        #         _innersX.append(firstpoint[0])
        #         _innersX.append(secondpoint[0])
        #         _innersY.append(firstpoint[1])
        #         _innersY.append(secondpoint[1])
        #         InnerX = _innersX
        #         InnerY = _innersY
        #         OuterX = _outersX
        #         OuterY = _outersY
        #
        #     if math.fabs(_betaReverse) <= sys.float_info.epsilon:
        #         beta = sys.float_info.max
        #         alpha = sys.float_info.min
        #     else:
        #         beta = 1/_betaReverse
        #         alpha = -_alphaReverse/_betaReverse
        if firstpoint[0] == secondpoint[0] :
            # vertical line and beta > 1 reverse it


            for i in range(len(datax)):
                if i != first and i != second:
                    if math.fabs(datax[i] - firstpoint[0])<= sigma:
                        _maxcount += 1
                        _innersX.append(datax[i])
                        _innersY.append(datay[i])
                    else:
                        _outersX.append(datax[i])
                        _outersY.append(datay[i])
            if _maxcount > maxcount:
                maxcount = _maxcount
                _innersX.append(firstpoint[0])
                _innersX.append(secondpoint[0])
                _innersY.append(firstpoint[1])
                _innersY.append(secondpoint[1])
                InnerX = _innersX
                InnerY = _innersY
                OuterX = _outersX
                OuterY = _outersY


            beta = sys.float_info.max
            alpha = sys.float_info.min



        else:
            # nonreverse line
            _beta = float(secondpoint[1]-firstpoint[1])/float(secondpoint[0]-firstpoint[0])
            _alpha = (float(secondpoint[0]*firstpoint[1])-float(firstpoint[0]*secondpoint[1]))/float(secondpoint[0]-firstpoint[0])
            for i in range(len(datax)):
                if i != first and i != second:
                    if math.fabs(_beta*datax[i] - datay[i] + _alpha)/math.sqrt(_beta*_beta + 1) <= sigma:
                        _maxcount += 1
                        _innersX.append(datax[i])
                        _innersY.append(datay[i])
                    else:
                        _outersX.append(datax[i])
                        _outersY.append(datay[i])
            if _maxcount > maxcount:
                maxcount = _maxcount
                _innersX.append(firstpoint[0])
                _innersX.append(secondpoint[0])
                _innersY.append(firstpoint[1])
                _innersY.append(secondpoint[1])
                InnerX = _innersX
                InnerY = _innersY
                OuterX = _outersX
                OuterY = _outersY
                beta = _beta
                alpha = _alpha

    return InnerX,InnerY,OuterX,OuterY,beta,alpha
def findTopNindex(arr,N):
    return np.argsort(arr)[::-1][:N]
class CornerDetector(object):

    def __init__(self,chessboardConfigPath,digitsDetector):
        json_data = open(chessboardConfigPath)
        data = json.load(json_data)
        self._chessboardsize = [data['chessboardsize']['height'],data['chessboardsize']['width']]
        self._imagesize = [data['imagesize']['height'],data['imagesize']['width']]
        self._patchsize = [data['patchsize']['height'],data['patchsize']['width']]
        self.__oneGroupCorners = []
        self.__digitDetector = digitsDetector

    def __validateContour(self,cnts,height,width):
        resultlist = []
        minval = height / (self._chessboardsize[0] * 2.0)
        maxval = height *2.0/ 3.0
        boarderWidth = height/6.0

        for i in range(len(cnts)):
            if cnts[i].shape[0] > minval and cnts[i].shape[0] < maxval :
                #remove boarder contours
                center = np.array([np.mean(cnts[i][:,:,0]),np.mean(cnts[i][:,:,1])])
                if center[0] < boarderWidth or center[0] > width-boarderWidth or center[1] < boarderWidth or center[1] > height -boarderWidth:
                    isnearBoarder = False
                    for j in range(cnts[i].shape[0]):
                        if cnts[i][j,0,0] < 5 or cnts[i][j,0,0] > width-5 or cnts[i][j,0,1]<5 or cnts[i][j,0,1] > height -5:
                            isnearBoarder = True
                            break
                    if isnearBoarder:
                        continue

                x, y, w, h = cv2.boundingRect(cnts[i])

                boundingboxArea = float(w*h)
                contourArea = cv2.contourArea(cnts[i])

                if contourArea/boundingboxArea > 0.3:
                    epsilon = 0.05 * cv2.arcLength(cnts[i], True)
                    approx = cv2.approxPolyDP(cnts[i], epsilon, True)
                    if len(approx) == 4:
                        resultlist.append(approx)
                    # resultlist.append(cnts[i])


        return resultlist

    def __PointOnLine(self,p0,p1,scale):
        return [p0[0]+(p1[0]-p0[0])*scale,p0[1]+(p1[1]-p0[1])*scale]


    #patch is 53 for
    def __cropRect(self,img,arect,patchsize = 54):
        assert arect.shape == (4,1,2)
        #cropimg = np.zeros(shape=(patchsize,patchsize,3),dtype="uint8")
        rect = copy.deepcopy(arect)
        rect = rect.reshape(4,2)
        #rect = np.flip(rect,1)
        t = np.copy(rect[2, :])
        rect[2, :] = rect[3, :]
        rect[3, :] = t
        rectM = np.float32(rect)
        trect = np.float32([[0,0],[patchsize,0],[0,patchsize],[patchsize,patchsize]])
        M = cv2.getPerspectiveTransform(rectM, trect)
        return cv2.warpPerspective(img, M, (patchsize, patchsize))
        # for i in range(patchsize):
        #     for j in range(patchsize):
        #         yscale = j/patchsize
        #         xscale = i/patchsize
        #         pn = self.__PointOnLine(rect[0,0,:],rect[3,0,:],yscale)
        #         pm = self.__PointOnLine(rect[1,0,:],rect[2,0,:],yscale)
        #         pp = self.__PointOnLine(pn,pm,xscale)
        #         scaleL = [pp[0]-math.floor(pp[0]), pp[1] - math.floor(pp[1])]  #get decimal part of float。
        #         a = scaleL[0]*scaleL[1]
        #         b = (1-scaleL[0])*scaleL[1]
        #         c = (1-scaleL[0])*(1-scaleL[1])
        #         d = scaleL[0]*(1-scaleL[1])
        #         crop = img[int(pp[0]):int(pp[0])+2 ,int(pp[1]):int(pp[1])+2,:]
        #         crop[0,0,:] = crop[0,0,:]*c
        #         crop[0,1,:] = crop[0,1,:]*d
        #         crop[1,0,:] = crop[1,0,:]*b
        #         crop[1,1,:] = crop[1,1,:]*a
        #         cropimg[i,j,:] = np.round( np.sum(np.sum(crop,axis = 1),axis = 0))
        #return cropimg


    def __findnearRects(self,Rect,cornersList):
        nearList = []
        leftList = []
        center = np.array([np.mean(Rect[:, :, 0]), np.mean(Rect[:, :, 1])])
        sum = 0.0
        for i in range(Rect.shape[0]):
            sum += np.sqrt(np.square(center[0] - Rect[i, 0, 0]) + np.square(center[1] - Rect[i, 0, 1]))

        meanR = sum/float(Rect.shape[0])

        for rect in cornersList:
            isnear = False
            for i in range(rect.shape[0]):
                for j in range(Rect.shape[0]):
                    if np.sqrt(np.square(rect[i,0,0]-Rect[j,0,0])+np.square(rect[i,0,1]-Rect[j,0,1])) < meanR*2.0/3.0:
                        nearList.append(rect)
                        isnear = True
                        break
                if isnear:
                    break
            if not(isnear):
                leftList.append(rect)
        return nearList,leftList

    def __findneargroupRects(self,nearRects,leftRects):
        for rect in nearRects:
            anearRects,leftRects = self.__findnearRects(rect,leftRects)
            for arect in anearRects:
                self.__oneGroupCorners.append(arect)
            if len(anearRects) > 0 :
                leftRects = self.__findneargroupRects(anearRects,leftRects)
        return leftRects

    def __groupCorners(self,cornersList):
        resultGroups = []

        while len(cornersList) > 0:
            self.__oneGroupCorners = []
            rootRect = cornersList.pop(0)
            self.__oneGroupCorners.append(rootRect)
            nearlist ,cornersList = self.__findnearRects(rootRect,cornersList)
            for rect in nearlist:
                self.__oneGroupCorners.append(rect)
            cornersList = self.__findneargroupRects(nearlist,cornersList)
            resultGroups.append(list(self.__oneGroupCorners))
        return resultGroups

    def __sortRectCorners(self,rect):
        center = np.array([np.mean(rect[:, :, 0]), np.mean(rect[:, :, 1])])
        upperList = rect[rect[:, 0, 1] <= center[1]]
        LowerList = rect[rect[:, 0, 1] > center[1]]
        upperList = upperList[upperList[:, 0, 0].argsort()]
        LowerList = -LowerList
        LowerList = -LowerList[LowerList[:, 0, 0].argsort()]
        return np.concatenate((upperList, LowerList), axis=0),center  # [p0,p1,p2,p3]

    def __combineGroups(self,redgroup,greengroup):
        redcenterlist = []
        greencenterlist = []
        redsizeList = []
        chessboardList = []
        for group in redgroup:
            center = np.array([0,0])
            round = 0
            for rect in group:
                arect,acenter = self.__sortRectCorners(rect)
                rect[:,0,:] = arect[:,0,:]
                center[0] += acenter[0]
                center[1] += acenter[1]
                round += (distofTwoPoints(rect[0,0,:],rect[1,0,:])+distofTwoPoints(rect[1,0,:],rect[2,0,:])+distofTwoPoints(rect[2,0,:],rect[3,0,:])+distofTwoPoints(rect[0,0,:],rect[3,0,:]))
            center = center/len(group)
            round = round /len(group)
            redcenterlist.append(center)
            redsizeList.append(round)

        for group in greengroup:
            center = np.array([0,0])
            for rect in group:
                arect,acenter = self.__sortRectCorners(rect)
                rect[:,0,:] = arect[:,0,:]
                center[0] += acenter[0]
                center[1] += acenter[1]
            center = center/len(group)
            greencenterlist.append(center)

        for i in range(len(redcenterlist)):
            agroup = []
            matchedgreencenter = -1
            for j in range(len(greencenterlist)):
                if distofTwoPoints(greencenterlist[j],redcenterlist[i]) < redsizeList[i]:
                    matchedgreencenter = j
                    break
            for rect in redgroup[i]:
                rectAreared = PatchArea('RED',rect)
                agroup.append(rectAreared)
            for rect in greengroup[matchedgreencenter]:
                rectAreagreen = PatchArea('GREEN',rect)
                agroup.append(rectAreagreen)
            if len(agroup)>=3:
                chessboardList.append(agroup)
        return chessboardList



    def __findtestpoint(self,midpoint,mirrorpoint,height,width):
        if (mirrorpoint[0]>3 or mirrorpoint[0] < width-4) and (mirrorpoint[1] < 3 or mirrorpoint[1] < height-4):
            return mirrorpoint
        else:
            center = findmidPoint(midpoint,mirrorpoint)
            return self.__findtestpoint(midpoint,center,height,width)
        # x = 0 if mirrorpoint[0] < 0 else (width if mirrorpoint[0] > width else mirrorpoint[0])
        # y = 0 if mirrorpoint[1] < 0 else (height if mirrorpoint[1] > height else mirrorpoint[1])


    def __verifychessboardGroup(self,chessboardGroups,chessboardMask):
        result = []
        height, width = chessboardMask.shape[:2]

        for group in chessboardGroups:
            cornersnum = 0
            isgoodgroup = False
            for rectArea in group:
                if cornersnum >= 4:
                    break
                arect = rectArea.rect
                center = np.array([np.mean(arect[:,0,0]),np.mean(arect[:,0,1])])

                p1 = findmidPoint(arect[0,0,:],arect[1,0,:])
                p2 = findmidPoint(arect[1,0,:],arect[2,0,:])
                p3 = findmidPoint(arect[2,0,:],arect[3,0,:])
                p4 = findmidPoint(arect[3,0,:],arect[0,0,:])

                p = np.zeros(shape=(4,1,2),dtype=float)
                p[0,0,:] = p1*2-center
                p[1,0,:] = p2*2-center
                p[2,0,:] = p3*2-center
                p[3,0,:] = p4*2-center

                blacksidenum = [0,0,0,0]
                for i in range(4):
                    pp = p1 if i==0 else (p2 if i==1 else (p3 if i==2 else p4))
                    testp = self.__findtestpoint(pp,p[i,0,:],height,width)

                    ii = 0
                    for w in range(-2,3):
                        for h in range(-2,3):
                            if chessboardMask[int(testp[1]+w),int(testp[0]+h)] == 255:
                                ii += 1
                    if ii < 6:
                        blacksidenum[i] = 1
                if (blacksidenum[0]==1 and blacksidenum[1]==1) or (blacksidenum[1]==1 and blacksidenum[2]==1) or (blacksidenum[2]==1 and blacksidenum[3]==1) or (blacksidenum[0]==1 and blacksidenum[3]==1) :
                    rectArea.iscorner = True
                    cornersnum +=1
                    isgoodgroup = True
            if isgoodgroup:
                result.append(group)
        return result

    def __rotateRectLeft90D(self,rect):
        arect = np.copy(rect)
        temp = arect[0,0,:]
        arect[0,0,:] = arect[1,0,:]
        arect[1,0,:] = arect[2,0,:]
        arect[2,0,:] = arect[3,0,:]
        arect[3,0,:] = temp
        return arect

    def __identifyRectArea(self,goodchessboardGroups,img):
        for group in goodchessboardGroups:
            indexlist = {}
            for rectArea in group:
                crop = self.__cropRect(img,rectArea.rect)
                result = self.__digitDetector.evaluate(crop)
                if len(result) == 3:
                    if(result[0]==3 or result[0]==4)and result[2] !=7 :
                        resultindex = result[0]*100+result[1]*10+result[2]
                        rectArea.possibleChessboardindexdict[resultindex] = rectArea.rect
                        if resultindex in indexlist:
                            indexlist[resultindex] += 1
                        else:
                            indexlist[resultindex] = 1
                rectL = self.__rotateRectLeft90D(rectArea.rect)
                cropL = self.__cropRect(img,rectL)
                resultL = self.__digitDetector.evaluate(cropL)
                if len(resultL)==3:
                    if(resultL[0]==3 or resultL[0]==4)and resultL[2] !=7 :
                        resultindex = resultL[0]*100+resultL[1]*10+resultL[2]
                        rectArea.possibleChessboardindexdict[resultindex] = rectL
                        if resultindex in indexlist:
                            indexlist[resultindex] += 1
                        else:
                            indexlist[resultindex] = 1
                rectV = self.__rotateRectLeft90D(rectL)
                cropV= self.__cropRect(img,rectV)
                resultV = self.__digitDetector.evaluate(cropV)
                if len(resultV) == 3:
                    if(resultV[0]==3 or resultV[0]==4)and resultV[2] !=7 :
                        resultindex = resultV[0]*100+resultV[1]*10+resultV[2]
                        rectArea.possibleChessboardindexdict[resultindex] = rectV

                        if resultindex in indexlist:
                            indexlist[resultindex] += 1
                        else:
                            indexlist[resultindex] = 1
                rectR = self.__rotateRectLeft90D(rectV)
                cropR = self.__cropRect(img,rectR)
                resultR = self.__digitDetector.evaluate(cropR)

                if len(resultR) == 3:
                    if(resultR[0]==3 or resultR[0]==4)and resultR[2] !=7 :
                        resultindex = resultR[0]*100+resultR[1]*10+resultR[2]
                        rectArea.possibleChessboardindexdict[resultindex] = rectR

                        if resultindex in indexlist:
                            indexlist[resultindex] += 1
                        else:
                            indexlist[resultindex] = 1
            maxkey = 0
            maxsum = 0
            for key in indexlist:
                if indexlist[key] > maxsum:
                    maxkey = key
                    maxsum = indexlist[key]
            for rectArea in group:
                for key in rectArea.possibleChessboardindexdict:
                    resultRect = rectArea.possibleChessboardindexdict[key]
                    if key == maxkey:
                        rectArea.identified = True
                        rectArea.rect = resultRect
                        rectArea.chessboardindex = key
                        if key < 400:
                            rectArea.isAssitance = True
                        break
            ### !!!!!!!!!!!!!!!!!!!!!! Here has some problems !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




    def __findRectsAround(self,target,rectArea):
        NorthRect = None
        SourthRect = None
        EastRect = None
        WestRect = None
        center = np.array([np.mean(target.rect[:, :, 0]), np.mean(target.rect[:, :, 1])])
        sum = 0.0
        for i in range(target.rect.shape[0]):
            sum += np.sqrt(np.square(center[0] - target.rect[i, 0, 0]) + np.square(center[1] - target.rect[i, 0, 1]))

        meanR = sum/(float(target.rect.shape[0])*2.0)

        leftAreaGroup = []
        aroundRectList = []
        identified = True
        # for rectArea in rectAreaGroup:
        if distofTwoPoints(rectArea.rect[0,0,:],target.rect[0,0,:]) < meanR and distofTwoPoints(rectArea.rect[3,0,:],target.rect[1,0,:]) < meanR:
            NorthRect = rectArea
            NorthRect.order = target.order - self._chessboardsize[1]
            NorthRect.rect = np.array([[rectArea.rect[1,0,:]],[rectArea.rect[2,0,:]],[rectArea.rect[3,0,:]],[rectArea.rect[0,0,:]]])
            NorthRect.hasAround[2] = True
            target.hasAround[0] = True
            aroundRectList.append(NorthRect)

            # continue
        elif distofTwoPoints(rectArea.rect[0,0,:],target.rect[1,0,:]) < meanR and distofTwoPoints(rectArea.rect[3,0,:],target.rect[2,0,:]) < meanR:
            EastRect = rectArea
            EastRect.order = target.order + 1
            EastRect.hasAround[3] = True
            target.hasAround[1] = True
            aroundRectList.append(EastRect)
            # continue
        elif distofTwoPoints(rectArea.rect[3,0,:],target.rect[0,0,:]) < meanR and distofTwoPoints(rectArea.rect[2,0,:],target.rect[1,0,:]) < meanR:
            NorthRect = rectArea
            NorthRect.order = target.order - self._chessboardsize[1]
            NorthRect.hasAround[2] = True
            target.hasAround[0] = True
            aroundRectList.append(NorthRect)
            # continue
        elif distofTwoPoints(rectArea.rect[3,0,:],target.rect[1,0,:]) < meanR and distofTwoPoints(rectArea.rect[2,0,:],target.rect[2,0,:]) < meanR:
            EastRect = rectArea
            EastRect.order = target.order + 1
            EastRect.rect = np.array([[rectArea.rect[3,0,:]],[rectArea.rect[0,0,:]],[rectArea.rect[1,0,:]],[rectArea.rect[2,0,:]]])
            EastRect.hasAround[3] = True
            target.hasAround[1] = True
            aroundRectList.append(EastRect)
            # continue
        elif distofTwoPoints(rectArea.rect[2,0,:],target.rect[0,0,:]) < meanR and distofTwoPoints(rectArea.rect[1,0,:],target.rect[1,0,:]) < meanR:
            NorthRect = rectArea
            NorthRect.order = target.order - self._chessboardsize[1]
            NorthRect.rect = np.array([[rectArea.rect[3,0,:]],[rectArea.rect[0,0,:]],[rectArea.rect[1,0,:]],[rectArea.rect[2,0,:]]])
            NorthRect.hasAround[2] = True
            aroundRectList.append(NorthRect)
            target.hasAround[0] = True
            # continue
        elif distofTwoPoints(rectArea.rect[2,0,:],target.rect[1,0,:]) < meanR and distofTwoPoints(rectArea.rect[1,0,:],target.rect[2,0,:]) < meanR:
            EastRect = rectArea
            EastRect.order = target.order + 1
            EastRect.rect = np.array([[rectArea.rect[2,0,:]],[rectArea.rect[3,0,:]],[rectArea.rect[0,0,:]],[rectArea.rect[1,0,:]]])
            EastRect.hasAround[3] = True
            target.hasAround[1] = True
            aroundRectList.append(EastRect)
            # continue
        elif distofTwoPoints(rectArea.rect[1,0,:],target.rect[0,0,:]) < meanR and distofTwoPoints(rectArea.rect[0,0,:],target.rect[1,0,:]) < meanR:
            NorthRect = rectArea
            NorthRect.order = target.order - self._chessboardsize[1]
            NorthRect.rect = np.array([[rectArea.rect[2,0,:]],[rectArea.rect[3,0,:]],[rectArea.rect[0,0,:]],[rectArea.rect[1,0,:]]])
            NorthRect.hasAround[2] = True
            target.hasAround[0] = True
            aroundRectList.append(NorthRect)
            # continue
        elif distofTwoPoints(rectArea.rect[1,0,:],target.rect[1,0,:]) < meanR and distofTwoPoints(rectArea.rect[0,0,:],target.rect[2,0,:]) < meanR:
            EastRect = rectArea
            EastRect.order = target.order + 1
            EastRect.rect = np.array([[rectArea.rect[1,0,:]],[rectArea.rect[2,0,:]],[rectArea.rect[3,0,:]],[rectArea.rect[0,0,:]]])
            EastRect.hasAround[3] = True
            target.hasAround[1] = True
            aroundRectList.append(EastRect)
            # continue

        elif distofTwoPoints(rectArea.rect[0,0,:],target.rect[3,0,:]) < meanR and distofTwoPoints(rectArea.rect[1,0,:],target.rect[2,0,:]) < meanR:
            SourthRect = rectArea
            SourthRect.order = target.order + self._chessboardsize[1]
            SourthRect.hasAround[0] = True
            target.hasAround[2] = True
            aroundRectList.append(SourthRect)
            # continue
        elif distofTwoPoints(rectArea.rect[0,0,:],target.rect[0,0,:]) < meanR and distofTwoPoints(rectArea.rect[1,0,:],target.rect[3,0,:]) < meanR:
            WestRect = rectArea
            WestRect.order = target.order - 1
            WestRect.rect = np.array([[rectArea.rect[3,0,:]],[rectArea.rect[0,0,:]],[rectArea.rect[1,0,:]],[rectArea.rect[2,0,:]]])
            WestRect.hasAround[1]=True
            target.hasAround[3] = True
            aroundRectList.append(WestRect)
            # continue
        elif distofTwoPoints(rectArea.rect[1,0,:],target.rect[3,0,:]) < meanR and distofTwoPoints(rectArea.rect[2,0,:],target.rect[2,0,:]) < meanR:
            SourthRect = rectArea
            SourthRect.order = target.order + self._chessboardsize[1]
            SourthRect.rect = np.array([[rectArea.rect[1, 0, :]], [rectArea.rect[2, 0, :]], [rectArea.rect[3, 0, :]],[rectArea.rect[0, 0, :]]])
            SourthRect.hasAround[0] = True
            target.hasAround[2] = True
            aroundRectList.append(SourthRect)
            # continue
        elif distofTwoPoints(rectArea.rect[1,0,:],target.rect[0,0,:]) < meanR and distofTwoPoints(rectArea.rect[2,0,:],target.rect[3,0,:]) < meanR:
            WestRect = rectArea
            WestRect.order = target.order - 1
            WestRect.hasAround[1]=True
            target.hasAround[3] = True
            aroundRectList.append(WestRect)
            # continue
        elif distofTwoPoints(rectArea.rect[2,0,:],target.rect[3,0,:]) < meanR and distofTwoPoints(rectArea.rect[3,0,:],target.rect[2,0,:]) < meanR:
            SourthRect = rectArea
            SourthRect.order = target.order + self._chessboardsize[1]
            SourthRect.rect = np.array([[rectArea.rect[2,0,:]],[rectArea.rect[3,0,:]],[rectArea.rect[0,0,:]],[rectArea.rect[1,0,:]]])
            SourthRect.hasAround[0] = True
            target.hasAround[2] = True
            aroundRectList.append(SourthRect)
            # continue
        elif distofTwoPoints(rectArea.rect[2,0,:],target.rect[0,0,:]) < meanR and distofTwoPoints(rectArea.rect[3,0,:],target.rect[3,0,:]) < meanR:
            WestRect = rectArea
            WestRect.order = target.order - 1
            WestRect.rect = np.array([[rectArea.rect[1, 0, :]], [rectArea.rect[2, 0, :]], [rectArea.rect[3, 0, :]],[rectArea.rect[0, 0, :]]])
            WestRect.hasAround[1]=True
            target.hasAround[3] = True
            aroundRectList.append(WestRect)
            # continue
        elif distofTwoPoints(rectArea.rect[3,0,:],target.rect[3,0,:]) < meanR and distofTwoPoints(rectArea.rect[0,0,:],target.rect[2,0,:]) < meanR:
            SourthRect = rectArea
            SourthRect.order = target.order + self._chessboardsize[1]
            SourthRect.rect = np.array([[rectArea.rect[3,0,:]],[rectArea.rect[0,0,:]],[rectArea.rect[1,0,:]],[rectArea.rect[2,0,:]]])
            SourthRect.hasAround[0] = True
            target.hasAround[2] = True
            aroundRectList.append(SourthRect)
            # continue
        elif distofTwoPoints(rectArea.rect[3,0,:],target.rect[0,0,:]) < meanR and distofTwoPoints(rectArea.rect[0,0,:],target.rect[3,0,:]) < meanR:
            WestRect = rectArea
            WestRect.order = target.order - 1
            WestRect.rect = np.array([[rectArea.rect[2,0,:]],[rectArea.rect[3,0,:]],[rectArea.rect[0,0,:]],[rectArea.rect[1,0,:]]])
            WestRect.hasAround[1]=True
            target.hasAround[3] = True
            aroundRectList.append(WestRect)
            # continue
        else:
            leftAreaGroup.append(rectArea)
            identified = False
        return identified


    def __findAround(self,aroundlist,leftlist):
        if len(aroundlist) > 0:
            for target in aroundlist:
                around,left = self.__findRectsAround(target,leftlist)
            if len(around) > 0 :
                leftlist =self.__findAround(around,left)

        return leftlist

    def __find4CrossPoint(self,rectlist, cornerindex):
        wstep = self._chessboardsize[1]
        wwstep = wstep + 1
        offset = math.floor(cornerindex / wwstep)

        if len(rectlist) == 4:
            for rect in rectlist:
                if rect.order == cornerindex - wstep - 1 - offset:
                    p1 = rect.rect[2,0,:]
                elif rect.order == cornerindex - wstep - offset:
                    p2 = rect.rect[3,0,:]
                elif rect.order == cornerindex - offset:
                    p3 = rect.rect[0,0,:]
                elif rect.order == cornerindex - 1 - offset:
                    p4 = rect.rect[1,0,:]
            p1 = p1.astype(float)
            p2 = p2.astype(float)
            p3 = p3.astype(float)
            p4 = p4.astype(float)
            #p1,p3 line
            if p1[0] == p3[0] and  p2[0]!= p4[0]:
                return np.array([p1[0],(p4[1]-p2[1])*p1[0]/(p4[0]-p2[0])+(p4[0]*p2[1]-p4[1]*p2[0])/(p4[0]-p2[0])])
            #p2,p4 line
            elif p1[0] != p3[0] and p2[0] == p4[0]:
                return np.array([p2[0],(p3[1]-p1[1])*p2[0]/(p3[0]-p1[0])+(p3[0]*p1[1]-p3[1]*p1[0])/(p3[0]-p1[0])])

            elif p1[0] != p3[0] and p3[0] != p4[0]:
                k1 = (p3[1]-p1[1])/(p3[0]-p1[0])
                b1 = (p3[0]*p1[1]-p3[1]*p1[0])/(p3[0]-p1[0])
                k2 = (p4[1]-p2[1])/(p4[0]-p2[0])
                b2 = (p4[0]*p2[1]-p4[1]*p2[0])/(p4[0]-p2[0])
                if k2 != k1:
                    return np.array([(b1-b2)/(k2-k1),(k2*b1-k1*b2)/(k2-k1)])
                else:
                    print('Invalid crosspoint')
                    return np.array([-1, -1])
            else:
                print('Invalid crosspoint')
                return np.array([-1,-1])
        else:
            print('rect list length is not 4')




    def __findROI(self,rectlist,cornerindex):
        roix = np.zeros(shape=[4,2],dtype='int')
        roiy = np.zeros(shape=[4,2],dtype='int')
        wstep = self._chessboardsize[1]
        wwstep = wstep+1
        scale = 0.2
        offset = math.floor(cornerindex / wwstep)

        if len(rectlist) == 4:
            for rect in rectlist:
                if rect.order == cornerindex - wstep - 1 - offset:
                    roix[0,:] = findmidPoint(rect.rect[2,0,:],rect.rect[3,0,:])
                    roiy[0,:] = findmidPoint(rect.rect[1,0,:],rect.rect[2,0,:])
                elif rect.order == cornerindex - wstep- offset:
                    roix[1,:] = findmidPoint(rect.rect[2,0,:],rect.rect[3,0,:])
                    roiy[1,:] = findmidPoint(rect.rect[0,0,:],rect.rect[3,0,:])
                elif rect.order == cornerindex- offset:
                    roix[2,:] = findmidPoint(rect.rect[0,0,:],rect.rect[1,0,:])
                    roiy[2,:] = findmidPoint(rect.rect[0,0,:],rect.rect[3,0,:])
                elif rect.order == cornerindex-1- offset:
                    roix[3,:] = findmidPoint(rect.rect[0,0,:],rect.rect[1,0,:])
                    roiy[3,:] = findmidPoint(rect.rect[1,0,:],rect.rect[2,0,:])
        elif len(rectlist) == 3:
            rectdic = {0:'LT',1:'RT',2:'RB',3:'LB'}
            for rect in rectlist:
                if rect.order == cornerindex - wstep - 1- offset :
                    rectdic.pop(0,None)
                    roix[0,:] = findmidPoint(rect.rect[2,0,:],rect.rect[3,0,:])
                    roiy[0,:] = findmidPoint(rect.rect[1,0,:],rect.rect[2,0,:])
                elif rect.order == cornerindex - wstep- offset:
                    rectdic.pop(1,None)
                    roix[1,:] = findmidPoint(rect.rect[2,0,:],rect.rect[3,0,:])
                    roiy[1,:] = findmidPoint(rect.rect[0,0,:],rect.rect[3,0,:])
                elif rect.order == cornerindex- offset:
                    rectdic.pop(2,None)
                    roix[2,:] = findmidPoint(rect.rect[0,0,:],rect.rect[1,0,:])
                    roiy[2,:] = findmidPoint(rect.rect[0,0,:],rect.rect[3,0,:])
                elif rect.order == cornerindex-1- offset:
                    rectdic.pop(3,None)
                    roix[3,:] = findmidPoint(rect.rect[0,0,:],rect.rect[1,0,:])
                    roiy[3,:] = findmidPoint(rect.rect[1,0,:],rect.rect[2,0,:])
            if 0 in rectdic.keys():
                for rect in rectlist:
                    if rect.order == cornerindex - wstep- offset:
                        x1 = findmidPoint(rect.rect[1, 0, :], rect.rect[2, 0, :])
                        x2 = findmidPoint(rect.rect[0, 0, :], rect.rect[3, 0, :])
                        roiy[0,:] = (1+scale)*x2 - scale * x1
                    elif rect.order == cornerindex - 1- offset:
                        x2 = findmidPoint(rect.rect[0, 0, :], rect.rect[1, 0, :])
                        x1 = findmidPoint(rect.rect[2, 0, :], rect.rect[3, 0, :])
                        roix[0,:] = (1+scale)*x2 - scale *x1
            elif 1 in rectdic.keys():
                for rect in rectlist:
                    if rect.order == cornerindex - wstep -1- offset:
                        x1 = findmidPoint(rect.rect[0, 0, :], rect.rect[3, 0, :])
                        x2 = findmidPoint(rect.rect[1, 0, :], rect.rect[2, 0, :])
                        roiy[1,:] = (1+scale)*x2 - scale * x1
                    elif rect.order == cornerindex - offset:
                        x2 = findmidPoint(rect.rect[0, 0, :], rect.rect[1, 0, :])
                        x1 = findmidPoint(rect.rect[2, 0, :], rect.rect[3, 0, :])
                        roix[1,:] = (1+scale)*x2 - scale *x1
            elif 2 in rectdic.keys():
                for rect in rectlist:
                    if rect.order == cornerindex - wstep- offset:
                        x1 = findmidPoint(rect.rect[0, 0, :], rect.rect[1, 0, :])
                        x2 = findmidPoint(rect.rect[2, 0, :], rect.rect[3, 0, :])
                        roix[2,:] = (1+scale)*x2 - scale * x1
                    elif rect.order == cornerindex - 1- offset:
                        x2 = findmidPoint(rect.rect[1, 0, :], rect.rect[2, 0, :])
                        x1 = findmidPoint(rect.rect[0, 0, :], rect.rect[3, 0, :])
                        roiy[2,:] = (1+scale)*x2 - scale *x1
            elif 3 in rectdic.keys():
                for rect in rectlist:
                    if rect.order == cornerindex - wstep -1- offset:
                        x1 = findmidPoint(rect.rect[0, 0, :], rect.rect[1, 0, :])
                        x2 = findmidPoint(rect.rect[2, 0, :], rect.rect[3, 0, :])
                        roix[3,:] = (1+scale)*x2 - scale * x1
                    elif rect.order == cornerindex - offset:
                        x2 = findmidPoint(rect.rect[0, 0, :], rect.rect[3, 0, :])
                        x1 = findmidPoint(rect.rect[2, 0, :], rect.rect[1, 0, :])
                        roiy[3,:] = (1+scale)*x2 - scale *x1

        elif len(rectlist) == 2:
            rectdic = {0:'LT',1:'RT',2:'RB',3:'LB'}
            for rect in rectlist:
                if rect.order == cornerindex - wstep - 1- offset:
                    rectdic.pop(0,None)
                elif rect.order == cornerindex - wstep- offset:
                    rectdic.pop(1,None)
                elif rect.order == cornerindex- offset:
                    rectdic.pop(2,None)
                elif rect.order == cornerindex-1- offset:
                    rectdic.pop(3,None)
            if 0 in rectdic.keys() and 1 in rectdic.keys():
                for rect in rectlist:
                    if rect.order == cornerindex -1- offset:
                        x1 = findmidPoint(rect.rect[2, 0, :], rect.rect[3, 0, :])
                        x2 = findmidPoint(rect.rect[0, 0, :], rect.rect[1, 0, :])
                        roix[0,:] = (1+scale)*x2 - scale * x1
                        roix[3,:] = x2
                        roiy[0,:] = rect.rect[1,0,:]
                        roiy[3,:] = rect.rect[2,0,:]
                    elif rect.order == cornerindex - offset:
                        x2 = findmidPoint(rect.rect[0, 0, :], rect.rect[1, 0, :])
                        x1 = findmidPoint(rect.rect[2, 0, :], rect.rect[3, 0, :])
                        roix[1,:] = (1+scale)*x2 - scale *x1
                        roix[2,:] = x2
                        roiy[2,:] = rect.rect[3,0,:]
                        roiy[1,:] = rect.rect[0,0,:]


            elif 1 in rectdic.keys() and 2 in rectdic.keys():
                for rect in rectlist:
                    if rect.order == cornerindex - wstep -1- offset:
                        x1 = findmidPoint(rect.rect[0, 0, :], rect.rect[3, 0, :])
                        x2 = findmidPoint(rect.rect[2, 0, :], rect.rect[1, 0, :])
                        roiy[1,:] = (1+scale)*x2 - scale * x1
                        roiy[0,:] = x2
                        roix[0,:] = rect.rect[3,0,:]
                        roix[1,:] = rect.rect[2,0,:]
                    elif rect.order == cornerindex - 1- offset:
                        x2 = findmidPoint(rect.rect[2, 0, :], rect.rect[1, 0, :])
                        x1 = findmidPoint(rect.rect[0, 0, :], rect.rect[3, 0, :])
                        roiy[2,:] = (1+scale)*x2 - scale *x1
                        roiy[3,:] = x2
                        roix[2,:] = rect.rect[1,0,:]
                        roix[3,:] = rect.rect[0,0,:]

            elif 2 in rectdic.keys() and 3 in rectdic.keys():
                for rect in rectlist:
                    if rect.order == cornerindex -wstep -1- offset:
                        x1 = findmidPoint(rect.rect[0, 0, :], rect.rect[1, 0, :])
                        x2 = findmidPoint(rect.rect[2, 0, :], rect.rect[3, 0, :])
                        roix[3,:] = (1+scale)*x2 - scale * x1
                        roix[0,:] = x2
                        roiy[0,:] = rect.rect[1,0,:]
                        roiy[3,:] = rect.rect[2,0,:]
                    elif rect.order == cornerindex - wstep - offset:
                        x2 = findmidPoint(rect.rect[2, 0, :], rect.rect[3, 0, :])
                        x1 = findmidPoint(rect.rect[0, 0, :], rect.rect[1, 0, :])
                        roix[2,:] = (1+scale)*x2 - scale *x1
                        roix[1,:] = x2
                        roiy[2,:] = rect.rect[3,0,:]
                        roiy[1,:] = rect.rect[0,0,:]

            elif 3 in rectdic.keys() and 0 in rectdic.keys():
                for rect in rectlist:
                    if rect.order == cornerindex -wstep- offset:
                        x1 = findmidPoint(rect.rect[1, 0, :], rect.rect[2, 0, :])
                        x2 = findmidPoint(rect.rect[0, 0, :], rect.rect[3, 0, :])
                        roiy[0,:] = (1+scale)*x2 - scale * x1
                        roiy[1,:] = x2
                        roix[0,:] = rect.rect[3,0,:]
                        roix[1,:] = rect.rect[2,0,:]
                    elif rect.order == cornerindex- offset :
                        x2 = findmidPoint(rect.rect[0, 0, :], rect.rect[3, 0, :])
                        x1 = findmidPoint(rect.rect[1, 0, :], rect.rect[2, 0, :])
                        roiy[3,:] = (1+scale)*x2 - scale *x1
                        roiy[2,:] = x2
                        roix[2,:] = rect.rect[1,0,:]
                        roix[3,:] = rect.rect[0,0,:]

        elif len(rectlist) == 1:
            for rect in rectlist:
                if rect.order == cornerindex - wstep - 1- offset:
                    roix[0,:] = rect.rect[3,0,:]
                    roix[1,:] = rect.rect[2,0,:]
                    roiy[0,:] = rect.rect[1,0,:]
                    roiy[3,:] = rect.rect[2,0,:]


                    roix[2, :] = (1 + scale) * rect.rect[2,0,:] - scale * rect.rect[1,0,:]
                    roix[3, :] = (1 + scale) * rect.rect[3,0,:] - scale * rect.rect[0,0,:]
                    roiy[1, :] = (1 + scale) * rect.rect[1,0,:] - scale * rect.rect[0,0,:]
                    roiy[2, :] = (1 + scale) * rect.rect[2,0,:] - scale * rect.rect[3,0,:]

                elif rect.order == cornerindex - wstep- offset:
                    roix[0,:] = rect.rect[3,0,:]
                    roix[1,:] = rect.rect[2,0,:]
                    roiy[1,:] = rect.rect[0,0,:]
                    roiy[2,:] = rect.rect[3,0,:]

                    roix[2, :] = (1 + scale) * rect.rect[2,0,:] - scale * rect.rect[1,0,:]
                    roix[3, :] = (1 + scale) * rect.rect[3,0,:] - scale * rect.rect[0,0,:]
                    roiy[0, :] = (1 + scale) * rect.rect[0,0,:] - scale * rect.rect[1,0,:]
                    roiy[3, :] = (1 + scale) * rect.rect[3,0,:] - scale * rect.rect[2,0,:]

                elif rect.order == cornerindex- offset:
                    roix[3,:] = rect.rect[0,0,:]
                    roix[2,:] = rect.rect[1,0,:]
                    roiy[1,:] = rect.rect[0,0,:]
                    roiy[2,:] = rect.rect[3,0,:]

                    roix[0, :] = (1 + scale) * rect.rect[0,0,:] - scale * rect.rect[3,0,:]
                    roix[1, :] = (1 + scale) * rect.rect[1,0,:] - scale * rect.rect[2,0,:]
                    roiy[0, :] = (1 + scale) * rect.rect[0,0,:] - scale * rect.rect[1,0,:]
                    roiy[3, :] = (1 + scale) * rect.rect[3,0,:] - scale * rect.rect[2,0,:]
                elif rect.order == cornerindex-1- offset:
                    roix[3, :] = rect.rect[0, 0, :]
                    roix[2, :] = rect.rect[1, 0, :]
                    roiy[0, :] = rect.rect[1, 0, :]
                    roiy[3, :] = rect.rect[2, 0, :]

                    roix[0, :] = (1 + scale) * rect.rect[0, 0, :] - scale * rect.rect[3, 0, :]
                    roix[1, :] = (1 + scale) * rect.rect[1, 0, :] - scale * rect.rect[2, 0, :]
                    roiy[1, :] = (1 + scale) * rect.rect[1, 0, :] - scale * rect.rect[0, 0, :]
                    roiy[2, :] = (1 + scale) * rect.rect[2, 0, :] - scale * rect.rect[3, 0, :]
        else:
            print("NOT ENOUGH rects")


        return roix,roiy

    def __computeLineParams(self,roi,edgeImg,cornerindex,ori):
        xmin = np.min(roi[:,0])
        xmax = np.max(roi[:,0])
        ymin = np.min(roi[:,1])
        ymax = np.max(roi[:,1])

        w = xmax - xmin + 1
        h = ymax - ymin + 1
        crop_im = edgeImg[ymin:ymin+h,xmin:xmin+w]
        mask = np.zeros(crop_im.shape,dtype=np.uint8)
        roi_corners = np.array([[(roi[0,0]-xmin,roi[0,1]-ymin),(roi[1,0]-xmin,roi[1,1]-ymin),(roi[2,0]-xmin,roi[2,1]-ymin),(roi[3,0]-xmin,roi[3,1]-ymin)]],dtype=np.int32)
        channel_count = 1

        if len(edgeImg.shape) > 2:
            channel_count = edgeImg.shape[2]

        ignore_maske_color = (255,)*channel_count
        cv2.fillPoly(mask,roi_corners,ignore_maske_color)
        masked_image = cv2.bitwise_and(crop_im,mask)
        cv2.imwrite('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/logitect/result/roi/masked_'+str(cornerindex)+'.png',masked_image)
        print(str(cornerindex)+' is saved...')
        print(roi)

        x_d = []
        y_d = []
        for i in range(w):
            for j in range(h):
                if masked_image[j,i] > 100:
                    x_d.append(i+xmin)
                    y_d.append(j+ymin)
        x_data = np.array(x_d,dtype=float)
        y_data = np.array(y_d,dtype=float)

        imax = np.max(x_data)
        jmax = np.max(y_data)
        if imax < jmax:
            imax = jmax
        x_data = x_data/imax
        y_data = y_data/imax

        HInnerX, HInnerY, HOuterX, HOuterY, ibeta, ialpha = doRANSAC(x_data,y_data)
        initBeta = ibeta
        initAlpha = ialpha
        if ibeta > 1 or ibeta < -1:
            _dataY = np.array([HInnerX]).copy()
            _dataX = np.array([HInnerY]).copy()
            initBeta = 1 / ibeta
            initAlpha = - ialpha / ibeta
        else:
            _dataX = np.array([HInnerX])
            _dataY = np.array([HInnerY])
        x_data = _dataX
        y_data = _dataY


        x = Variable(torch.Tensor(x_data),requires_grad = False)
        y = Variable(torch.Tensor(y_data),requires_grad = False)

        #x*beta+alpha = y

        beta = Variable(torch.Tensor(np.array([initBeta])),requires_grad = True)
        alpha = Variable(torch.Tensor(np.array([initAlpha])),requires_grad = True)
        learning_rate = 1e-4
        optimizer = torch.optim.SGD([beta,alpha],lr=learning_rate)
        for t in range(1000):
            y_pred=x.mul(beta).add(alpha)
            loss = (y_pred - y).pow(2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        isVertical = False
        if math.fabs(beta.data.numpy()) <= sys.float_info.epsilon:
            isVertical = True

        elif ibeta > 1 or ibeta < -1:
            tmpBeta = beta
            beta = 1 / beta
            alpha = - alpha / tmpBeta
        return np.stack((beta.data.numpy(),alpha.data.numpy()*imax)),isVertical

    def __computeCrossPoint(self,roix,roiy,edgeImg,cornerindex):
        #[beta,alpha]
        paramx,isverticalX = self.__computeLineParams(roix,edgeImg,cornerindex,'x')
        paramy,isverticalY = self.__computeLineParams(roiy,edgeImg,cornerindex,'y')
        print(cornerindex,'==>',paramx,paramy,isverticalX,isverticalY)
        print('======================================================================')

        if not isverticalX and not isverticalY:
            return [(paramy[1]-paramx[1])/(paramx[0]-paramy[0]),(paramx[0]*paramy[1] - paramy[0]*paramx[1])/(paramx[0]-paramy[0])]
        elif isverticalX and not isverticalY:
            return [paramx[1],paramy[0]*paramx[1]+paramy[1]]
        elif isverticalY and not isverticalX:
            return [paramy[1],paramx[0]*paramy[1]+paramx[1]]
        else:
            print('WRONG line params')

    def __findCorners(self,goodChessboardGroups,img,scale):
        chessgroups = copy.deepcopy(goodChessboardGroups)

        filterimg = cv2.bilateralFilter(img,11,75,75)
        edge = cv2.Canny(filterimg, 80, 180)
        removeShortelements(edge)
        cv2.imwrite('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/logitect/result/roi/edge.png',edge)
        chessBoardAreaGroup = []
        for group in chessgroups:
            identifiedRect = None

            while(identifiedRect == None or not(identifiedRect.identified) ):
                index = random.randint(0,len(group)-1)
                if group[index].identified == True:
                    identifiedRect = group.pop(index)
                    break

            identifiedRect.order = 0
            corners = {}
            cornersSet = {'LT':0,'RT':self._chessboardsize[1]-1,'LB':(self._chessboardsize[0]-1)*self._chessboardsize[1],'RB':self._chessboardsize[0]*self._chessboardsize[1]-1}
            identifiedList = []
            identifiedList.append(identifiedRect)
            unidentifieddic = {}
            for i in range(len(group)):
                unidentifieddic[i] = group[i]

            while(len(unidentifieddic) > 0):
                for key in unidentifieddic.keys():
                    # reidenfiedlist = []
                    hasaround = False
                    for rect in identifiedList:
                        if self.__findRectsAround(rect,unidentifieddic[key]):
                            hasaround = True
                    # reidenfiedlist.append()

                    # for rect in reidenfiedlist:
                    if hasaround:
                        identifiedList.append(unidentifieddic.pop(key, None))




            # self.__findAround([identifiedRect],group)
            # group.append(identifiedRect)
            group = identifiedList
            for rect in group:
                if rect.iscorner:
                    if rect.hasAround[0] and rect.hasAround[1] and not(rect.hasAround[2]) and not(rect.hasAround[3]):
                        #cv2.line(img,(rect.rect[0,0,0]*2,rect.rect[0,0,1]*2),(rect.rect[1,0,0]*2,rect.rect[1,0,1]*2),(255,0,0),3)
                        corners['LB'] = rect
                    elif rect.hasAround[1] and rect.hasAround[2] and not(rect.hasAround[0]) and not(rect.hasAround[3]):
                        #cv2.line(img,(rect.rect[0,0,0]*2,rect.rect[0,0,1]*2),(rect.rect[1,0,0]*2,rect.rect[1,0,1]*2),(0,255,0),3)
                        corners['LT'] = rect
                    elif rect.hasAround[3] and rect.hasAround[2] and not (rect.hasAround[0]) and not (rect.hasAround[1]):
                        #cv2.line(img,(rect.rect[0,0,0]*2,rect.rect[0,0,1]*2),(rect.rect[1,0,0]*2,rect.rect[1,0,1]*2),(0,255,255),3)
                        corners['RT'] = rect
                    elif rect.hasAround[0] and rect.hasAround[3] and not(rect.hasAround[1]) and not(rect.hasAround[2]):
                        #cv2.line(img,(rect.rect[0,0,0]*2,rect.rect[0,0,1]*2),(rect.rect[1,0,0]*2,rect.rect[1,0,1]*2),(255,255,0),3)
                        corners['RB'] = rect

            # saveimage(img, 'originimage')

            offsetdict={}
            mean = 0
            for key in corners:
                offsetdict[key] = int(corners[key].order - cornersSet[key])
                mean += offsetdict[key]

            mean = mean/len(offsetdict)
            for key in offsetdict:
                if offsetdict[key] != mean:
                    print("WRONG CORNER INDEX!!!!!!!")

            index = -1
            for rect in group:
                rect.order -= mean
                if index == -1:
                    index = rect.chessboardindex
                elif index == rect.chessboardindex:
                    continue
                else:
                    print('WRONG patch chessboardindex')

            aBoardAreaGroup = ChessBoardArea(index)
            aBoardAreaGroup.rectgroup = group
            cornersDict = {}
            for rect in group:
                #center = np.array([np.mean(rect.rect[:,0,0]),np.mean(rect.rect[:,0,1])])
                #dist = np.mean((distofTwoPoints(center,rect.rect[0,0,:]),distofTwoPoints(center,rect.rect[1,0,:]),distofTwoPoints(center,rect.rect[2,0,:]),distofTwoPoints(center,rect.rect[3,0,:])))

                for i in range(4):

                    wstep = self._chessboardsize[1]
                    cornerindex = rect.order + i+ math.floor(rect.order/wstep)
                    if i == 2:
                        cornerindex = rect.order+wstep+2+math.floor(rect.order/wstep)
                    elif i == 3:
                        cornerindex = rect.order + wstep+math.floor(rect.order/wstep)+1


                    if cornerindex in cornersDict.keys():
                        continue
                    else:
                        #find cross point roi
                        rectlist = []
                        rectlist.append(rect)
                        for nrect in group:
                            if i == 0:
                                if (nrect.order == rect.order - wstep -1 or nrect.order == rect.order - wstep or nrect.order == rect.order -1) and rect.order%wstep != 0:
                                    rectlist.append(nrect)
                                elif (rect.order % wstep == 0) and nrect.order == rect.order - wstep:
                                    rectlist.append(nrect)
                            elif i == 1:
                                if (nrect.order == rect.order - wstep +1 or nrect.order == rect.order - wstep or nrect.order == rect.order +1) and (rect.order+1) %wstep != 0:
                                    rectlist.append(nrect)
                                elif (rect.order+1) %wstep == 0 and nrect.order == rect.order - wstep:
                                    rectlist.append(nrect)
                            elif i == 2:
                                if (nrect.order == rect.order + wstep +1 or nrect.order == rect.order + wstep or nrect.order == rect.order +1) and (rect.order+1) %wstep != 0:
                                    rectlist.append(nrect)
                                elif (rect.order+1) %wstep == 0 and nrect.order == rect.order + wstep:
                                    rectlist.append(nrect)
                            elif i == 3:
                                if (nrect.order == rect.order + wstep -1 or nrect.order == rect.order + wstep or nrect.order == rect.order -1) and rect.order%wstep != 0:
                                    rectlist.append(nrect)
                                elif (rect.order % wstep == 0) and nrect.order == rect.order + wstep:
                                    rectlist.append(nrect)

                        # if len(rectlist) == 4:
                            # FourCornersDict[cornerindex] = self.__find4CrossPoint(rectlist,cornerindex)
                            #cv2.circle(img, (int(FourCornersDict[cornerindex][0]*scale), int(FourCornersDict[cornerindex][1]*scale)), 2,(0, 0, 255), 1)

                        roix,roiy = self.__findROI(rectlist,cornerindex)
                        roix = roix * scale
                        roiy = roiy * scale
                        roix = roix.astype(int)
                        roiy = roiy.astype(int)



                        cornersDict[cornerindex] = self.__computeCrossPoint(roix,roiy,edge,cornerindex)
                        cv2.circle(img,(cornersDict[cornerindex][0],cornersDict[cornerindex][1]),2,(0,255,255),1)
                        # pts = np.array([roix[0,:],roix[1,:],roix[2,:],roix[3,:]],np.int32)
                        # pts = pts.reshape((-1,1,2))
                        # cv2.polylines(img,[pts],True,(0,255,255))
                        # pts = np.array([roiy[0,:],roiy[1,:],roiy[2,:],roiy[3,:]],np.int32)
                        # pts = pts.reshape((-1,1,2))
                        # cv2.polylines(img,[pts],True,(0,255,255))
            aBoardAreaGroup.cornersDict = cornersDict

            # for rect in group:
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     cv2.putText(img, str(rect.order), (int(rect.rect[0, 0, 0]*scale), int(rect.rect[0, 0, 1]*scale)), font, 1,(255, 255, 255), 2, cv2.LINE_AA)
            saveimage(img,'result')

            chessBoardAreaGroup.append(aBoardAreaGroup)
        return chessBoardAreaGroup



    def detect(self,img):
        height, width, channel = img.shape[:3]
        resizescale = 1.0
        colorEquilibrium(img)
        originimg = np.copy(img)
        originimg = enhanceColor(originimg)

        img = cv2.resize(img, (int(width / resizescale), int(height / resizescale)))
        height, width, channel = img.shape[:3]


        imgR = img[:, :, 2]
        imgG = img[:, :, 1]
        imgB = img[:, :, 0]

        imgRed = np.zeros(shape=(height, width), dtype="uint8")
        imgGreen = np.zeros(shape=(height, width), dtype="uint8")
        chessboardBound = np.zeros(shape=(height, width), dtype="uint8")

        for i in range(height):
            for j in range(width):
                iRed = float(imgR[i, j])
                iGreen = float(imgG[i, j])
                iBlue = float(imgB[i, j])
                if iRed - iGreen > 80  and iRed > 1.6 * iGreen and iRed > 100 :
                    imgRed[i, j] = 255
                    #imgGreen[i, j] = 255
                if iGreen - iRed > 40 :
                    imgGreen[i, j] = 255
                if imgRed[i,j] == 255 or imgGreen[i, j] == 255:
                    chessboardBound[i,j] = 255

        # edgeR = cv2.Canny(imgRed,0,255)
        # edgeG = cv2.Canny(imgGreen,0,255)
        # edge = cv2.Canny(chessboardBound,0,255)
        element = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        chessboardBound = cv2.erode(chessboardBound,element,dst=None,anchor=(-1,-1),iterations=2)
        chessboardBound = cv2.dilate(chessboardBound,element,dst=None,anchor=(-1,-1,),iterations=6)




        imgRed = cv2.dilate(imgRed,element,dst=None,anchor=(-1,-1),iterations=3)
        imgGreen = cv2.dilate(imgGreen, element, dst=None, anchor=(-1, -1), iterations=3)


        imgRed = cv2.erode(imgRed,element,dst=None,anchor=(-1,-1),iterations=5)
        imgGreen = cv2.erode(imgGreen, element, dst=None, anchor=(-1, -1), iterations=5)

        saveimage(imgRed,'imgRederod')
        saveimage(imgGreen,'greenerode')

        imre,contoursR,hierarchyR = cv2.findContours(imgRed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        imgr,contoursG,hierarchyG = cv2.findContours(imgGreen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contourimg = img.copy()
        for i in range(len(contoursR)):
            cv2.drawContours(contourimg,contoursR,i,(0,255,0))
        for i in range(len(contoursG)):
            cv2.drawContours(contourimg, contoursG, i, (0, 0, 255))
        saveimage(contourimg,'contourR')

        redPatchAreaList = self.__validateContour(contoursR,height,width)
        redRectGroups = self.__groupCorners(redPatchAreaList)

        greenPatchAreaList = self.__validateContour(contoursG,height,width)
        greenRectGroups = self.__groupCorners(greenPatchAreaList)

        chessboardGroups = self.__combineGroups(redRectGroups,greenRectGroups)
        goodchessboardGroups = self.__verifychessboardGroup(chessboardGroups,chessboardBound)

        self.__identifyRectArea(goodchessboardGroups,img)

        for group in redRectGroups:
            for rect in group:
                 # for i in range(rect.shape[0]):
                 cv2.circle(img,(rect[0,0,0],rect[0,0,1]),2,(0,0,255),2)
                 cv2.circle(img,(rect[1,0,0],rect[1,0,1]),2,(0,255,255),2)
                 cv2.circle(img,(rect[2,0,0],rect[2,0,1]),2,(0,255,0),2)
                 cv2.circle(img,(rect[3,0,0],rect[3,0,1]),2,(255,0,0),2)

        for group in greenRectGroups:
            for rect in group:
                # for i in range(rect.shape[0]):
                cv2.circle(img, (rect[0, 0, 0], rect[0, 0, 1]), 2, (0, 0, 255), 2)
                cv2.circle(img, (rect[1, 0, 0], rect[1, 0, 1]), 2, (0, 255, 255), 2)
                cv2.circle(img, (rect[2, 0, 0], rect[2, 0, 1]), 2, (0, 255, 0), 2)
                cv2.circle(img, (rect[3, 0, 0], rect[3, 0, 1]), 2, (255, 0, 0), 2)

        saveimage(img, 'img')

        return self.__findCorners(goodchessboardGroups,originimg,resizescale)



        # saveimage(imgRed, 'imgRed')
        # saveimage(imgGreen, 'imgGreen')
        #
        # saveimage(imgB, 'imgB')
        # saveimage(imgG, 'imgG')
        # saveimage(imgR, 'imgR')
        #

        #
        #
        #
        # #return []  #return a list contains all board corners array   [boarder ,index , x , y]
        # saveimage(chessboardBound, 'chessboardBound')
        # # showMat(edgeG, scale, 'edgeG')
        # # showMat(edgeR, scale, 'edgeR')
        #
        #
