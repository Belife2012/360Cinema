import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import random
# import calendar
# import time
import math
import sys

def getdata(img):
    w = img.shape[1]
    h = img.shape[0]

    x_d = []
    y_d = []

    for i in range(w):
        for j in range(h):
            if img[j, i] > 100:
                x_d.append(i)
                y_d.append(j)
    x_data = np.array(x_d, dtype=float)
    y_data = np.array(y_d, dtype=float)

    imax = np.max(x_data)
    jmax = np.max(y_data)
    if imax < jmax:
        imax = jmax
    x_data = x_data / imax
    y_data = y_data / imax

    return x_data,y_data
def getparams(x_data,y_data,ibeta,ialpha):

    initBeta = ibeta
    initAlpha = ialpha
    if ibeta > 1 or ibeta<-1:
        _dataY = np.array([x_data]).copy()
        _dataX = np.array([y_data]).copy()
        initBeta = 1/ibeta
        initAlpha = - ialpha/ibeta
    else:
        _dataX = np.array([x_data])
        _dataY = np.array([y_data])
    x_data = _dataX
    y_data = _dataY


    x = Variable(torch.Tensor(x_data), requires_grad=False)
    y = Variable(torch.Tensor(y_data), requires_grad=False)

    # x*beta+alpha = y

    beta = Variable(torch.Tensor(np.array([initBeta])), requires_grad=True)
    alpha = Variable(torch.Tensor(np.array([initAlpha])), requires_grad=True)
    learning_rate = 1e-4
    optimizer = torch.optim.SGD([beta, alpha], lr=learning_rate)
    for t in range(1000):
        y_pred = x.mul(beta).add(alpha)
        loss = (y_pred - y).pow(2).sum()
        if t == 999:
            print('LOSS:',loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    isVertical = False
    if math.fabs(beta.data.numpy()) <= sys.float_info.epsilon:
        isVertical = True

    elif ibeta > 1 or ibeta < -1:
        tmpBeta = beta
        beta =  1/beta
        alpha = - alpha/tmpBeta


    return beta,alpha,isVertical

def findTopNindex(arr,N):
    return np.argsort(arr)[::-1][:N]

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
    # return innerX and Y, OUT X and Y ,Params Beta and ALPHA ,isvertical and param is alpha
def main():

    for cornerindex in range(221):
        crop_imx = cv2.imread('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/patterns/cropx' + str(cornerindex) + '.0.png',0)
        maskx = cv2.imread('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/patterns/maskx' + str(cornerindex) + '.0.png',0)
        masked_imagex = cv2.imread('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/patterns/cleanx' + str(cornerindex) + '.0.png', 0)


        crop_imy = cv2.imread('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/patterns/cropy' + str(cornerindex) + '.0.png', 0)
        masky = cv2.imread('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/patterns/masky' + str(cornerindex) + '.0.png', 0)
        masked_imagey = cv2.imread('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/patterns/cleany' + str(cornerindex) + '.0.png', 0)

        Hdata_x,Hdata_y = getdata(masked_imagex)
        Vdata_x,Vdata_y = getdata(masked_imagey)

        HInnerX, HInnerY, HOuterX, HOuterY, Hbeta, Halpha = doRANSAC(Hdata_x,Hdata_y)
        VInnerX, VInnerY, VOuterX, VOuterY, Vbeta, Valpha = doRANSAC(Vdata_x,Vdata_y)

        print(cornerindex,'H')
        Hbeta,Halpha,Hisvertical = getparams(HInnerX,HInnerY,Hbeta,Halpha)
        print('-----------------')
        print(cornerindex,'V')
        Vbeta,Valpha,Visvertical = getparams(VInnerX,VInnerY,Vbeta,Valpha)
        print('====================================')
        Hbeta = Hbeta.data.numpy()
        Halpha = Halpha.data.numpy()
        Vbeta = Vbeta.data.numpy()
        Valpha = Valpha.data.numpy()

        fig = plt.figure()
        a = fig.add_subplot(121)
        a.plot(HInnerX,HInnerY,'r,')
        a.plot(HOuterX,HOuterY,'g,')
        Hminx = np.min(Hdata_x)
        Hmaxx = np.max(Hdata_x)
        a.plot([Hminx,1],[Halpha+Hminx*Hbeta,Halpha+Hmaxx*Hbeta])
        ratioa = float(masked_imagex.shape[0])/float(masked_imagex.shape[1])
        a.set_aspect(1)
        #plt.subplot(1,2,1)
        #plt.plot(Hdata_x,Hdata_y,'r+')
        #plt.axis([0,1,0,1])

        b = fig.add_subplot(122)
        b.plot(VInnerX,VInnerY,'r,')
        b.plot(VOuterX,VOuterY,'g,')
        Vminx = np.min(Vdata_x)
        Vmaxx = np.max(Vdata_x)
        b.plot([Vminx,Vmaxx],[Valpha+Vminx*Vbeta,Valpha+Vmaxx*Vbeta])
        ratiob = float(masked_imagey.shape[0]) / float(masked_imagey.shape[1])
        b.set_aspect(1)

        # plt.subplot(1,2,2)
        # plt.plot(Vdata_x,Vdata_y,'r+')
        # plt.axis([0,1,0,1])

        # plt.show(
        plt.savefig('/home/roby/Desktop/digitClassifier/SVHNClassifier-PyTorch-master/images/figures/' + str(int(cornerindex)) + '.png')




if __name__ == '__main__':
    main()