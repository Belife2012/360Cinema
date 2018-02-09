import cv2
import numpy as np
def main():
    # Load an color image in grayscale
    img = cv2.imread(
        '/Users/zhaochangkai/Books/Computer_Vision/Multiple_View_Geometry_in_Computer_Vision__2nd_Edition/Project/360cinema/testpatchs2/Green227V.jpg',
        1)
    img2 = np.copy(img)
    height,width,cn = img.shape[:3]


    imgred = img[:,:,2]
    imggreen = img[:,:,1]
    imgblue = img[:,:,0]
    cv2.imshow('red',imgred)
    cv2.imshow('green',imggreen)
    cv2.imshow('blue',imgblue)


    meanRed = np.mean(img[:,:,2])
    eq = np.zeros(img.shape,img.dtype)
    if meanRed > 100:
        eq = cv2.equalizeHist(img[:,:,2])
    else:
        eq = cv2.equalizeHist(img[:,:,1])
    cv2.imshow('eq', eq)

    img[:,:,0] = eq
    img[:,:,1] = eq
    img[:,:,2] = eq



    # for i in range(height):
    #     for j in range(width):
    #         if np.mean(img[i,j,:]) < 128:
    #             img[i,j,:] = [255,255,255]
    #         else:
    #             img[i,j,:] = [0,0,0]

    cv2.imshow('pathch',img)
    cv2.imshow('img',img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print ('Done')


if __name__ == '__main__':
    main()