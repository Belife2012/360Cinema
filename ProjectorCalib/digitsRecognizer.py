
from model import Model
from torch.autograd import Variable
import numpy as np
import torch.utils.data
import cv2


def augumentpatch(img):
    meanRed = np.mean(img[:,:,2])
    equalizedmap = np.zeros(img.shape,img.dtype)
    if meanRed > 120:
        #red patch
        equalizedmap = img[:,:,2]#cv2.equalizeHist(img[:,:,2])
    else:
        #green patch
        equalizedmap = img[:,:,1]#cv2.equalizeHist(img[:,:,1])
    img[:,:,0] = equalizedmap
    img[:,:,1] = equalizedmap
    img[:,:,2] = equalizedmap

    return img
class DigitsRecognizer(object):
    def __init__(self, modelpath):
        #self._image = Image.open(image_path)
        self.__model = Model()
        self.__model.load(modelpath)
    def evaluate(self, img):

        self.__model.eval()

        img = augumentpatch(img)

        img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        image_tenser = torch.FloatTensor(3, 54, 54).zero_()
        image_tenser[0] = torch.from_numpy(img[:, :, 0])
        image_tenser[1] = torch.from_numpy(img[:, :, 1])
        image_tenser[2] = torch.from_numpy(img[:, :, 2])

        # transform = torchvision.transforms.ToTensor()
        # image_tenser = transform(self._image)
        image_array_tensor = torch.FloatTensor(1, 3, 54, 54).zero_()
        image_array_tensor[0] = image_tenser
        image = Variable(image_array_tensor)
        length_logits, digits_logits = self.__model(image)
        length_predictions = length_logits.data.max(1)[1]
        digits_predictions = [digit_logits.data.max(1)[1] for digit_logits in digits_logits]

        result = []
        for i in range(5):
            if digits_predictions[i][0] != 10:
                result.append(digits_predictions[i][0])
        if len(result) == 3:
            return [result[0],result[1], result[2]]
        else:
            return []