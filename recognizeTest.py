import argparse
import os
from model import Model
from torch.autograd import Variable
import torchvision
from dataset import Dataset
import torch.utils.data
from PIL import Image
import cv2
import numpy as np
import json

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
class mEvaluator(object):
    def __init__(self, image_path):
        self._image = Image.open(image_path)
        self._image_path = '/home/roby/Desktop/digitClassifier/data/testpatchs2/'
    def evaluate(self, model):
        model.eval()
        data = {}
        for name in os.listdir(self._image_path):
            if name.endswith('.jpg'):
                img = cv2.imread(self._image_path+name)
                img = augumentpatch(img)
                height,width,cn = img.shape[:3]
                # for i in range(height):
                #     for j in range(width):
                #         if np.mean(img[i,j,:]) < 128:
                #             img[i,j,:] = [255,255,255]
                #         else:
                #             img[i,j,:] = [0,0,0]


                img = cv2.normalize(img.astype('float'),None,0.0,1.0,cv2.NORM_MINMAX)
                image_tenser = torch.FloatTensor(3,54,54).zero_()
                image_tenser[0] = torch.from_numpy(img[:,:,0])
                image_tenser[1] = torch.from_numpy(img[:,:,1])
                image_tenser[2] = torch.from_numpy(img[:,:,2])


                # transform = torchvision.transforms.ToTensor()
                # image_tenser = transform(self._image)
                image_array_tensor = torch.FloatTensor(1,3,54,54).zero_()
                image_array_tensor[0] = image_tenser
                image = Variable(image_array_tensor)
                length_logits, digits_logits = model(image)
                length_predictions = length_logits.data.max(1)[1]
                digits_predictions = [digit_logits.data.max(1)[1] for digit_logits in digits_logits]


                result = []
                for i in range(5):
                    if digits_predictions[i][0] != 10:
                        result.append(digits_predictions[i][0])
                #return [digits_predictions[0],digits_predictions[1],digits_predictions[2],digits_predictions[3],digits_predictions[4]]
                if name.endswith('L.jpg')or name.endswith('R.jpg')or name.endswith('V.jpg'):
                    if (result[0]==3 or result[0] == 4) and len(result) == 3 :
                        data[name] = result
                else:
                    if (result[0] != 3 and result[0] != 4) or len(result) <3:
                        data[name] = result
        with open('/home/roby/Desktop/digitClassifier/result5.txt','w') as outfile:
            json.dump(data,outfile)

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str, help='path to evaluate checkpoint, e.g. ./logs/model-100.tar')
parser.add_argument('-d', '--image_path', default='./data', help='directory to read image')


def _eval(path_to_checkpoint_file, image_path):
    model = Model()
    model.load(path_to_checkpoint_file)
    #model.cuda()
    mEvaluator(image_path).evaluate(model)
    #print (result)


def main(args):

    image_path = os.path.join(args.image_path)
    path_to_checkpoint_file = args.checkpoint

    print 'Start evaluating'
    # _eval(path_to_checkpoint_file, path_to_train_lmdb_dir)
    # _eval(path_to_checkpoint_file, path_to_val_lmdb_dir)
    _eval(path_to_checkpoint_file, image_path)
    print 'Done'


if __name__ == '__main__':
    main(parser.parse_args())