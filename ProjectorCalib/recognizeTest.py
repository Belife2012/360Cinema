import argparse
import os
from model import Model
from torch.autograd import Variable
import torchvision
from dataset import Dataset
import torch.utils.data
from PIL import Image
import cv2

class mEvaluator(object):
    def __init__(self, image_path):
        #self._image = Image.open(image_path)
        self._imagePath = '/Users/zhaochangkai/Books/Computer_Vision/Multiple_View_Geometry_in_Computer_Vision__2nd_Edition/Project/360cinema/testpatchs/'
    def evaluate(self, model):

        model.eval()
        for name in os.listdir(self._imagePath):
            if name.endswith('.jpg'):
                name = 'Green0'
                img = cv2.imread(self._imagePath+name)
                #img = cv2.imread('/Users/zhaochangkai/Books/Computer_Vision/Multiple_View_Geometry_in_Computer_Vision__2nd_Edition/Project/360cinema/pySphereCinema/SVHNClassifier-PyTorch-master/images/test12.jpg')
                img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
                #transform = torchvision.transforms.ToTensor()
                #image_tenser = transform(self._image)
                image_tenser = torch.FloatTensor(3,54,54).zero_()

                image_tenser[0]=torch.from_numpy(img[:,:,0])
                image_tenser[1]=torch.from_numpy(img[:,:,1])
                image_tenser[2]=torch.from_numpy(img[:,:,2])

                image_array_tensor = torch.FloatTensor(1,3,54,54).zero_()
                image_array_tensor[0] = image_tenser
                image = Variable(image_array_tensor)
                length_logits, digits_logits = model(image)
                length_predictions = length_logits.data.max(1)[1]
                digits_predictions = [digit_logits.data.max(1)[1] for digit_logits in digits_logits]


                result = []
                for i in range(5):
                    if digits_predictions[i] != 10:
                        result.append()
        return [digits_predictions[0],digits_predictions[1],digits_predictions[2],digits_predictions[3],digits_predictions[4]]

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str, help='path to evaluate checkpoint, e.g. ./logs/model-100.tar')
parser.add_argument('-d', '--image_path', default='./data', help='directory to read image')


def _eval(path_to_checkpoint_file, image_path):
    model = Model()
    model.load(path_to_checkpoint_file)
    #model.cuda()

    result = mEvaluator(image_path).evaluate(model)
    print (result)


def main(args):

    image_path = os.path.join(args.image_path)
    path_to_checkpoint_file = args.checkpoint

    print('Start evaluating')
    # _eval(path_to_checkpoint_file, path_to_train_lmdb_dir)
    # _eval(path_to_checkpoint_file, path_to_val_lmdb_dir)
    _eval(path_to_checkpoint_file, image_path)
    print ('Done')


if __name__ == '__main__':
    main(parser.parse_args())