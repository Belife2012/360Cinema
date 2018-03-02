import cv2



class ChessBoardArea(object):
    def __init__(self,index=-1):
        self.index = index
        self.rectgroup = None
        self.cornersDict = None