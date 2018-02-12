

class PatchArea(object):
    def __init__(self,color,rect,index=-1,order=-1,isCorner = False):
        self.color = color
        self.rect = rect       # [leftup,rightup,rightdown,leftdown]
        self.chessboardindex = index  #  chessboard  index
        self.order = order  # in chessboard
        self.iscorner = isCorner
        self.isAssitance = False
        self.possibleChessboardindexdict = {}
        self.identified = False
        self.hasAround = [False,False,False,False] #[North,East,Sourth,West]
