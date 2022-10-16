import numpy as np
import torch

from Stabilizers.JoinedAdaptiveGNetStabilizer import JoinedAdaptiveGNetStabilizer
from Stabilizers.MSPhotometric import MSPhotometric

CHUNKLEN = 7
class ComposedStabilizer():
    def __init__(self, Video, OptNet,stabilizers = [], span = 11, cutoffFreq = 5, crop=.8):
        self.Video = Video
        self.span = span
        self.frames = torch.from_numpy(np.transpose(Video.getFrames(),(0,3,1,2)))/255.0
        self.shape = [self.frames.shape[-2], self.frames.shape[-1]]
        self.optimalSpans = None
        self.cropFeild = None

        self.stabilizers = []
        for stabilizer in stabilizers:
            if stabilizer == 'GNetAffine':
                AffineStabilizer = JoinedAdaptiveGNetStabilizer(self.frames, OptNet, dSpan=32, crop=crop)
                self.stabilizers.append(AffineStabilizer)
                self.frames, self.cropFeild = AffineStabilizer.getStabilizedFrames()
            elif stabilizer == 'MSPhotometric':
                Stabilizer = MSPhotometric(self.frames, OptNet, span=13, cutoffFreq=5)
                Stabilizer.stabilizeFrames()
                

    def cvtTensor2Uint8(self, img):
        img = np.round(255*np.transpose(img.numpy(),(1,2,0))).astype('uint8')
        return img
    

    def getStabilizedFrame(self, idx):
        return self.cvtTensor2Uint8(self.frames[idx])
