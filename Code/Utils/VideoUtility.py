import glob
import hashlib
import math
import os
import shutil

import cv2
import numpy as np
from numpy.lib.npyio import load
from Utils.hashUtil import md5


class VideoReader:

    def __init__(self, path, loadAllFrames=True, maxFrames=math.inf):
        self.path = path
        self.video = cv2.VideoCapture(self.path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.nrFrames = min(int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)), maxFrames)
        if loadAllFrames:
            self.frames = self.extractFrames()
        else:
            self.frames = None

    def getFrame(self, idx):
        return self.frames[idx]

    def getFrames(self, startIdx=0, stopIdx=None):
        if stopIdx is None:
            stopIdx = self.nrFrames
        return self.frames[startIdx: stopIdx]

    def extractFrames(self):
        frames = np.zeros((self.nrFrames, self.height, self.width, 3), dtype='uint8')

        for i in range(self.nrFrames):
            frame = self.getNextFrame()
            if frame is None:
                self.nrFrames = i
                break

            frames[i] = frame

        return frames

    def getNextFrame(self):
        ret, frame = self.video.read()
        if frame is not None:
            frame = np.flip(frame, axis=2)
        return frame

    def getFPS(self):
        return self.fps

    def getSize(self):
        return self.width, self.height

    def getNrFrames(self):
        return int(self.nrFrames)

    def close(self):
        self.video.release()


class VideoWriter:
    video = None

    def __init__(self, path, fps=30):
        self.path = path
        self.fps = fps

    def writeFrame(self, frame):
        if frame.dtype != 'uint8':
            frame = (np.round(255 * frame)).astype('uint8')

        frame = np.flip(frame, axis=2)

        if (self.video is None):
            h, w = frame.shape[0], frame.shape[1]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
            if os.path.exists(self.path) is False:
                print('Video folder does not exist')

        self.video.write(frame)

    def close(self):
        self.video.release()
