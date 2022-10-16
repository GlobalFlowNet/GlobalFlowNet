import random

import numpy as np


class Augmenter():
    augmenters = None
    cropSize = None

    def __init__(self, cropSize, randomFlip=True, randcomChannel=True):
        self.cropSize = cropSize
        self.randcomChannel = randcomChannel
        self.randomFlip = randomFlip

    def transform(self, imgs):
        isList = type(imgs) == list
        if isList is False:
            imgs = [imgs]
        imgs = randomCrop(imgs, self.cropSize)
        # imgs = randomFlip(imgs)
        # imgs = randomChannelPermute(imgs)

        if isList is False:
            return imgs[0]
        else:
            return imgs[0], imgs[1]


def randomChannelPermute(imgs):
    order = np.random.permutation(3)

    permutedImgs = []

    for img in imgs:
        img = img[:, :, order]
        permutedImgs.append(img)

    return permutedImgs


def randomFlip(imgs):
    hFlip = random.randint(0, 1)
    vFlip = random.randint(0, 1)

    flipImgs = imgs

    if hFlip:
        for i in range(len(flipImgs)):
            flipImgs[i] = np.flip(flipImgs[i], axis=1).copy()

    if vFlip:
        for i in range(len(flipImgs)):
            flipImgs[i] = np.flip(flipImgs[i], axis=0).copy()

    return flipImgs


def randomCrop(imgs, cropSize):
    im1 = imgs[0]
    imgSize = im1.shape
    h = cropSize[0]
    w = cropSize[1]
    r = random.randint(0, imgSize[0] - cropSize[0])
    c = random.randint(0, imgSize[1] - cropSize[1])

    croppedImgs = []

    for img in imgs:
        img = img[r:r + h, c:c + w]
        croppedImgs.append(img)

    return croppedImgs
