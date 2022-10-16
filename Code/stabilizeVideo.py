import argparse
import json
import os
import random

import numpy as np
from GlobalFlowNets.GlobalPWCNets import getGlobalPWCModel
from Stabilizers.ComposedStabilizer import ComposedStabilizer
from Utils.VideoUtility import VideoReader, VideoWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
random.seed(27)
np.random.seed(27)


def getConfig(filePath):
    with open(filePath, 'r') as openfile:
        config = json.load(openfile)
    return config


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--inpVideoPath', dest='inpVideoPath', default='inputs/sample.avi')
    parser.add_argument('--outVideoPath', dest='outVideoPath', default='outputs/sample.avi')
    parser.add_argument('--maxAffineCrop', dest='maxAffineCrop', default=0.8, type=float)

    args = parser.parse_args()

    inpVideo = VideoReader(args.inpVideoPath, maxFrames=30)
    outVideo = VideoWriter(args.outVideoPath, fps=inpVideo.getFPS())

    Stabilizers = {}
    Stabilizers['GNetAffine'] = ['GNetAffine']
    Stabilizers['MSPhotometric'] = ['MSPhotometric']
    Stabilizers['GNetMSPhotometric'] = ['GNetAffine', 'MSPhotometric']

    inpTag = 'Original'
    outTag = 'GNetMSPhotometric'
    modelTag = 'GLNoWarp4YTBB'
    maxAffineCrop = .8

    config = getConfig('GlobalFlowNets/trainedModels/config.json')['GlobalNetModelParameters']
    OptNet = getGlobalPWCModel(config, 'GlobalFlowNets/trainedModels/GFlowNet.pth')
    OptNet = OptNet.eval().cuda()
    stabilizer = ComposedStabilizer(inpVideo, OptNet, stabilizers=Stabilizers[outTag], crop=args.maxAffineCrop)

    for f in range(inpVideo.getNrFrames()):
        outFrame = stabilizer.getStabilizedFrame(f)
        outVideo.writeFrame(outFrame)
    outVideo.close()
