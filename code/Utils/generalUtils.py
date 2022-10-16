import torch
import numpy as np
from datetime import datetime
import sys
import os
from GeneralConfig import WORKINGDIR, CUDADEVICES
from shutil import copyfile
import copy
import torch.nn as nn
import math
from torch.optim import lr_scheduler

def seed():
	torch.manual_seed(0)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	import numpy as np
	np.random.seed(0)
	print('Seeded')


def getCurrentTimeStamp():
    return datetime.now().strftime("%d-%m-%Y:%H:%M:%S")


def deleteLastLine():
    "Use this function to delete the last line in the STDOUT"
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')


def prepare(modelDir):
    saveDir = getCurrentTimeStamp()
    modelDir = os.path.join(WORKINGDIR, modelDir,saveDir)
    os.mkdir(modelDir)
    print("Model Dir: ", modelDir)
    return modelDir


def saveModel(model, modelDir, counter):
    if type(model) is nn.DataParallel:
        model = model.module

    modelWts = copy.deepcopy(model.state_dict())
    torch.save(modelWts, os.path.join(modelDir, str(counter)+'.pth'))

def createLRScheduler(optimizer, maxIters, step):
    milestones = []
    start = step
    for milestone in range(start,maxIters,step):
        milestone = math.ceil(milestone)
        milestones.append(milestone)


    lrScheduler = lr_scheduler.MultiStepLR(optimizer, milestones = milestones , gamma=0.5)
    return lrScheduler