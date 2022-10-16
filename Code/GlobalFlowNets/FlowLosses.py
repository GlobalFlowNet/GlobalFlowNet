import torch
import torch.nn as nn


class GlobalFlowLoss(nn.Module):
    def __init__(self, teacherNet, paras):
        super(GlobalFlowLoss, self).__init__()

        self.teacherNet = teacherNet

    def computeLoss(self, predFlow, gtFlow):
        pass

    def forward(self, optFlowNet, im1, im2):
        predFlow = optFlowNet.estimateFlow(im1, im2)
        with torch.no_grad():
            gtFlow = self.teacherNet.estimateFlow(im1, im2)
        loss = self.calculateLoss(predFlow, gtFlow)
        return loss


class CEPEFlowLoss(GlobalFlowLoss):
    def __init__(self, teacherNet, LOSSSWITCHITER, cutoff, startCF=10):
        super(CEPEFlowLoss, self).__init__(teacherNet)
        self.nrIters = 0
        self.LSIter = LOSSSWITCHITER
        self.cutoff = cutoff
        self.startCF = startCF

    def computeLoss(self, model, source, target, nrIters):
        gtFlow = self.teacherNet.estimateFlow(source, target)
        predFlow = model.estimateFlow(source, target)
        epe = torch.linalg.norm(predFlow - gtFlow, dim=1)
        # epeStep = smoothStepFunction(epe, cutoff)
        # epeTanh = torch.tanh(2*epe/cutoff)
        # itersRatio = max(nrIters, self.LSIter) / self.LSIter

        # if nrIters < self.LSIter:
        #     ptLoss = shiftedTanH(epe, c=self.startCF)
        # else:
        ptLoss = .9 * \
            smoothStepFunction(epe, self.cutoff, e=1e-4) + .1 * \
            shiftedTanH(epe, c=self.cutoff)

        loss = torch.mean(ptLoss)
        return loss


class RobustLoss(GlobalFlowLoss):
    def __init__(self, teacherNet, LOSSSWITCHITER, cutoff, startCF=10):
        super(RobustLoss, self).__init__(teacherNet)
        self.nrIters = 0
        self.LSIter = LOSSSWITCHITER
        self.cutoff = cutoff
        self.startCF = startCF

    def computeLoss(self, model, source, target, nrIters):
        gtFlow = self.teacherNet.estimateFlow(source, target)
        predFlow = model.estimateFlow(source, target)
        epe = torch.linalg.norm(predFlow - gtFlow, dim=1)
        # epeStep = smoothStepFunction(epe, cutoff)
        # epeTanh = torch.tanh(2*epe/cutoff)
        # itersRatio = max(nrIters, self.LSIter) / self.LSIter

        # if nrIters < self.LSIter:
        #     ptLoss = shiftedTanH(epe, c=self.startCF)
        # else:
        ptLoss = torch.pow(epe + 1e-4, .1)

        loss = torch.mean(ptLoss)
        return loss


class GoogleRobustLoss(nn.Module):
    def __init__(self, teacherNet, paras):
        super(GoogleRobustLoss, self).__init__()
        self.teacherNet = teacherNet
        self.a = paras['a']
        self.c = paras['c']

    def computeLoss(self, model, source, target, skips, nrIters):
        with torch.no_grad():
            gtFlow = self.teacherNet.estimateFlow(source, target)
        predFlow = model.estimateFlow(source, target)
        epe = torch.linalg.norm(predFlow - gtFlow, dim=1)

        epe = epe / skips[:, None, None, None]
        a = self.a
        c = self.c
        ptLoss = (abs(a - 2) / a) * \
            (torch.pow(((((epe / c)**2) / (abs(a - 2))) + 1), a / 2) - 1)

        loss = torch.mean(ptLoss)
        return loss


class AdaptiveGoogleRobustLoss(nn.Module):
    def __init__(self, teacherNet, paras):
        super(AdaptiveGoogleRobustLoss, self).__init__()
        self.teacherNet = teacherNet
        self.aStart = paras['aStart']
        self.cStart = paras['cStop']
        self.aStop = paras['aStart']
        self.cStop = paras['cStop']
        self.span = paras['span']

    def computeLoss(self, model, source, target, skips, nrIters):
        with torch.no_grad():
            gtFlow = self.teacherNet.estimateFlow(source, target)
        predFlow = model.estimateFlow(source, target)
        epe = torch.linalg.norm(predFlow - gtFlow, dim=1)

        epe = epe / skips[:, None, None, None]

        r = min(1, nrIters / self.span)

        a = (1 - r) * self.aStart + r * self.aStop
        c = (1 - r) * self.cStart + r * self.cStop

        ptLoss = (abs(a - 2) / a) * \
            (torch.pow(((((epe / c)**2) / (abs(a - 2))) + 1), a / 2) - 1)

        loss = torch.mean(ptLoss)
        return loss


def smoothStepFunction(x, c, e):
    x = (1 + (x - c) / (torch.sqrt((x - c)**2 + e))) / 2
    return x


def shiftedTanH(x, c):
    x = torch.tanh(2 * x / c)
    return x
