import torch
from PathStabilizers.StdPathStabilizerQP import smoothPathStdQP
from torch.nn.functional import grid_sample
from Utils.AffineUtility import AffineUility
from Utils.CroppingUtil import cropFrames

CHUNKLEN = 15


class JoinedAdaptiveGNetStabilizer():
    def __init__(self, frames, OptNet, dSpan=48, crop=.8):
        self.frames = frames
        self.dSpan = dSpan
        self.crop = crop
        self.shape = [self.frames.shape[-2], self.frames.shape[-1]]
        self.OptNet = OptNet
        self.OptNet = self.OptNet.eval().cuda()
        self.AffineUtil = AffineUility(self.shape)

    def getStabilizedFrames(self):
        fwdCoeffs = self.getFlowCoeffs(self.frames, dire=1)
        # fwdCoeffs = FeaturePtUtility().getTransforms(self.frames).cuda()

        # fwdCoeffs[:,0] = fwdCoeffsFP[:,0]
        # fwdCoeffs[:,-1] = fwdCoeffsFP[:,-1]
        # fwdCoeffs[:,-1] =0.0

        stabCoeffs = self.getStabilizationCoeffs(fwdCoeffs)
        nrFrames = self.frames.shape[0]
        stabFrames = torch.zeros_like(self.frames)
        mask = torch.tensor(True).cuda()
        with torch.no_grad():
            i = 0
            while i < nrFrames:
                stopIdx = min(i + CHUNKLEN, nrFrames)
                batchFrames = self.frames[i:stopIdx].clone().cuda()
                wrpFeilds = self.AffineUtil.getAffineGrids(stabCoeffs[i:stopIdx])
                stabFrames[i:stopIdx] = grid_sample(batchFrames, wrpFeilds).data.cpu()
                i = stopIdx

                outSide = torch.abs(wrpFeilds) > 1
                outSide = torch.sum(outSide, dim=[0, -1]) > 0
                inside = torch.logical_not(outSide)
                mask = torch.logical_and(mask, inside)

        mask = 1.0 * mask
        stabFrames, cropFeild = cropFrames(stabFrames, mask)

        return stabFrames, cropFeild

    def getFlowCoeffs(self, frames, dire=1):
        flowCoeffs = torch.zeros((frames.shape[0], 4)).cuda()

        with torch.no_grad():
            i = 1
            while i < frames.shape[0]:
                stopIdx = min(i + CHUNKLEN, frames.shape[0])
                batchFrames = frames[i - 1:stopIdx].clone().cuda()
                sourceFrames = batchFrames[:-1]
                targetFrames = batchFrames[1:]
                flows = self.OptNet.estimateFlowFull(sourceFrames, targetFrames)
                flowCoeffs[i:stopIdx] = self.AffineUtil.getFlowCoeffs(flows)
                i = stopIdx
            for iRes in range(stopIdx - 1, frames.shape[0]):
                flowCoeffs[iRes] = flowCoeffs[stopIdx - 2]
            # flowCoeffs = torch.from_numpy(np.load('R0.npy').astype('float32')).cuda()
        return flowCoeffs

    def getStabilizationCoeffs(self, fwdCoeffs):
        coeffs = fwdCoeffs
        cumCoeffs = torch.cumsum(coeffs, dim=0)

        stabCoeffs = smoothPathStdQP(cumCoeffs.cpu().T, w=self.shape[-1], h=self.shape[-2], minOverlap=self.crop).T.cuda()
        stabCoeffsRes = cumCoeffs - stabCoeffs
        return stabCoeffsRes

    def getOptimalSpans(self):
        return self.optimalSpans
