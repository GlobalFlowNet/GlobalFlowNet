
import torch
import torch.nn.functional as F
from Utils.CroppingUtil import cropFrames
from Utils.DCTUtility import DCTUtility
from Utils.FilteringUtils import getGuassianKernel
from Utils.PolyomialFit import PolyFit

CHUNKLEN = 15


class MSPhotometric():
    def __init__(self, frames, OptNet, span=11, cutoffFreq=5):
        self.frames = frames
        self.span = span
        self.shape = [self.frames.shape[-2], self.frames.shape[-1]]
        self.OptNet = OptNet
        self.OptNet = self.OptNet.eval().cuda()
        self.cf = cutoffFreq
        self.DCTUtil = DCTUtility(self.shape, cutoffFreq)
        self.PolyFit = PolyFit()

        # self.stabilizeFrames()

    def stabilizeFrames(self):
        frames = self.frames
        stabFrames = torch.zeros_like(frames)
        mask = torch.tensor(True).cuda()
        for f in range(1, frames.shape[0] - 1):
            lSpan = min(f, self.span)
            rSpan = min(self.span, frames.shape[0] - 1 - f)
            span = min(lSpan, rSpan)
            stabFrame, wrpFeild = self.getStabilizedFrame(frames[f - span:f + span + 1], span)
            frames[f] = stabFrame.data.cpu()

            outSide = torch.abs(wrpFeild) > 1
            outSide = torch.sum(outSide, dim=[0, -1]) > 0
            inside = torch.logical_not(outSide)
            mask = torch.logical_and(mask, inside)

        mask = 1.0 * mask

        self.frames = cropFrames(frames, mask)
        self.mask = mask

    def warp(self, img, stabCoeffs):
        grid = self.DCTUtil.cvtFlowCoeffs2Grid(stabCoeffs)
        scale = torch.tensor([img.shape[-1], img.shape[-2]]).cuda()
        grid = (2 * grid - scale[:, None, None]) / scale[:, None, None]
        grid = torch.swapdims(grid[:, :, :, None], 0, 3)
        stabFrame = torch.squeeze(F.grid_sample(img[None], grid))
        return stabFrame, grid

    def findPolyApprox(self, coeffs, refIdx, degree=3):
        stabCoeffs = torch.zeros_like(coeffs[0])
        if coeffs.shape[0] < degree * 4:
            return coeffs[refIdx]

        for i in range(2):
            for u in range(stabCoeffs.shape[-2]):
                for v in range(stabCoeffs.shape[-1]):
                    s = coeffs[:, i, u, v]
                    sP = self.PolyFit.fit(degree, s, refIdx)
                    stabCoeffs[i, u, v] = sP[refIdx]

        return stabCoeffs

    def weightedMeanApprox(self, coeffs, photometricWts, refIdx):
        span = min(refIdx, coeffs.shape[0] - 1 - refIdx)
        photometricWts = photometricWts[refIdx - span:refIdx + span + 1]
        photometricWts = torch.minimum(photometricWts, torch.flip(photometricWts, dims=[0]))
        photometricWts = torch.exp(-photometricWts**2 / (2 * .1**2))
        photometricWts = photometricWts / photometricWts.sum()
        coeffs = coeffs[refIdx - span:refIdx + span + 1]
        # dists = torch.abs(torch.linspace(-span,span,2*span+1)).cuda()
        # dists[span] = 1e6
        # stabCoeffs = stabCoeffs/dists[:, None, None, None]
        gK = getGuassianKernel(span, span / 3.0).cuda()
        gK[span] = 0
        gK = gK / gK.sum()
        wts = gK * photometricWts
        # wts = gK
        wts = wts / wts.sum()
        stabCoeffs = torch.tensordot(coeffs, wts, dims=([0], [0]))
        return stabCoeffs

    def getStabilizedFrame(self, frames, refIdx):
        span = min(refIdx, frames.shape[0] - 1 - refIdx)
        refFrame = frames[refIdx, None].cuda()
        nrFrames = frames.shape[0]

        coeffs = torch.zeros((nrFrames, 2, self.cf, self.cf)).cuda()
        # flows = torch.zeros((nrFrames,2,self.cf,self.cf)).cuda()

        photometricWts = torch.ones(nrFrames).cuda()
        with torch.no_grad():
            i = 0
            while i < nrFrames:
                stopIdx = min(i + CHUNKLEN, nrFrames)
                sampleFrames = frames[i:stopIdx].cuda()
                refFrameBatch = refFrame.expand_as(sampleFrames)
                flows = self.OptNet.estimateFlowFull(sampleFrames, refFrameBatch)
                coeffs[i:stopIdx] = self.DCTUtil.getFlowCoeffs(flows)
                photometricWts[i:stopIdx] = torch.mean(torch.abs(refFrameBatch - sampleFrames), dim=(1, 2, 3))
                # acCoeffs[:,:,0,0] = 0.0

                i = stopIdx

        coeffs[:, :, 0, 0] = 0.0
        stabCoeffs = self.weightedMeanApprox(coeffs, photometricWts, refIdx)
        # stabCoeff = self.findPolyApprox(coeffs, refIdx, degree=3)
        stabFrame, grid = self.warp(refFrame[0], stabCoeffs)
        return stabFrame, grid

    def getStabilizedFrames(self):
        self.stabilizeFrames()
        return self.frames
