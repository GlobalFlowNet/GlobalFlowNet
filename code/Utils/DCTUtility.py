import os
import math
import cv2
import numpy as np
from numpy.lib.utils import source
from scipy.ndimage import gaussian_filter as guassianSmooth
from scipy.ndimage import median_filter as medianFilter
from scipy.signal import savgol_filter, medfilt2d
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import grid_sample, affine_grid
import skvideo.io
from scipy import signal
from GlobalFlowNets.GlobalPWCNets import getGlobalPWCModel
from Utils.VideoUtility import VideoReader, VideoWriter
from math import sqrt, pi

CHUNKLEN = 11

class DCTUtility():
    def __init__(self, shape, cutoffFreq = 5):
        self.shape = shape
        self.cf = cutoffFreq
        self.loadDCTBases()
    
    def getUniformGrid(self):
        M = self.shape[-1]; N = self.shape[-2]
        UY, UX = torch.meshgrid(1.0*torch.arange(N).cuda(), 1.0*torch.arange(M).cuda())
        return UX, UY
    
    def getDCTBase(self, X, Y, u, v):
        M = self.shape[-1]; N = self.shape[-2]
        cu = sqrt(1/2) if u == 0 else 1
        cv = sqrt(1/2) if v == 0 else 1
        u = u*1.0; v = v*1.0
        base = sqrt(2/N) * sqrt(2/M)*cu*cv*torch.cos((pi*u/(2*N))*(2*Y+1))*torch.cos((pi*v/(2*M))*(2*X + 1))
        return base

    def loadDCTBases(self):
        cf = self.cf
        gridX, gridY = self.getUniformGrid()
        M = self.shape[-1]; N = self.shape[-2]
        bases = torch.zeros((N, M, cf, cf), requires_grad=False).cuda()
        for u in range(cf):
            for v in range(cf):
                base = self.getDCTBase(gridX, gridY, u, v)
                bases[:,:, u, v] = base
        
        self.dctBases = bases
    
    def cvtCoeffs2Flow(self, coeffs, X=None,Y=None):
        if (X == None) or (Y == None):
            X, Y = self.getUniformGrid()

        flow = 0
        for u in range(coeffs.shape[1]):
            for v in range(coeffs.shape[2]):
                flow += coeffs[:,u,v][:,None,None]*self.getDCTBase(X,Y,u,v)[None]
        return flow
    
    def getReverseFlowCoeffs(self, coeffs):
        UX, UY = self.getUniformGrid()
        flow = self.cvtCoeffs2Flow(coeffs)
        X = UX + flow[0]
        Y = UY + flow[1]

        revCoeffs = torch.zeros_like(coeffs)
        for u in range(coeffs.shape[1]):
            for v in range(coeffs.shape[2]):
                revCoeffs[:,u,v] = torch.tensordot(self.getDCTBase(X,Y,u,v), -flow, dims=([-2,-1], [-2,-1]))
        
        return revCoeffs

    # def getReverseFlowCoeffs(self, coeffs):
    #     UX, UY = self.getUniformGrid()
    #     flow = self.cvtCoeffs2Flow(coeffs)
    #     X = UX + flow[0]
    #     Y = UY + flow[1]
    #     Xind = torch.clip(torch.round(X),0,X.shape[1]-1).long()
    #     Yind = torch.clip(torch.round(Y),0,Y.shape[0]-1).long()
    #     revFlow = torch.zeros_like(flow)
    #     revFlow[:,Yind,Xind] = flow
    #     revCoeffs = self.getFlowCoeffs(revFlow)
    #     return revCoeffs

    #     revCoeffs = torch.zeros_like(coeffs)
    #     for u in range(coeffs.shape[1]):
    #         for v in range(coeffs.shape[2]):
    #             revCoeffs[:,u,v] = torch.tensordot(self.getDCTBase(X,Y,u,v), -flow, dims=([-2,-1], [-2,-1]))
        
    #     return revCoeffs
    
    def getReverseFlow(self, flow):
        coeffs = self.getFlowCoeffs(flow)
        invCoeffs = self.getReverseFlowCoeffs(coeffs)
        revFlow = self.cvtCoeffs2Flow(invCoeffs)
        return revFlow
    
    # def getReverseFlow(self, flow):
    #     UX, UY = self.getUniformGrid()
    #     X = UX + flow[0]
    #     Y = UY + flow[1]
    #     Xind = torch.clip(torch.round(X),0,X.shape[1]-1).long()
    #     Yind = torch.clip(torch.round(Y),0,Y.shape[0]-1).long()
    #     revFlow = torch.zeros_like(flow)
    #     revFlow[:,Yind,Xind] = -flow
    #     return revFlow

    def cvtFlow2Grid(self, flow):
        #flow = self.cvtCoeffs2Flow(coeffs)
        gridX, gridY = self.getUniformGrid()
        grid = torch.cat((gridX[None], gridY[None]), dim=0)
        grid = grid + flow
        return grid

    def cvtFlowCoeffs2Grid(self, coeffs):
        flow = self.cvtCoeffs2Flow(coeffs)
        grid = self.cvtFlow2Grid(flow)
        return grid

    def getFlowCoeffs(self, flow):
        return torch.tensordot(flow,self.dctBases,([-2,-1],[0,1]))
    
    def getUniformDCTBases(self):
        return self.dctBases
        
    def composeCoeffs(self, coeffs):
        compCoeffs = torch.zeros_like(coeffs)
        UX, UY = self.getUniformGrid()
        gblFlow = self.cvtCoeffs2Flow(coeffs[0], UX , UY)     
        compCoeffs[0] = coeffs[0]
        for f in range(1, coeffs.shape[0]):
            flow = self.cvtCoeffs2Flow(coeffs[f], UX + gblFlow[0], UY + gblFlow[1])            
            gblFlow+=flow
            compCoeffs[f] = self.getFlowCoeffs(gblFlow)

        return compCoeffs   