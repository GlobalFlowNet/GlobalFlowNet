import torch
import numpy as np
from math import acos, pi, sqrt, cos, sin, tan, asin, atan2, atan, exp, log
import math
import matplotlib.pyplot as plt
from scipy.linalg import circulant
from cvxopt import matrix
from cvxopt import solvers
from scipy.fftpack import dct
from scipy.stats import mode


class StdPathStabilizerQP:
    def __init__(self, path, w, h, minOverlap):
        self.origPath = path
        self.path = path.clone()
        self.sigLen = path.shape[1]
        self.scales = self.getPathScales(self.path, w, h, minOverlap)
        self.T = self.getMaxT(w, h, self.scales, minOverlap)
        agPath = self.fitPathQP()
        std = torch.std(path-agPath, dim=1)
        scaleUp = std[:-1]/self.scales[:-1]
        scaleUp = scaleUp/scaleUp.max()
        self.scales[:-1] = self.scales[:-1]*scaleUp
        self.T = self.getMaxT(w, h, self.scales, minOverlap)


    def getPathScales(self, path, w, h, minOverlap):
        scales = torch.zeros_like(path[:, 0])
        theta = math.atan2(h, w)
        C = 1 - minOverlap
        if h < w:
            scales[0] = asin(sin(theta)/(1-C)) - theta
        else:
            scales[0] = theta - acos(theta/(1-C))

        scales[1] = w*C
        scales[2] = h*C
        scales[3] = abs(log(1-C))*0
        return scales

    def getPathFit(self, r=0.0):
        return self.path + r*self.T*self.scales[:, None]

    def fitPathQPi(self, i):
        pathLB = self.getPathFit(-1).numpy()[i]
        pathUB = self.getPathFit(1).numpy()[i]
        col = np.zeros(self.sigLen+1)
        col[0] = 1
        col[1] = -2
        col[2] = 1
        D = circulant(col)[1:, :-1]
        D[:, 0] = 0.0
        D[:, -1] = 0.0

        P = .5*np.matmul(D, np.transpose(D))
        q = np.zeros(self.sigLen)
        G = np.zeros((self.sigLen*2, self.sigLen))
        h = np.zeros(self.sigLen*2)
        for i in range(self.sigLen):
            G[2*i, i] = -1.0
            G[2*i+1, i] = 1.0
            h[2*i] = -pathLB[i]
            h[2*i+1] = pathUB[i]

        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h)

        return np.squeeze(np.array(sol['x']))

    def fitPathQP(self):
        smoothPath = np.zeros_like(self.path)
        for i in range(4):
            smoothPath[i] = self.fitPathQPi(i)
        return smoothPath

    def getMaxT(self, w, h, scales, minOverlap):
        lB = 0.0
        uB = 1.0

        while abs(uB - lB) > .001:
            b = (uB+lB)/2.0
            iR = checkCrops(w, h, b*scales)
            if iR < minOverlap:
                uB = b
            else:
                lB = b
        return b


def diff(cumsum):
    cumsum_shifted = np.insert(np.delete(cumsum, -1), 0, 0)
    return cumsum - cumsum_shifted


def fft(signal):
    return np.absolute(np.fft.fft(signal)[:len(signal)//2])


def computeStability(cumsum):
    decumulate = False
    fourier = True
    if decumulate:
        signal = diff(cumsum)[1:-1]
    else:
        signal = cumsum

    signal = signal - np.mean(signal)

    if fourier:
        ft = fft(signal)
    else:
        ft = dct(signal, norm='ortho')

    energy = ft**2
    lfNorm = np.sqrt(np.sum(energy[:7]))
    fullNorm = np.sqrt(np.sum(energy)) + 1e-10
    if fullNorm <= 1e-3:
        return 1.0
    return lfNorm/fullNorm


def checkCrops(w, h, eps):
    w = int(w)
    h = int(h)

    R = eps[0] % pi
    S = exp(-abs(eps[3]))
    if R < pi/2:
        a = h/w
    else:
        R = R - pi
        a = -h/w

    ratio = 1.0
    for Tx in [-eps[1], eps[1]]:
        for Ty in [-eps[2], eps[2]]:
            x1 = abs(((Tx*cos(R)) + (Ty*sin(R)) + (S*w/2)) / (cos(R) + a*sin(R)))
            x2 = abs(((Tx*cos(R)) + (Ty*sin(R)) - (S*w/2)) / (cos(R) + a*sin(R)))
            x3 = abs(((Tx*sin(R)) - (Ty*cos(R)) + (S*h/2)) / (sin(R) + a*cos(R)))
            x4 = abs(((Tx*sin(R)) - (Ty*cos(R)) - (S*h/2)) / (sin(R) + a*cos(R)))

            ratio = min(2 * min(x1, x2, x3, x4) / w, ratio)

    return ratio


def smoothPathStdQP(path, w, h, minOverlap):
    AGSmooth = StdPathStabilizerQP(path, w=w, h=h, minOverlap=minOverlap)
    agPath = AGSmooth.fitPathQP()
    return torch.from_numpy(agPath)
