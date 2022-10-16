import torch
import torch.nn.functional as F
from Utils.crop import crop_mask
import numpy as np
import cv2

def fillHoles(mask):
    shape = mask.shape
    maskImg = np.zeros((shape[0], shape[1]), dtype = 'uint8')
    maskImg[mask>0] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closedMask = cv2.morphologyEx(maskImg, cv2.MORPH_CLOSE, kernel)
    return closedMask>0


def getCropFeildDJ(mask):
    shape = mask.shape
    M = shape[1]; N = shape[0]
    mask = mask.data.cpu().numpy()
    mask = fillHoles(mask)
    r,c, H, W = crop_mask(mask > 0)
    Y,X = torch.meshgrid(1.0*torch.linspace(r,r+H-1,N).cuda(), 1.0*torch.linspace(c,c+W-1,M).cuda())
    X = 2.0*(X - M/2.0)/M
    Y = 2.0*(Y - N/2.0)/N
    grid = torch.cat((X[:,:,None], Y[:,:,None]), dim=-1)
    return grid, min(H/N, W/M)




def getCropFeild(mask):
    mask = fillHoles(mask.data.cpu().numpy()>0)
    mask = torch.from_numpy(mask).cuda()*1.0
    shape = mask.shape
    M = shape[1]; N = shape[0]
    UY, UX = torch.meshgrid(torch.arange(N).cuda(), torch.arange(M).cuda())
    UX = 2.0*(UX - M/2.0)/M
    UY = 2.0*(UY - N/2.0)/N
    grid = torch.cat((UX[:,:,None], UY[:,:,None]), dim=-1)[None]
    mask = mask[None,None]

    lowerZoom = .2
    upperZoom = 1

    while abs(lowerZoom - upperZoom) > 1/max(M,N):
        probeZoom = (lowerZoom + upperZoom)/2.0
        probeGrid = grid*probeZoom
        cropRegion = F.grid_sample(mask, probeGrid)
        fitStatus = (M*N -torch.sum(cropRegion>0))<20
        if fitStatus:
            lowerZoom = probeZoom
        else:
            upperZoom = probeZoom
    #print(probeZoom)
    return torch.squeeze(probeGrid), probeZoom


def cropFrames(frames, mask):
    cropFeild, crop = getCropFeildDJ(mask)
    cropFeild = cropFeild.data.cpu()
    print(crop)

    for f in range(frames.shape[0]):
        frames[f] = F.grid_sample(frames[f][None], cropFeild[None])
    
    return frames, cropFeild

