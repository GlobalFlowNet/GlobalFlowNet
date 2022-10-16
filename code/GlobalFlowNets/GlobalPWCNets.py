from GlobalFlowNets.GlobalPWCBase import GlobalPWCBase
import torch
from math import pi, sqrt
import torch.nn as nn
import torch.nn.functional as F


class GlobalPWCDCT(GlobalPWCBase):

    def __init__(self, cfs, warp):
        super(GlobalPWCDCT,self).__init__(md=4)
        self.cfs = cfs
        self.doWarp = warp

    def warp(self, x, flo):
        if self.doWarp:
            return super().warp(x, flo)
        else:
            return x

    def getUniformGrid(self, shape):
        M = shape[-1]; N = shape[-2]
        UY, UX = torch.meshgrid(torch.arange(N).cuda(), torch.arange(M).cuda())
        return UX, UY
    
    def getDCTBase(self, X, Y, u, v):
        shape = X.shape
        M = shape[-1]; N = shape[-2]
        cu = sqrt(1/2) if u == 0 else 1
        cv = sqrt(1/2) if v == 0 else 1
        return sqrt(2/N) * sqrt(2/M)*cu*cv*torch.cos((pi*u/(2*N))*(2*Y+1))*torch.cos((pi*v/(2*M))*(2*X + 1))

    def filterFlow(self, flow, level):
        #with torch.no_grad():
        
        cf = self.cfs[str(level)]

        UX, UY = self.getUniformGrid(flow.shape)
        filFlow = 0.0
        for u in range(cf + 1):
            for v in range(cf + 1):
                base = self.getDCTBase(UX, UY, u, v)
                coeffs = torch.tensordot(base, flow, dims=([-1,-2], [-1,-2]))
                filFlow = filFlow + coeffs[:,:,None, None]*base[None,None]
      
        return filFlow



class GlobalPWCAE(GlobalPWCBase):

    def __init__(self, cfs, warp):
        super(GlobalPWCAE,self).__init__(md=4)
        self.cfs = cfs
        self.doWarp = warp
        self.level6 = nn.Sequential(
            nn.Conv2d(2, 2, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose2d(2, 2, 2, stride=2),
            nn.LeakyReLU(.1))

        self.level5 = nn.Sequential(
            nn.Conv2d(2, 2, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            nn.Conv2d(2, 4, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            
            
            nn.ConvTranspose2d(4, 2, 2, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose2d(2, 2, 2, stride=2),
            nn.LeakyReLU(.1))
        
        
        self.level4 = nn.Sequential(
            nn.Conv2d(2, 2, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            nn.Conv2d(2, 4, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            nn.Conv2d(4, 8, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            
            nn.ConvTranspose2d(8, 4, 2, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose2d(4, 2, 2, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose2d(2, 2, 2, stride=2),
            nn.LeakyReLU(.1))

        self.level3 = nn.Sequential(
            nn.Conv2d(2, 2, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            nn.Conv2d(2, 4, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            nn.Conv2d(4, 8, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            
            nn.ConvTranspose2d(8, 4, 2, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose2d(4, 4, 2, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose2d(4, 2, 2, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose2d(2, 2, 2, stride=2),
            nn.LeakyReLU(.1))
        
        self.level2 = nn.Sequential(
            nn.Conv2d(2, 4, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            nn.Conv2d(4, 8, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(.1),
            
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose2d(8, 8, 2, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose2d(8, 4, 2, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose2d(4, 2, 2, stride=2),
            nn.LeakyReLU(.1))

        

    def warp(self, x, flo):
        if self.doWarp:
            return super().warp(x, flo)
        else:
            return x



    def filterFlow(self, flow, level):        
        if level == 6:
            filFlow = self.level6(flow)
        elif level ==5:
            filFlow = self.level5(flow)
        elif level ==4:
            filFlow = self.level4(flow)
        elif level ==3:
            filFlow = self.level3(flow)
        elif level ==2:
            filFlow = self.level2(flow)
        
        
            

        return filFlow


def getGlobalPWCModel(config, path=None, loadInCPU = False):
    if 'compression' in config.keys():
        
        if config['compression'] == 'DCT':
            model = GlobalPWCDCT(config['cfs'], config['warp'])
        elif config['compression'] == 'AE':
            model = GlobalPWCAE(config['cfs'], config['warp'])
    else:
        model = GlobalPWCDCT(config['cfs'], config['warp'])
        

    if path is not None:
        if loadInCPU is False:
            data = torch.load(path)
        else:
            data = torch.load(path, map_location=lambda storage, loc: storage)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'], strict=False)
        else:
            model.load_state_dict(data, strict=False)
    return model.eval().cuda()