
import torch


class AffineUility():
    def __init__(self, shape):
        self.shape = shape

    def getUniformGrid(self, centered=False, cuda=True):
        M = self.shape[-1]
        N = self.shape[-2]
        if cuda:
            UY, UX = torch.meshgrid(1.0 * torch.arange(N).cuda(), 1.0 * torch.arange(M).cuda())
        else:
            UY, UX = torch.meshgrid(1.0 * torch.arange(N), 1.0 * torch.arange(M))

        if centered:
            UX = UX - torch.mean(UX)
            UY = UY - torch.mean(UY)
        return UX, UY

    def getFlowCoeffs(self, flows, frames=None):

        coeffs = torch.zeros((flows.shape[0], 4)).cuda()
        UX, UY = self.getUniformGrid(centered=True)
        UTheta = torch.atan(UY / (UX + 1e-6))
        UR = torch.sqrt(UX**2 + UY**2)
        UR[UR < 1] = 1

        for f in range(flows.shape[0]):
            flow = flows[f]
            tx = torch.median(flow[0])
            ty = torch.median(flow[1])

            X = UX + flow[0] - tx
            Y = UY + flow[1] - ty

            # X = X - torch.mean(X)
            # Y = Y - torch.mean(Y)

            R = torch.sqrt(X**2 + Y**2)
            R[R < 1] = 1

            theta = torch.atan(Y / (X + 1e-6))

            theta = torch.median(theta - UTheta)

            scale = torch.median(R / UR)

            coeffs[f, :] = torch.tensor([theta, tx, ty, torch.log(scale)])

        return coeffs

    def getAffineGrids(self, coeffs, cuda=True):
        if len(coeffs.shape) == 1:
            coeffs = coeffs[None]

        theta = coeffs[:, 0, None, None]
        tx = coeffs[:, 1, None, None]
        ty = coeffs[:, 2, None, None]
        s = torch.exp(coeffs[:, 3, None, None])

        scaleX = self.shape[1] / 2.0
        scaleY = self.shape[0] / 2.0

        UX, UY = self.getUniformGrid(centered=True, cuda=cuda)

        X = UX[None] * s * torch.cos(theta) + UY[None] * s * torch.sin(-theta) + tx
        Y = UX[None] * s * torch.sin(theta) + UY[None] * s * torch.cos(theta) + ty

        X = X / scaleX
        Y = Y / scaleY

        grids = torch.cat((X[:, :, :, None], Y[:, :, :, None]), dim=-1)

        return grids
