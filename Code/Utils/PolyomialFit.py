import torch


class PolyFit:
    def __init__(self) -> None:
        self.key = None
        self.A = None
        self.AInv = None
        self.t = None

    def getKey(self, degree, lenS, zeroPt):
        key = '{:d}_{:d}_{:d}'.format(degree, lenS, zeroPt)
        return key

    def fit(self, degree, s, zeroPt):
        key = self.getKey(degree, len(s), zeroPt)
        if self.key != key:
            self.A, self.AInv = self.createMatrix(degree, len(s), zeroPt)
            self.key = key

        fit = torch.matmul(self.A, torch.matmul(self.AInv, s))
        return fit

    def createMatrix(self, degree, lenS, zeroPt):
        t = torch.linspace(-zeroPt, lenS - zeroPt - 1, lenS).cuda()
        powers = torch.flip(torch.arange(degree + 1), dims=[0]).cuda()
        A = torch.pow(t[:, None], powers[None, :])
        AInv = torch.linalg.pinv(A)
        return A, AInv
