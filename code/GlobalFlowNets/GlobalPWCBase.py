from GlobalFlowNets.PWCNet import PWCNet
import torch
import math

class GlobalPWCBase(PWCNet):

    def __init__(self, md=4):
        super(GlobalPWCBase,self).__init__(md=md)


    def forward(self,im1, im2, multiScaleFlows=False):        
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))


        corr6 = self.corr(c16, c26) 
        corr6 = self.leakyRELU(corr6)   


        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)

        flow6 = self.filterFlow(self.predict_flow6(x), level = 6)
        up_flow6 = self.filterFlow(self.deconv6(flow6), level = 6)
        up_feat6 = self.filterFlow(self.upfeat6(x), level = 6)



        warp5 = self.warp(c25, up_flow6*0.625)
        corr5 = self.corr(c15, warp5) 
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)


        flow5 = self.filterFlow(self.predict_flow5(x), level = 5)
        up_flow5 = self.filterFlow(self.deconv5(flow5), level = 5)
        up_feat5 = self.filterFlow(self.upfeat5(x), level = 5)

       
        warp4 = self.warp(c24, up_flow5*1.25)
        corr4 = self.corr(c14, warp4)  
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)


        flow4 = self.filterFlow(self.predict_flow4(x), level = 4)
        up_flow4 = self.filterFlow(self.deconv4(flow4), level = 4)
        up_feat4 = self.filterFlow(self.upfeat4(x), level = 4)


        warp3 = self.warp(c23, up_flow4*2.5)
        corr3 = self.corr(c13, warp3) 
        corr3 = self.leakyRELU(corr3)
        

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)


        flow3 = self.filterFlow(self.predict_flow3(x), level = 3)
        up_flow3 = self.filterFlow(self.deconv3(flow3), level = 3)
        up_feat3 = self.filterFlow(self.upfeat3(x), level = 3)

        warp2 = self.warp(c22, up_flow3*5.0) 
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)

        flow2 = self.filterFlow(flow2, level = 2)
 
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        flow2 = self.filterFlow(flow2, level = 2)
        
        if multiScaleFlows:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2
    
    def estimateFlowFull(self, im1, im2, interPolate=True):
        shape = im1.shape
        newShape= [int(64*math.ceil(im1.shape[2]/64.0)), int(64*math.ceil(im1.shape[3]/64.0))]
        im1 = torch.nn.functional.interpolate(im1,size=newShape, mode = 'bilinear')
        im2 = torch.nn.functional.interpolate(im2,size=newShape, mode = 'bilinear')
        flow = self.estimateFlow(im1,im2, fullScale = interPolate)
        if interPolate is False:
            shape[2]/=4
            shape[3]/=4
            newShape[1]/=4
            newShape[0]/=4

        flow = torch.nn.functional.interpolate(flow, size=shape[2:], mode = 'bilinear')
        flow[0,0] = flow[0,0] * shape[3]/newShape[1]
        flow[0,1] = flow[0,1] * shape[2]/newShape[0]
        return flow
    


    def filterFlow(self, flow, level):
        return flow
