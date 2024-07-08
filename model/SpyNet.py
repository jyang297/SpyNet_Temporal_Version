#!/usr/bin/env python

import getopt
import math
import numpy
import PIL
import PIL.Image as Image
import numpy as np
import sys
import torch
##########################################################


torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

# args_strModel = 'sintel-final' # 'sintel-final', or 'sintel-clean', or 'chairs-final', or 'chairs-clean', or 'kitti-final'
# args_strOne = '/root/spynetCheck/detach_pytorch_spynet/images/one.png'
# args_strTwo = '/root/spynetCheck/detach_pytorch_spynet/images/two.png'
# args_strOut = '/root/spynetCheck/detach_pytorch_spynet/out.flo'

# for strOption, strArg in getopt.getopt(sys.argv[1:], '', [
#     'model=',
#     'one=',
#     'two=',
#     'out=',
# ])[0]:
#     if strOption == '--model' and strArg != '': args_strModel = strArg # which model to use, see below
#     if strOption == '--one' and strArg != '': args_strOne = strArg # path to the first frame
#     if strOption == '--two' and strArg != '': args_strTwo = strArg # path to the second frame
#     if strOption == '--out' and strArg != '': args_strOut = strArg # path to where the output should be stored
# # end

##########################################################

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0)), tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0)) ], 1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)
# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Preprocess(torch.nn.Module):
            def __init__(self):
                super().__init__()
            # end

            def forward(self, tenInput):
                tenInput = tenInput.flip([1])
                tenInput = tenInput - torch.tensor(data=[0.485, 0.456, 0.406], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)
                tenInput = tenInput * torch.tensor(data=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

                return tenInput
            # end
        # end

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )
            # end

            def forward(self, tenInput):
                return self.netBasic(tenInput)
            # end
        # end

        self.netPreprocess = Preprocess()

        self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

    # end
    def load_pretrained_weights(self, file_path):
        state_dict = torch.load(file_path)
        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in state_dict.items()})
    
    def forward(self, tenOne, tenTwo):
        tenFlow = []

        tenOne = [ self.netPreprocess(tenOne) ]
        tenTwo = [ self.netPreprocess(tenTwo) ]

        for intLevel in range(5):
            if tenOne[0].shape[2] > 32 or tenOne[0].shape[3] > 32:
                tenOne.insert(0, torch.nn.functional.avg_pool2d(input=tenOne[0], kernel_size=2, stride=2, count_include_pad=False))
                tenTwo.insert(0, torch.nn.functional.avg_pool2d(input=tenTwo[0], kernel_size=2, stride=2, count_include_pad=False))
            # end
        # end

        tenFlow = tenOne[0].new_zeros([ tenOne[0].shape[0], 2, int(math.floor(tenOne[0].shape[2] / 2.0)), int(math.floor(tenOne[0].shape[3] / 2.0)) ])
        tenFlow_list = []
        for intLevel in range(len(tenOne)):
            tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if tenUpsampled.shape[2] != tenOne[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
            if tenUpsampled.shape[3] != tenOne[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

            tenFlow = self.netBasic[intLevel](torch.cat([ tenOne[intLevel], backwarp(tenInput=tenTwo[intLevel], tenFlow=tenUpsampled), tenUpsampled ], 1)) + tenUpsampled
            tenFlow_list.append(tenFlow.clone())
        # end
        
        
        

        return tenFlow_list
    # end
# end



##########################################################
def pad_to_multiple_of_32(tenInput):
    intHeight, intWidth = tenInput.shape[2], tenInput.shape[3]
    padHeight = (32 - intHeight % 32) % 32
    padWidth = (32 - intWidth % 32) % 32

    padTop = padHeight // 2
    padBottom = padHeight - padTop
    padLeft = padWidth // 2
    padRight = padWidth - padLeft

    padding = (padLeft, padRight, padTop, padBottom)  # (left, right, top, bottom)
    tenPadded = torch.nn.functional.pad(tenInput, padding, mode='constant', value=0)
    return tenPadded


def estimate(netNetwork, tenOne, tenTwo):


    intWidth = tenOne.shape[3]
    intHeight = tenOne.shape[2]

    # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 416) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenOne = pad_to_multiple_of_32(tenOne.cuda())
    tenTwo = pad_to_multiple_of_32(tenTwo.cuda())
    
    
    tenFlow_list  = netNetwork(tenOne, tenTwo)


    # return tenFlow[0, :, :, :].cpu()
    return tenFlow_list

# end

##########################################################




if __name__ == '__main__':
    tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(Image.open(args_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(Image.open(args_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))


    matrix_1 = matrixs_moving(num_frames=2, initial_position=(10, 0), matrix_size=224, block_size=80).unsqueeze(0)
    matrix_2 = matrixs_moving(num_frames=2, initial_position=(10, 0), matrix_size=224, block_size=80).unsqueeze(0)
    
    tenOne = torch.cat([matrix_1[:,0], matrix_2[:,0]], dim=0)
    tenTwo = torch.cat([matrix_1[:,1], matrix_2[:,1]], dim=0)
    
    
    # Usage
    netNetwork = Network()
    netNetwork.load_pretrained_weights(file_path='./spynet-sintel.pth')
    netNetwork = Network().cuda().eval()
    tenFlow_list = estimate(netNetwork=netNetwork, tenOne=tenOne, tenTwo=tenTwo)

    tenFlow_421 = [tenFlow_list[-3], tenFlow_list[-2], tenFlow_list[-1]]



# end
