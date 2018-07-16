import torch
import torch.nn as nn
from functools import reduce
from torch.autograd import Variable
from scipy.ndimage import zoom
import cv2
import os
from PIL import Image
import argparse
import numpy as np
from skimage.color import rgb2yuv,yuv2rgb

def parse_args():
    parser = argparse.ArgumentParser(description="Colorize images")
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        required=True,
                        help="input image/input dir")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        required=True,
                        help="output image/output dir")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        required=True,
                        help="location for model (Generator)")
    parser.add_argument("--gpu",
                        type=int,
                        default=-1,
                        help="which GPU to use? [-1 for cpu]")
    args = parser.parse_args()
    return args

class shave_block(nn.Module):
    def __init__(self, s):
        super(shave_block, self).__init__()
        self.s=s
    def forward(self,x):
        return x[:,:,self.s:-self.s,self.s:-self.s]
class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

args = parse_args()

G = nn.Sequential( # Sequential,
    nn.ReflectionPad2d((40, 40, 40, 40)),
    nn.Conv2d(1,32,(9, 9),(1, 1),(4, 4)),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32,64,(3, 3),(2, 2),(1, 1)),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64,128,(3, 3),(2, 2),(1, 1)),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Sequential( # Sequential,
        LambdaMap(lambda x: x, # ConcatTable,
            nn.Sequential( # Sequential,
                nn.Conv2d(128,128,(3, 3)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,(3, 3)),
                nn.BatchNorm2d(128),
            ),
            shave_block(2),
        ),
        LambdaReduce(lambda x,y: x+y), # CAddTable,
    ),
    nn.Sequential( # Sequential,
        LambdaMap(lambda x: x, # ConcatTable,
            nn.Sequential( # Sequential,
                nn.Conv2d(128,128,(3, 3)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,(3, 3)),
                nn.BatchNorm2d(128),
            ),
            shave_block(2),
        ),
        LambdaReduce(lambda x,y: x+y), # CAddTable,
    ),
    nn.Sequential( # Sequential,
        LambdaMap(lambda x: x, # ConcatTable,
            nn.Sequential( # Sequential,
                nn.Conv2d(128,128,(3, 3)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,(3, 3)),
                nn.BatchNorm2d(128),
            ),
            shave_block(2),
        ),
        LambdaReduce(lambda x,y: x+y), # CAddTable,
    ),
    nn.Sequential( # Sequential,
        LambdaMap(lambda x: x, # ConcatTable,
            nn.Sequential( # Sequential,
                nn.Conv2d(128,128,(3, 3)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,(3, 3)),
                nn.BatchNorm2d(128),
            ),
            shave_block(2),
        ),
        LambdaReduce(lambda x,y: x+y), # CAddTable,
    ),
    nn.Sequential( # Sequential,
        LambdaMap(lambda x: x, # ConcatTable,
            nn.Sequential( # Sequential,
                nn.Conv2d(128,128,(3, 3)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,(3, 3)),
                nn.BatchNorm2d(128),
            ),
            shave_block(2),
        ),
        LambdaReduce(lambda x,y: x+y), # CAddTable,
    ),
    nn.ConvTranspose2d(128,64,(3, 3),(2, 2),(1, 1),(1, 1)),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.ConvTranspose2d(64,32,(3, 3),(2, 2),(1, 1),(1, 1)),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32,2,(9, 9),(1, 1),(4, 4)),
    nn.Tanh(),
)
if args.gpu>=0:
    G=G.cuda(args.gpu)
    G.load_state_dict(torch.load(args.model))
else:
    G.load_state_dict(torch.load(args.model,map_location={'cuda:0': 'cpu'}))

def inference(G,in_path,out_path):
    p=Image.open(in_path).convert('RGB')
    img_yuv = rgb2yuv(p)
    H,W,_ = img_yuv.shape
    infimg = np.expand_dims(np.expand_dims(img_yuv[...,0], axis=0), axis=0)
    img_variable = Variable(torch.Tensor(infimg-0.5))
    if args.gpu>=0:
        img_variable=img_variable.cuda(args.gpu)
    res = G(img_variable)
    uv=res.cpu().detach().numpy()
    uv[:,0,:,:] *= 0.436
    uv[:,1,:,:] *= 0.615
    (_,_,H1,W1) = uv.shape
    uv = zoom(uv,(1,1,H/H1,W/W1))
    yuv = np.concatenate([infimg,uv],axis=1)[0]
    rgb=yuv2rgb(yuv.transpose(1,2,0))
    cv2.imwrite(out_path,(rgb.clip(min=0,max=1)*256)[:,:,[2,1,0]])


if not os.path.isdir(args.input):
    inference(G,args.input,args.output)
else:
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    for f in os.listdir(args.input):
        inference(G,os.path.join(args.input,f),os.path.join(args.output,f))