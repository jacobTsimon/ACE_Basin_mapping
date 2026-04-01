import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from PanNet_dataset import PlanetScope, PlanetMask, ElevationData
from PanNet_dataset import plot_batch
import torchgeo
from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets.utils import stack_samples
import matplotlib.pyplot as plt
from PIL import Image
from helper_functions import get_extents,getstats
from torchgeo.datasets import BoundingBox
from torchgeo.transforms import AppendNDWI,AppendNDVI
from torchvision import transforms
import datetime
import pickle




class U_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3) #check out these numbers and the needed modification
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64 * 2, 32, 3, 1)
        self.upconv1 = self.expand_block(32 * 2, out_channels, 3, 1)

    def __call__(self, x):
        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1)) #concatenate the prior block and horizontal block
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1)) #these are skip connections

        return upconv1

    def contract_block(self,in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return contract

    # now create expand blocks (broaden channels back out)
    def expand_block(self,in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        return expand

#test minmax functions from torchgeo docs: https://torchgeo.readthedocs.io/en/latest/tutorials/transforms.html
import ipywidgets as widgets
import kornia.augmentation as K
import torch
import torch.nn as nn
import torchvision.transforms as T
from IPython.display import display
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.transforms import indices
from typing import Dict, List
class MinMaxNormalize(nn.Module):
    """Normalize channels to the range [0, 1] using min/max values."""

    def __init__(self, min: List[float], max: List[float]) -> None:
        super().__init__()
        self.min = min.clone().detach()[:, None, None]
        self.max = max.clone().detach()[:, None, None]
        self.denominator = (self.max - self.min)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = inputs

        # Batch
        if x.ndim == 4:
          x = (x - self.min[None, ...]) / self.denominator[None, ...]
        # Sample
        else:
          x = (x - self.min) / self.denominator

        inputs = x.clamp(0, 1)
        return inputs

#needs to be reworked - potential cause of illegible imgs
def Z1Norm(input,bands,vis = True):
    t_n = torch.zeros([3,bands,32,32])
    for b in range(bands):
        if vis:
            imax, imin = input[:, b, :, :].max(), input[:, b, :, :].min()
            nmin, nmax = 0, 1
            # imax,imin = maxs[b],mins[b]
            plus = input[:, b, :, :]  # + abs(imin)

            t_n[:, b, :, :] = (plus - imin) / (imax - imin)
    else:
            imax, imin = input[:, b, :, :].max(), input[:, b, :, :].min()
            nmin, nmax = 0, 1
            # imax,imin = maxs[b],mins[b]
            plus = input[:, b, :, :]  # + abs(imin)

            t_n[:, b, :, :] = (plus - imin) / (imax - imin)


    #         t_n[:,b,:,:] = (plus - imin) / (imax - imin)  # * (nmax - nmin) + nmin
    mask = torch.any(t_n.flatten(2,3) > 1.0,dim=2)
    t_n[mask] = 1

    return t_n


modpath = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/saved_models/WM_run1.pth'
UNet = U_Net(in_channels=7,out_channels=2)
#RF = pickle.load(open('./saved_models/bestRFmod3.pickle','rb'))


UNet.load_state_dict(torch.load(modpath))

val_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/val_clips/'
imgPlanet = PlanetScope(val_data)

val_truth = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/WMval_clips/'
truth = PlanetMask(val_truth)
elev = ElevationData('/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/elev')
img = imgPlanet & truth & elev

#set up roi
valROIfile = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/val_extents23.txt' #length = 4
trainROIfile = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/train_extents23.txt'

#extents
valROI = get_extents(valROIfile)
trainROI = get_extents(trainROIfile)
maxt = datetime.datetime(2023,12,31).timestamp()
mint = datetime.datetime(2023,1,1).timestamp()
ind = valROI[str(1)]
vroi = BoundingBox(minx=ind["xmin"]["val"],maxx=ind["xmax"]["val"],miny=ind["ymin"]["val"],
                      maxy=ind["ymax"]["val"],mint= mint,maxt = maxt)

#need to use a gridded sampler instead of random
sampler = GridGeoSampler(img,size = 32,stride=48)
dataloader = DataLoader(img, sampler=sampler,collate_fn=stack_samples)
import numpy as np

pixel_size = imgPlanet.res
crs = imgPlanet.crs.to_epsg()

addIndices = transforms.Compose([
    AppendNDVI(index_nir= 8,index_red=6),
    AppendNDWI(index_nir=8,index_green=4)
])
# stats

V_Img_Stats = getstats(val_data,bands = 4,imgcount=val_data.__len__())
VMax = V_Img_Stats["AvgMax"]
VMax = np.append(VMax,1.5)
VMin = V_Img_Stats["AvgMin"]
VMin = np.append(VMin,0)
stats = [VMax.astype('float64'),VMin.astype('float64')]

VMax = torch.tensor(stats[0]).float()
VMin = torch.tensor(stats[1]).float()



addIndicesV = transforms.Compose([

    MinMaxNormalize(min=VMin, max=VMax),
    AppendNDVI(index_nir=4, index_red=3),
    AppendNDWI(index_nir=4, index_green=2)
])
with torch.no_grad():
    UNet.eval()

    # chips_generator = geochip_generator(dataloader, UNet, crs, pixel_size)
    # file_name = './preds/run2_merged_pred.tif'
    # merge_geochips(chips_generator, file_name)

    for batch in dataloader:
        x = batch["image"]
        #x = addIndices(x)
        x = addIndicesV(x)
        #x = x[:, 0:3, :, :]
        predmap = UNet(x)
        print(predmap.shape)
        print(np.unique(predmap))
        predmap1 = predmap.argmax(dim=1)
        print(predmap1)

        print(predmap1.shape)
        print("Unique pred vals: ",np.unique(predmap1))
        #tot_batch = batch + predmap
        _,axs = plt.subplots(nrows=3,ncols=1)
        #plot_batch(batch,nrows = 3)

        print(np.unique(batch['mask']))
        # batch['mask'][batch['mask'] != 0] = 1
        # x1 = torch.flatten(x[0, :, :, :], start_dim=1, end_dim=2)  # ,start_dim=1,end_dim=2
        # x1 = torch.swapaxes(x1, 0, 1)
        # predmap2_1 = RF.predict(x1)
        # predmap2_1 = predmap2_1.transpose().reshape(( 512, 512))
        # predmap2_1[predmap2_1 != 0] = 1
        #
        # print(predmap2_1.shape)
        # x2 = torch.flatten(x[1, :, :, :], start_dim=1, end_dim=2)  # ,start_dim=1,end_dim=2
        # x2 = torch.swapaxes(x2, 0, 1)
        # predmap2_2 = RF.predict(x2)
        # predmap2_2 = predmap2_2.transpose().reshape(( 512, 512))
        # predmap2_2[predmap2_2 != 0] = 1
        #
        # x3 = torch.flatten(x[2, :, :, :], start_dim=1, end_dim=2)  # ,start_dim=1,end_dim=2
        # x3 = torch.swapaxes(x3, 0, 1)
        # predmap2_3 = RF.predict(x3)
        # predmap2_3 = predmap2_3.transpose().reshape(( 512, 512))
        # predmap2_3[predmap2_3 != 0] = 1


        batchTrans = Z1Norm(batch["image"],bands=5)
        print(np.unique(x))
        print(np.unique(batchTrans))

        b1 = batchTrans[0].permute(1, 2, 0).numpy().astype(float)
        b2 = batchTrans[1].permute(1, 2, 0).numpy().astype(float)
        b3 = batchTrans[2].permute(1, 2, 0).numpy().astype(float)
        t1 = batch["mask"][0].permute(1, 2, 0).numpy().astype(int)
        #t2 = batch["mask"][1].permute(1, 2, 0).numpy().astype(int)
        #t3 = batch["mask"][2].permute(1, 2, 0).numpy().astype(int)

        print(b1.shape)
        # print(np.unique(predmap2_1))
        # print(np.unique(predmap2_2))
        # print(np.unique(predmap2_3))

        axs[0].imshow(b1[:,:,0:3].astype(np.float64),cmap= "Accent")
        #axs[0][1].imshow(b2[:,:,0:3])
        #axs[0][2].imshow(b3[:,:,0:3])
        axs[1].imshow(t1)
        #axs[1][1].imshow(t2)
        #axs[1][2].imshow(t3)
        axs[2].imshow(predmap1[0,:,:],cmap = "Accent")
        #axs[2][1].imshow(predmap1[1],cmap = "Accent")
        #axs[2][2].imshow(predmap1[2],cmap = "Accent")
        # axs[3][0].imshow(predmap2_1,cmap = "Blues")
        # axs[3][1].imshow(predmap2_2,cmap = "Blues")
        # axs[3][2].imshow(predmap2_3,cmap = "Blues")
        # plt.subplots(predmap1[0])
        # plt.subplots(predmap1[1])
        # plt.subplots(predmap1[2])
        plt.show()

