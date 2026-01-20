import torch
from torch import nn
import numpy as np
import torchvision
import datetime
import torchgeo
import random
from torchgeo.datasets import RasterDataset, unbind_samples,stack_samples,BoundingBox
from torchgeo.samplers import RandomBatchGeoSampler,Units
from torch.utils.data import DataLoader

from PanNet_dataset import PlanetScope, PlanetMask, ElevationData
from saltpan_trainer import train
from UNet_model import U_Net
import optuna
from optuna.trial import TrialState
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.optim import lr_scheduler
from helper_functions import get_extents,getstats


#Define the root dirs for scenes and masks

train_data = './imgs/train_clips/'
val_data = './imgs/val_clips/'
train_mask = './imgs/masks/train/'
val_mask = './imgs/masks/val/'
elev_data = './imgs/elev/'

#create geodatasets from
Tdata = PlanetScope(train_data)
print(Tdata.crs)
print(Tdata)
Vdata = PlanetScope(val_data)
elev = ElevationData(elev_data)
print(elev.crs)
print(elev)
traintruth = PlanetMask(train_mask)
valtruth = PlanetMask(val_mask)

#join our datasets with two intersections
trainDS = Tdata & traintruth
print(trainDS.crs)
trainDS = trainDS & elev
print(trainDS.crs)
#validation
valDS = Vdata & valtruth
valDS = valDS & elev

#static time encompassing'23 for now:
# qthe ideal will be a list of bboxes that iterates over the val/train roi.shp files w timestamps
maxt = datetime.datetime(2023,12,31).timestamp()
mint = datetime.datetime(2023,1,1).timestamp()

#define some useful params
batch_size = 5
length = 300
batches = length/batch_size
epochs = 20

#get useful data for normalizing images
print("TEST!!",Tdata.__len__())
T_Img_Stats = getstats(train_data,bands = 4,imgcount=Tdata.__len__())
TMax = T_Img_Stats["AvgMax"]
TMax = np.append(TMax,1.5)
TMin = T_Img_Stats["AvgMin"]
TMin = np.append(TMin,0)
V_Img_Stats = getstats(val_data,bands = 4,imgcount=Vdata.__len__())
VMax = V_Img_Stats["AvgMax"]
VMax = np.append(VMax,1.5)
VMin = V_Img_Stats["AvgMin"]
VMin = np.append(VMin,0)
stats = [TMax.astype('float64'),TMin.astype('float64'),VMax.astype('float64'),VMin.astype('float64')]

#trial very smaLL background weights
class_weights = torch.FloatTensor([1.0e-20,10.0,20.0,50.0]).cuda()

#instantiate model
UNet = U_Net(in_channels=7,out_channels = 4)
ResNet = deeplabv3_resnet50(weights = None,num_classes = 3)
modname = 'UNet'


#use optuna to find best LR
lr = 0.005

# set train functions
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.SGD(UNet.parameters(),lr = lr)#torch.optim.Adam(UNet.parameters(), lr=0.01) #
scheduler = lr_scheduler.StepLR(optimizer=optimizer,step_size=25,gamma=0.8)


#begin training!
train_sampler = RandomBatchGeoSampler(trainDS, size=32, length=length,
                                      batch_size=batch_size)
train_dataloader = DataLoader(trainDS, batch_sampler=train_sampler, collate_fn=stack_samples)  #

# create validation sampler/loader inside of train function for randomization
val_sampler = RandomBatchGeoSampler(valDS, size=32, length=length,
                                    batch_size=batch_size)
val_dataloader = DataLoader(valDS, batch_sampler=val_sampler, collate_fn=stack_samples)

#VIS FOR TROUBLESHOOTING

# for batchID, scene in enumerate(train_dataloader):
#     Tdata.plot(scene)

train_prec, val_prec, train_rec, val_rec, train_F1, val_F1, UNet, bestscore = train(UNet, train_dataloader,
                                                                         val_dataloader,
                                                                         loss_fn, optimizer,
                                                                         scheduler=scheduler,
                                                                         epochs=50, batches=batches,
                                                                         modname=modname,
                                                                         modID="itB_run6",stats= stats)

print("Training Precision: {}".format(train_prec))
print("Validation Precision: {}".format(val_prec))
print("Training Recall: {}".format(train_rec))
print("Validation Recall: {}".format(val_rec))
print("Train F1 (AVERAGE): {}".format(train_F1))
print("Validation F1 (AVERAGE): {}".format(val_F1))

