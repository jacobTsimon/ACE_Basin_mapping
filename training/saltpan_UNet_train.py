import warnings

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

from PanNet_dataset import PlanetScope, PlanetMask, ElevationData,RandomTemporalDataset
from saltpan_trainer import train
from UNet_model import U_Net
import optuna
from optuna.trial import TrialState
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.optim import lr_scheduler
from helper_functions import get_extents,getstats,getclassweights
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')

#Define the root dirs for scenes and masks per month

May_train_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/train/2025-05/images/'
May_val_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/val/2025-05/images/'
May_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/train/2025-05/masks/'
May_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/val/2025-05/masks/'


June_train_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/train/2025-06/images/'
June_val_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/val/2025-06/images/'
June_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/train/2025-06/masks/'
June_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/val/2025-06/masks/'


Aug_train_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/train/2025-08/images/'
Aug_val_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/val/2025-08/images/'
Aug_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/train/2025-08/masks/'
Aug_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/val/2025-08/masks/'


Sept_train_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/train/2025-09/images/'
Sept_val_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/val/2025-09/images/'
Sept_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/train/2025-09/masks/'
Sept_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/val/2025-09/masks/'


Oct_train_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/train/2025-10/images/'
Oct_val_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/val/2025-10/images/'
Oct_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/train/2025-10/masks/'
Oct_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/val/2025-10/masks/'

#elev data - MAY NEED TO BE REPLACED
elev_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/elev/'
#elevmask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/elev_mask_0to1point5.tif'


#create geodatasets from each month's mask and images
MayTdata = PlanetScope(May_train_data)
print(MayTdata)
MayVdata = PlanetScope(May_val_data)

May_traintruth = PlanetMask(May_train_mask)
May_valtruth = PlanetMask(May_val_mask)

#June
JuneTdata = PlanetScope(June_train_data)
print(JuneTdata)
JuneVdata = PlanetScope(June_val_data)

June_traintruth = PlanetMask(June_train_mask)
June_valtruth = PlanetMask(June_val_mask)

#Aug
AugTdata = PlanetScope(Aug_train_data)
print(AugTdata)
#AugVdata = PlanetScope(Aug_val_data)

Aug_traintruth = PlanetMask(Aug_train_mask)
#Aug_valtruth = PlanetMask(Aug_val_mask)

#Sept
SeptTdata = PlanetScope(Sept_train_data)
print(SeptTdata.crs)
SeptVdata = PlanetScope(Sept_val_data)

Sept_traintruth = PlanetMask(Sept_train_mask)
Sept_valtruth = PlanetMask(Sept_val_mask)

#Oct
OctTdata = PlanetScope(Oct_train_data)
print(OctTdata.crs)
OctVdata = PlanetScope(Oct_val_data)

Oct_traintruth = PlanetMask(Oct_train_mask)
Oct_valtruth = PlanetMask(Oct_val_mask)

#elev
elev = ElevationData(elev_data)
print(elev.crs)
print(elev)

#join our datasets with two intersections
#each needs elevation due to the randomtemporal wrapper selecting a dataset to train on at
# sampling time

MAYtrain = MayTdata & May_traintruth & elev
JUNEtrain = JuneTdata & June_traintruth & elev
AUGtrain = AugTdata & Aug_traintruth & elev
SEPTtrain = SeptTdata & Sept_traintruth & elev
OCTtrain = OctTdata & Oct_traintruth & elev


trainDS = RandomTemporalDataset([MAYtrain, JUNEtrain, AUGtrain, SEPTtrain, OCTtrain])
print(trainDS)

#validation
MAYval = MayVdata & May_valtruth & elev
JUNEval = JuneVdata & June_valtruth & elev
SEPTval = SeptVdata & Sept_valtruth & elev
OCTval = OctVdata & Oct_valtruth & elev

valDS = RandomTemporalDataset([MAYval, JUNEval, SEPTval, OCTval])

#get useful data for normalizing images
#MAY
MAYT_Img_Stats = getstats(May_train_data,bands = 4,imgcount=MayTdata.__len__())
MAYTMax = MAYT_Img_Stats["AvgMax"]
MAYTMin = MAYT_Img_Stats["AvgMin"]
#June
JuneT_Img_Stats = getstats(June_train_data,bands = 4,imgcount=JuneTdata.__len__())
JuneTMax = JuneT_Img_Stats["AvgMax"]
JuneTMin = JuneT_Img_Stats["AvgMin"]
#Aug
AugT_Img_Stats = getstats(Aug_train_data,bands = 4,imgcount=AugTdata.__len__())
AugTMax = AugT_Img_Stats["AvgMax"]
AugTMin = AugT_Img_Stats["AvgMin"]
#Sept
SeptT_Img_Stats = getstats(Sept_train_data,bands = 4,imgcount=SeptTdata.__len__())
SeptTMax = SeptT_Img_Stats["AvgMax"]
SeptTMin = SeptT_Img_Stats["AvgMin"]
#Oct
OctT_Img_Stats = getstats(Oct_train_data,bands = 4,imgcount=OctTdata.__len__())
OctTMax = OctT_Img_Stats["AvgMax"]
OctTMin = OctT_Img_Stats["AvgMin"]

#now for validation sets
#MAY
MAYV_Img_Stats = getstats(May_val_data,bands = 4,imgcount=MayVdata.__len__())
MAYVMax = MAYV_Img_Stats["AvgMax"]
MAYVMin = MAYV_Img_Stats["AvgMin"]
#June
JuneV_Img_Stats = getstats(June_val_data,bands = 4,imgcount=JuneVdata.__len__())
JuneVMax = JuneV_Img_Stats["AvgMax"]
JuneVMin = JuneV_Img_Stats["AvgMin"]
#NO AUG VAL
#Sept
SeptV_Img_Stats = getstats(Sept_val_data,bands = 4,imgcount=SeptVdata.__len__())
SeptVMax = SeptV_Img_Stats["AvgMax"]
SeptVMin = SeptV_Img_Stats["AvgMin"]
#Oct
OctV_Img_Stats = getstats(Oct_val_data,bands = 4,imgcount=OctVdata.__len__())
OctVMax = OctV_Img_Stats["AvgMax"]
OctVMin = OctV_Img_Stats["AvgMin"]
#calc outb avg
AllTMax = np.array([MAYTMax,AugTMax,SeptTMax,JuneTMax,OctTMax])
TMax = np.mean(AllTMax,axis = 0)
print(TMax)

AllTMin = np.array([MAYTMin,AugTMin,SeptTMin,JuneTMin,OctTMin])
TMin = np.mean(AllTMin,axis = 0)
print(TMin)

AllVMax = np.array([MAYVMax,SeptVMax,JuneVMax,OctVMax])
VMax = np.mean(AllVMax,axis = 0)
print(VMax)

AllVMin = np.array([MAYVMin,SeptVMin,JuneVMin,OctVMin])
VMin = np.mean(AllVMin,axis = 0)
print(VMin)

#add in known elev min/max
TMax = np.append(TMax,1.5)
TMin = np.append(TMin,0)
VMax = np.append(VMax,1.5)
VMin = np.append(VMin,0)
stats = [TMax.astype('float64'),TMin.astype('float64'),VMax.astype('float64'),VMin.astype('float64')]



#instantiate model
UNet = U_Net(in_channels=7,out_channels = 4)
modname = 'UNet'


lr = 0.001
#define some useful params
batch_size = 4
length = 1000
batches = length/batch_size
epochs = 100


#begin training!
train_sampler = RandomBatchGeoSampler(trainDS, size=128, length=length,
                                      batch_size=batch_size)
train_dataloader = DataLoader(trainDS, batch_sampler=train_sampler, collate_fn=stack_samples)  #

# create validation sampler/loader inside of train function for randomization
val_sampler = RandomBatchGeoSampler(valDS, size=128, length=length,
                                    batch_size=batch_size)
val_dataloader = DataLoader(valDS, batch_sampler=val_sampler, collate_fn=stack_samples)

#class weights per dataset
Tclass_weights = getclassweights(train_dataloader,num_class=4)
#Vclass_weights = getclassweights(val_dataloader,num_class=4)


# set train functions
Tloss_fn = nn.CrossEntropyLoss(ignore_index=-1, weight=Tclass_weights.cuda())
Vloss_fn = nn.CrossEntropyLoss(ignore_index=-1)
#loss functions per phase, to avoid weights impacting val loss
loss_fn = {'train': Tloss_fn, 'val': Vloss_fn}

optimizer = torch.optim.SGD(UNet.parameters(),lr = lr)#torch.optim.Adam(UNet.parameters(), lr=0.01) #
scheduler = lr_scheduler.StepLR(optimizer=optimizer,step_size=25,gamma=0.8)

#VIS FOR TROUBLESHOOTING

# for batchID, scene in enumerate(train_dataloader):
#     Tdata.plot(scene)

#BEGIN TRAIN LOOP
train_prec, val_prec, train_rec, val_rec, train_F1, val_F1, UNet, bestscore = train(UNet, train_dataloader,
                                                                         val_dataloader,
                                                                         loss_fn, optimizer,
                                                                         scheduler=scheduler,
                                                                         epochs=epochs, batches=batches,
                                                                         batchsize = batch_size,
                                                                         modname=modname,
                                                                         modID="itF_run15",stats= stats)

print("Training Precision: {}".format(train_prec))
print("Validation Precision: {}".format(val_prec))
print("Training Recall: {}".format(train_rec))
print("Validation Recall: {}".format(val_rec))
print("Train F1 (AVERAGE): {}".format(train_F1))
print("Validation F1 (AVERAGE): {}".format(val_F1))

