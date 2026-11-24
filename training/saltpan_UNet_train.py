import torch
import torchvision
import numpy as np
import json
from torch import nn
from torchgeo.datasets import RasterDataset, unbind_samples,stack_samples,BoundingBox,IntersectionDataset
from torchgeo.samplers import RandomGeoSampler,RandomBatchGeoSampler,Units
from torch.utils.data import DataLoader
from PanNet_dataset import PlanetScope, PlanetMask, ElevationData, NWIData,SlopeData,CreekProxData,RandomTemporalDataset
from saltpan_trainer import singleclass_train, multiclass_train, collect_object_diagnostics, size_stratified_report
from UNet_model import U_Net, AttU_Net
from torch.optim import lr_scheduler
from helper_functions import get_extents,getstats,getclassweights
from shape_loss import BoundaryLoss, HausdorffDTLoss, ShapeCompoundLoss



print(torch.__version__)
print(torch.version.cuda)          # CUDA version torch was built against
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())

x = torch.randn(8, 3, 32, 32).cuda()   # plain CUDA tensor op - no cuDNN involved
print(x.sum())                          # if this fails too, it's a CUDA driver problem, not cuDNN

conv = torch.nn.Conv2d(3, 8, 3).cuda()  # this specifically exercises cuDNN
print(conv(x).shape)


#Define the root dirs for scenes and masks per month

cleanmask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/mask3mRes32up/'

May_train_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-05/images/'
May_val_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-05/images/'
May_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-05/masks/'
May_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-05/masks/'


June_train_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-06/images/'
June_val_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-06/images/'
June_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-06/masks/'
June_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-06/masks/'

#
Aug_train_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-08/images/'
Aug_val_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-08/images/'
Aug_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-08/masks/'
Aug_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-08/masks/'


Sept_train_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-09/images/'
Sept_val_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-09/images/'
Sept_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-09/masks/'
Sept_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-09/masks/'


Oct_train_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-10/images/'
Oct_val_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-10/images/'
Oct_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-10/masks/'
Oct_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-10/masks/'


#elev data
elev_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/elev/'
#DEM slope data
slope_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/slope/'
#NWI id raster
nwi_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/roi_shpfiles/wetlands_inv/usable/'
#creek proximity raster
creekproxdata = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/creekprox/'

#create geodatasets from each month's mask and images

#BUILD THESE INTO TRY STATEMENTS
MayTdata = PlanetScope(May_train_data)
print(MayTdata)
#no val data in may
MayVdata = PlanetScope(May_val_data)

May_traintruth = PlanetMask(cleanmask)
May_valtruth = PlanetMask(cleanmask)

#June
JuneTdata = PlanetScope(June_train_data)
print(JuneTdata)
JuneVdata = PlanetScope(June_val_data)

June_traintruth = PlanetMask(cleanmask)
June_valtruth = PlanetMask(cleanmask)

#Aug
AugTdata = PlanetScope(Aug_train_data)
# print(AugTdata)
AugVdata = PlanetScope(Aug_val_data)
#
Aug_traintruth = PlanetMask(cleanmask)
Aug_valtruth = PlanetMask(cleanmask)

#Sept
SeptTdata = PlanetScope(Sept_train_data)
print(SeptTdata.crs)
SeptVdata = PlanetScope(Sept_val_data)

Sept_traintruth = PlanetMask(cleanmask)
Sept_valtruth = PlanetMask(cleanmask)

#Oct
OctTdata = PlanetScope(Oct_train_data)
print(OctTdata.crs)
OctVdata = PlanetScope(Oct_val_data)

Oct_traintruth = PlanetMask(cleanmask)
Oct_valtruth = PlanetMask(cleanmask)

#elev
elev = ElevationData(elev_data)
print(elev.crs)

#slope
slope = SlopeData(slope_data)

#NWI
nwi = NWIData(nwi_data)
print(nwi.crs)

#Creek distance
creekprox = CreekProxData(creekproxdata)

#join our datasets with two intersections
#each needs elevation due to the randomtemporal wrapper selecting a dataset to train on at
# sampling time

MAYtrain = (MayTdata & May_traintruth) & (elev & nwi) & (slope & creekprox)

JUNEtrain = (JuneTdata & June_traintruth) & (elev & nwi) & (slope & creekprox)
AUGtrain = (AugTdata & Aug_traintruth) & (elev & nwi) & (slope & creekprox)
SEPTtrain = (SeptTdata & Sept_traintruth) & (elev & nwi) & (slope & creekprox)
OCTtrain = (OctTdata & Oct_traintruth) & (elev & nwi) & (slope & creekprox)




trainDS = RandomTemporalDataset([MAYtrain, JUNEtrain,AUGtrain, SEPTtrain, OCTtrain])
print(trainDS)

#validation
MAYval = (MayVdata & May_valtruth) & (elev & nwi) & (slope & creekprox)
JUNEval = (JuneVdata & June_valtruth) & (elev & nwi) & (slope & creekprox)
AUGval = (AugVdata & Aug_valtruth) & (elev & nwi) & (slope & creekprox)
SEPTval = (SeptVdata & Sept_valtruth) & (elev & nwi) & (slope & creekprox)
OCTval = (OctVdata & Oct_valtruth) & (elev & nwi) & (slope & creekprox)

valDS = RandomTemporalDataset([MAYval,JUNEval,AUGval,SEPTval, OCTval])

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
AllTMax = np.array([MAYTMax,SeptTMax,JuneTMax,AugTMax,OctTMax])
TMax = np.max(AllTMax,axis = 0)
print(TMax)

AllTMin = np.array([MAYTMin,SeptTMin,JuneTMin,AugTMin,OctTMin])
TMin = np.min(AllTMin,axis = 0)
print(TMin)

AllVMax = np.array([MAYVMax,JuneVMax,SeptVMax,OctVMax])
VMax = np.max(AllVMax,axis = 0)
print(VMax)

AllVMin = np.array([MAYVMin,JuneVMin,SeptVMin,OctVMin])
VMin = np.min(AllVMin,axis = 0)
print(VMin)

#get elevation and slope stats
ElevStats = getstats(elev_data,bands = 1,imgcount = 1)
emax = float(ElevStats['Max'].item())
print(emax)
emin = float(ElevStats['Min'].item())
SlopeStats = getstats(slope_data,bands = 1,imgcount = 1)
smax = float(SlopeStats['Max'].item())
smin = float(SlopeStats['Min'].item())
CreekProxStats = getstats(creekproxdata,bands = 1,imgcount = 1)
cpmax = float(CreekProxStats['Max'].item())
cpmin = float(CreekProxStats['Min'].item())

#Add in those values, plus NWI range
TMax,TMin,VMax,VMin = np.append(TMax,[emax,8.0,smax,cpmax],axis=0), np.append(TMin,[emin,0.0,smin,cpmin],axis=0),np.append(VMax,[emax,8.0,smax,cpmax],axis=0), np.append(VMin,[emin,0.0,smin,cpmin],axis=0)


#add in known elev min/max
# TMax = np.append(TMax,1.5)
# TMin = np.append(TMin,0)
# VMax = np.append(VMax,1.5)
# VMin = np.append(VMin,0)
#
# #MANUAL STAT SET
# TMax = torch.tensor([0.1,0.1,0.1,0.25,1.5,8.0])
# TMin = torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0])
# VMax = torch.tensor([0.1,0.1,0.1,0.25,1.5,8.0])
# VMin = torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0])

stats = [TMax,TMin,VMax,VMin]
print(f"Statistics readout per band: \n formatted TrainMax, TrainMin, ValidMax, ValidMin \n {stats}")
# stats = [TMax.astype('float64'),TMin.astype('float64'),VMax.astype('float64'),VMin.astype('float64')]
np.save('./allTMax.npy',TMax)
np.save('./allTMin.npy',TMin)
np.save('./allVMax.npy',VMax)
np.save('./allVMin.npy',VMin)




#PARAMS
modname = 'UNet'
lr = 0.001
batch_size = 16
length = 480
batches = length/batch_size
epochs = 150
patchsize = 256
#early stopping params
patience = 5
tol =0.08


#begin training!
train_sampler = RandomGeoSampler(trainDS, size=patchsize, length=length)
train_dataloader = DataLoader(trainDS, sampler=train_sampler, collate_fn=stack_samples,batch_size=batch_size)  #

# create validation sampler/loader
val_sampler = RandomGeoSampler(valDS, size=patchsize, length=length)
val_dataloader = DataLoader(valDS, sampler=val_sampler, collate_fn=stack_samples,batch_size=batch_size)

#class weights per dataset
# Tclass_weights = torch.ones(1,2,1,1)#run13torch.tensor([0.5,3.0,5.0,5.0])#run12torch.tensor([1.0,3.0,5.0,5.0]) #run11torch.tensor([0.5,2,3,3]) #run9torch.tensor([0.1,1,2,2])#getclassweights(train_dataloader,num_class=4)
beta = 0.9

Tclass_weights = torch.tensor([1.0,4.0]).cuda()
print(Tclass_weights)

#Vclass_weights = getclassweights(val_dataloader,num_class=4)


# set train functions

ce_fn = nn.CrossEntropyLoss(weight=Tclass_weights)
shape_loss = ShapeCompoundLoss(
    ce_loss_fn=ce_fn,
    boundary_loss_fn=BoundaryLoss(d_max=20.0, per_object_interior=False),
    hd_loss_fn=None,                 # experiment 1: boundary only
    w_boundary=1.0, ramp_epochs=10)
loss_fn = {'train': shape_loss, 'val': shape_loss}

#
# Tloss_fn = nn.CrossEntropyLoss(weight=Tclass_weights)
# Vloss_fn = nn.CrossEntropyLoss(weight=Tclass_weights)
#loss functions per phase, to avoid weights impacting val loss
# loss_fn = {'train': Tloss_fn, 'val': Vloss_fn}
#


"This is code I was using for "
# #~~~~~~~~~~~~~~~~~~~TRIAL SIZE REPORT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# from torchgeo.samplers import GridGeoSampler
# from torchvision.transforms import v2
# from torchgeo.transforms import AppendNDWI
# from helper_functions import MinMaxNormalize
# from saltpan_trainer import MSAVI
# from itertools import chain
#
# # Use whichever min/max your validation phase actually used (VMin/VMax,
# # or TMin/TMax if you switched, which I'd recommend)
# addIndicesV = v2.Compose([
#     MinMaxNormalize(min=torch.tensor(VMin).float(), max=torch.tensor(VMax).float()),
#     MSAVI(indexNIR=3, indexRED=2),
#     AppendNDWI(index_nir=3, index_green=1),
# ])
#
# def prepare_val_batch(scene):
#     # mirrors the validation branch of singleclass_train (cname == "SP")
#     mask = scene["mask"].clamp(min=-1)
#     mask[mask > 1] = 0
#     img = scene["image"]
#     zeros = (img[:, 0:3] <= 0).all(dim=1)
#     mask[zeros] = -1
#     img[:, 4][zeros] = -1
#     img[:, 5][zeros] = -1
#     if (mask != -1).sum() == 0:
#         return None
#     nodata = mask == -1
#     x = addIndicesV(img)
#     y = mask.clamp(min=0)
#     return x, y, nodata
#
# def _box_key(b, spatial_only=False):
#     if hasattr(b, "minx"):                      # older torchgeo: BoundingBox
#         key = (b.minx, b.maxx, b.miny, b.maxy)
#         return key if spatial_only else key + (b.mint, b.maxt)
#     parts = [(s.start, s.stop, s.step) if isinstance(s, slice) else s for s in b]
#     return tuple(parts[:2]) if spatial_only else tuple(parts)   # assumes order (x, y, t)
#
# def make_val_loader(ds, size=256, stride=256, batch_size=16):
#     boxes = list(GridGeoSampler(ds, size=size, stride=stride))
#     print("example box:", boxes[0])
#     n_spatial = len({_box_key(b, spatial_only=True) for b in boxes})
#     unique = list({_box_key(b): b for b in boxes}.values())
#     print(f"{len(boxes)} boxes | {n_spatial} unique spatially | {len(unique)} unique incl. time")
#     return DataLoader(ds, batch_size=batch_size, sampler=unique, collate_fn=stack_samples)
#
# model = AttU_Net(img_ch=9, output_ch=2)
# model.load_state_dict(torch.load('./saved_models/UM_L_AttUnet_v4_bestF1_0.51_epoch7.pth', map_location="cuda"))
# model.cuda()
#
# all_results = []
# for ds in [MAYval,JUNEval,AUGval,SEPTval,OCTval]:
#     all_results += collect_object_diagnostics(model, make_val_loader(ds), prepare_val_batch)
#
# size_stratified_report(all_results, iou_thresh=0.5,bins=[0, 16, 52, 125, 500, np.inf])
# size_stratified_report(all_results, iou_thresh=0.3,bins=[0, 16, 52, 125, 500, np.inf])
#
# from object_metrics import (collect_pred_target_tiles, estimate_pred_label_offset,
#                            plot_offset_result)
# val_sets = {"MAY": MAYval,"JUNE": JUNEval, "AUG":AUGval,"SEPT": SEPTval,"OCT": OCTval}   # add ,  for all four
#
# tiles = {}
# for name, ds in val_sets.items():
#     tiles[name] = collect_pred_target_tiles(
#         model, make_val_loader(ds), prepare_val_batch, max_tiles=300, stride=2)
#     print(f"\n=== {name} ===")
#     est=estimate_pred_label_offset(*tiles[name], max_shift=4)
#     plot_offset_result(est)
#
# # pooled over all months
# preds   = [p for t in tiles.values() for p in t[0]]
# targs   = [g for t in tiles.values() for g in t[1]]
# valids  = [v for t in tiles.values() for v in t[2]]
# res_all = estimate_pred_label_offset(preds, targs, valids, max_shift=4)
# plot_offset_result(res_all)
#
# from object_metrics import collect_pan_crops, plot_pan_crops
#
# loader = chain(make_val_loader(MAYval),make_val_loader(JUNEval),make_val_loader(AUGval),make_val_loader(SEPTval), make_val_loader(OCTval))
# crops = collect_pan_crops(loader, prepare_val_batch, size_range=(500,np.inf),
#                           n_crops=24, window=48, model=model)
# plot_pan_crops(crops, bands="rgb", save_path="small_pans_rgb.png") # NIR-R-G false colour

#BEGIN TRAIN LOOP
threshlist = [0.7]#[0.9,0.8,0.7,0.6,0.4,0.3,0.2]
namelist = ["UM_L_creekproxUnet_cleanmask32res3m_v2"]#["thresh0_9run3","thresh0_8run3","thresh0_7run3","thresh0_6run3","thresh0_4run3","thresh0_3run3","thresh0_2run3"]
for ind in range(len(threshlist)):
    # instantiate model
    in_ch = 9
    out_ch = 2
    if modname == "UNet":
        UNet = U_Net(in_channels=in_ch,out_channels = out_ch)
    if modname == "AttUNet":
        UNet = AttU_Net(img_ch=in_ch, output_ch=out_ch)

    optimizer = torch.optim.Adam(UNet.parameters(), lr=lr)  # torch.optim.SGD(UNet.parameters(),lr = lr)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.8)

    print("Training loop {}: threshold {}".format(ind,threshlist[ind]))
    paramdict = {
        "LR_init":lr,
        "batch_size":batch_size,
        "batches":batches,
        "patchsize":patchsize,
        "epochs":epochs,
        "classweights":Tclass_weights.cpu().numpy().tolist(),
        "modtype":modname,
        "patchsize":patchsize,
        "num_channels":in_ch,
        "lossfn":"{}".format(loss_fn['train']),
        "scheduler_params":[scheduler.step_size,scheduler.gamma],
        "early_stopping_params":[patience,tol],
        "notes":" unmasked images | large (256) | AttUnet arch | objLoss ramp set to 10, per interior is OFF,classweights up to 4"
    }

    #log helpful notes
    logname = "/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/param_logs/" + namelist[ind] + "_paramlog.txt"

    with open(logname,"w") as file:
        json.dump(paramdict,file)
    if __name__ == "__main__":
        train_prec, val_prec, train_rec, val_rec, train_F1, val_F1, UNet, bestscore = singleclass_train(UNet, train_dataloader,
                                                                                 val_dataloader,
                                                                                 loss_fn, optimizer,
                                                                                 scheduler=scheduler,
                                                                                 epochs=epochs, batches=batches,
                                                                                 batchsize = batch_size,
                                                                                 thresholds=threshlist[ind],
                                                                                 modID=namelist[ind],stats= stats,cname="SP",patience=patience,tolerance=tol)

        print("Training Precision: {}".format(train_prec))
        print("Validation Precision: {}".format(val_prec))
        print("Training Recall: {}".format(train_rec))
        print("Validation Recall: {}".format(val_rec))
        print("Train F1 (AVERAGE): {}".format(train_F1))
        print("Validation F1 (AVERAGE): {}".format(val_F1))

