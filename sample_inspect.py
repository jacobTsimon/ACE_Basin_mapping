import torch
import torchgeo
from torchgeo.samplers import RandomBatchGeoSampler
from PanNet_dataset import PlanetScope,PlanetMask,ElevationData

#load

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


#create geodatasets from
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
MAYtrain = MayTdata & May_traintruth & elev
JUNEtrain = JuneTdata & June_traintruth & elev
AUGtrain = AugTdata & Aug_traintruth & elev
SEPTtrain = SeptTdata & Sept_traintruth & elev
OCTtrain = OctTdata & Oct_traintruth & elev

trainDS = MAYtrain | JUNEtrain | AUGtrain | SEPTtrain | OCTtrain
print(trainDS)

#validation
MAYval = MayVdata & May_valtruth & elev
JUNEval = JuneVdata & June_valtruth & elev
#AUGval = AugVdata & Aug_valtruth & elev
SEPTval = SeptVdata & Sept_valtruth & elev
OCTval = OctVdata & Oct_valtruth & elev
valDS =  MAYval | JUNEval | SEPTval | OCTval

# Create sampler
Tsampler = RandomBatchGeoSampler(trainDS, size=128, length=100, batch_size=4)
Vsampler = RandomBatchGeoSampler(valDS, size=128, length=100, batch_size=4)

# Get one batch of bounding boxes
for phase in ['T','V']:
    if phase == 'T':
        Tsampler_iter = iter(Tsampler)
        bbox_batch = next(Tsampler_iter)
        print("T phase")
    else:
        Vsampler_iter = iter(Vsampler)
        bbox_batch = next(Vsampler_iter)
        print("V phase")
    print(f"Batch contains {len(bbox_batch)} bounding boxes:")
    for i, bbox in enumerate(bbox_batch):
        print(f"\nSample {i}:")
        print(bbox)
        print(f"  Spatial extent: ({bbox[0]}) to ({bbox[1]})")
        print(f"  Temporal extent: {bbox[2]} ")

