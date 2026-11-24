
#function for getting the extent coordinates from custom ROI files
#now defunct
from datetime import datetime
maxt = datetime(2023,12,31).timestamp()
mint = datetime(2023,1,1).timestamp()
def get_extents(file):
    # get ROI dataframes
    file = open(file, "r")

    ROIdict = {}
    for line in file:
        print(line)
        num, xmin, xmax, ymin, ymax, n = line.split(sep="\t")  # n is a placeholder for the newline char
        #store extent vals in a dict
        linedict = {"xmin": {"val": float(xmin)}, "xmax": {"val": float(xmax)},
                    "ymin": {"val": float(ymin)}, "ymax": {"val": float(ymax)}
                    }
        #nested dict for return
        ROIdict[str(num)] = linedict

    return ROIdict
#F1 function
def F1_score(precision,recall):
    f1 = 2 * (precision * recall)/(precision + recall)
    return f1
#z1 normalization - defunct
def Z1Norm(input,bands):
    t_n = torch.zeros([5,bands,128,128])
    for b in range(bands):
        imax, imin = input[:,b,:,:].max(), input[:,b,:,:].min()
        nmin, nmax = 0, 1
        # imax,imin = maxs[b],mins[b]
        plus = input[:,b,:,:] #+ abs(imin)

        t_n[:,b,:,:] = (plus - imin) / (imax - imin)  # * (nmax - nmin) + nmin
    #add in new attempt at norm - flattening outliers to 1
    mask = torch.any(t_n.flatten(2, 3) > 1.0, dim=2)
    t_n[mask] = 1.0
    return t_n

#test minmax functions from torchgeo docs: https://torchgeo.readthedocs.io/en/latest/tutorials/transforms.html
import torch
import torch.nn as nn
from typing import Dict, List
class MinMaxNormalize(nn.Module):
    #Normalize channels to the range [0, 1] using min/max values.

    def __init__(self, min: List[float], max: List[float]) -> None:
        super().__init__()
        if min.dim()==1:
            self.min = min.clone().detach()[:, None, None]
        if max.dim()==1:
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
        if torch.isnan(inputs).any():
            print("NANs in MinMaxNormalize")
            inputs = torch.nan_to_num(inputs,nan=0.0)
        return inputs


#function for acquiring information about tif images in the training/validation folders
#usefu for normalization during training
import numpy as np
import os
import rasterio
import re

def getstats(imagefolder,bands,imgcount): #function for getting min / max values into a np array
    out = []
    ind = 0
    for dirs,dir,files in os.walk(imagefolder): #go trhough img folder
        for file in files:
            out.append( [] )
            if re.search('.*xml$',file):
                continue
            print(file)
            with rasterio.open(dirs + file) as tif:
                for i in range(tif.count):

                    arr = tif.read()  # tunr into np array

                    #screen out large nodata values
                    arr[arr > 10000] = 0
                    #and negative values
                    arr[arr < 0] = 0

                    print(arr.shape)
                    b = arr[i] #index into band
                    print("Band # ",i)

                    min =np.nanpercentile(b, 0)
                    max = np.nanpercentile(b,98)

                    print("0th percentile",min)
                    print("98th percentile",max)
                    out[ind].append([max,min])
            ind += 1
    maxout = np.empty((imgcount,bands)) # max value per band per image
    minout = np.empty((imgcount,bands)) # same as above for min
    for i in range(ind): #loop over num images
        for j in range(bands): #loop over bands
            maxout[i, j] = out[i][j][0]
            minout[i,j] = out[i][j][1]


    maxmu = np.empty(bands)
    minmu = np.empty(bands)

    for j in range(bands):  # loop over bands
        maxmu[j] = np.mean(maxout[:,j])
        minmu[j] = np.mean(minout[:,j])

    outdict = {
        "Max":maxout,
        "Min":minout,
        "AvgMax": maxmu,
        "AvgMin": minmu
    }
    return outdict

#function for getting class imbalances
def getclassweights(dataloader,num_class):
    class_counts = torch.zeros(num_class)

    print("Calculating class distribution...")
    for batch in dataloader:
        mask = batch['mask']
        valid_mask = mask[mask >= 0]  # Exclude nodata

        for class_id in range(num_class):
            class_counts[class_id] += (valid_mask == class_id).sum()

    # Inverse frequency
    total = class_counts.sum()
    class_weights = total / (len(class_counts) * class_counts)

    # Normalize so they sum to number of classes
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")

    return class_weights

#color masking for tensorboard
def apply_colormap_to_mask(mask, num_classes=4):
    """
    Convert class indices to RGB colormap for visualization

    Args:
        mask: Tensor of shape [B, H, W] or [B, 1, H, W] with class indices
        num_classes: Number of classes

    Returns:
        RGB tensor [B, 3, H, W]
    """
    # Remove channel dimension if present
    if mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)

    # Define colors for each class (BG, SP, DB, WP)
    colors = torch.tensor([
        #[0,0,0],        #Class -1 (NoData): black
        [50, 50, 50],  # Class 0 (BG): Dark gray
        [255, 100, 100],  # Class 1 (SP): Red
        [100, 255, 100],  # Class 2 (DB): Green
        [100, 100, 255],  # Class 3 (WP): Blue
    ], dtype=torch.float32) / 255.0

    b, h, w = mask.shape
    rgb = torch.zeros((b, 3, h, w))

    for class_id in range(num_classes):
        class_mask = (mask == class_id)
        for c in range(3):
            rgb[:, c][class_mask] = colors[class_id, c]

    return rgb

