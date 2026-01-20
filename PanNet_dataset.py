import torch
import torchvision
import torchgeo

import torchvision.transforms as transforms
import numpy as np
from torchgeo.transforms import indices
import matplotlib.pyplot as plt
from PIL import Image

##RATIO OF BACKGROUND TO PAN PIXELS IS 1280:1


##CREATE RANDOM SAMPLER

#create RandomGeoSampler
from torchgeo.datasets import RasterDataset, unbind_samples,stack_samples
from torchgeo.samplers import RandomGeoSampler
from torch.utils.data import DataLoader
from torchgeo.samplers import Units

#identify folder containing data
trainDS = './imgs/scenes/'
trainTRUTH = './imgs/masks/'

#create raster dataset (from custom raster doc: https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html)
class PlanetScope(RasterDataset):


    filename_glob = '20230628_**.tif'  # "2022*_3B_*.tif"
    filename_regex = ".*"  # ^(?P<date>\d{8}_\d{6})_.{2}_.{4}_(3B_*).*"
    #date_format = "%Y%m%d_%H%M%S"
    is_image = True
    separate_files = False
    #setup is specific for planet bands, which are layed in an unorthodox way as follows:
    #B1:coastal blue B2:blue B3:green_i  B4:green    B5:yellow   B6:red  B7:rededge  B8:NIR
    all_bands = ["1", "2", "3", "4", "5","6","7","8"]
    rgb_bands = ["6", "4", "2"]



    def plot(self, sample):
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image

        image = sample["image"][0]
        mask = sample["mask"][0]
        print(image.shape)
        print(mask.shape)

        image = image[rgb_indices].permute(1, 2, 0)
        print(image.shape)
        print(type(image))

        image = torch.clamp(image / 1000 , min=0, max=255).numpy()
        #image = image.numpy().astype(int)


        # Plot the image
        fig, ax = plt.subplots(nrows = 1,ncols=2)
        ax[0].imshow(image)
        ax[1].imshow(mask[0])
        plt.show()
        return fig



#testing functionality:
# train_data = './imgs/train_clips/'
# Tdata = PlanetScope(train_data)
# print(Tdata.crs)
# print(Tdata)


#Elev dataset
class ElevationData(RasterDataset):

    #transforms = transform
    filename_glob = "ElevThresh_*.tif" #"elevationLayer*.tif"
    filename_regex = "^ElevThresh_.*" #"^elevation.*"

    is_image = True
    separate_files = False
    all_bands = ["1"]

#NEED AN NDVI DATASET CLASS
class NDVIData(RasterDataset):

    #transforms = transform
    filename_glob = "NDVI_*.tif" 
    filename_regex = "^NDVI*.*"

    is_image = True
    separate_files = False
    all_bands = ["1"]

#Annotation masks load in here
class PlanetMask(RasterDataset):
    filename_glob = "*.tif" #"pan*F.tif"
    filename_regex = ".*" #"^pan.*"
    date_format = None
    is_image = False
    separate_files = False
    all_bands = ["1"]



##visualization functions
from typing import Iterable, List

def plot_imgs(images: Iterable, axs: Iterable, chnls: List[int] = [2, 1, 0], bright: float = 3.):
    for img, ax in zip(images, axs):
        #arr = torch.clamp(bright * img, min=0, max=1).numpy()
        rgb = img.permute(1, 2, 0).numpy().astype(int) #[:, :, chnls]
        ax.imshow(rgb)
        ax.axis('off')


def plot_msks(masks: Iterable, axs: Iterable):
    for mask, ax in zip(masks, axs):
        ax.imshow(mask.squeeze().numpy())
        ax.axis('off')


def plot_batch(batch: dict, bright: float = 3., cols: int = 3, width: int = 5, chnls: List[int] = [2, 1, 0],nrows = None):
    # Get the samples and the number of items in the batch
    samples = unbind_samples(batch.copy())

    # if batch contains images and masks, the number of images will be doubled
    n = 2 * len(samples) if ('image' in batch) and ('mask' in batch) else len(samples)

    # calculate the number of rows in the grid
    rows = n // cols + (1 if n % cols != 0 else 0)

    if nrows:
        rows = nrows
    # create a grid
    fig, axs = plt.subplots(rows, cols, figsize=(cols * width, rows * width))

    if ('image' in batch) and ('mask' in batch):
        # plot the images on the even axis
        plot_imgs(images=map(lambda x: x['image'], samples), axs=axs.reshape(-1)[::2], chnls=chnls,
                  bright=bright)  # type: ignore

        # plot the masks on the odd axis
        plot_msks(masks=map(lambda x: x['mask'], samples), axs= axs.reshape(-1)[1::2])  # type: ignore

    else:

        if 'image' in batch:
            plot_imgs(images=map(lambda x: x['image'], samples), axs=axs.reshape(-1), chnls=chnls,
                      bright=bright)  # type: ignore

        elif 'mask' in batch:
            plot_msks(masks=map(lambda x: x['mask'], samples), axs=axs.reshape(-1))  # type: ignore
    return fig, axs






