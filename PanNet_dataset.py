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
import rasterio

#identify folder containing data
trainDS = './imgs/scenes/'
trainTRUTH = './imgs/masks/'

#create raster dataset (from custom raster doc: https://torchgeo.readthedocs.io/en/stable/tutorials/custom_raster_dataset.html)
class PlanetScope(RasterDataset):

    filename_glob = '**/*.tif'  # "2022*_3B_*.tif" 20230628_*
    filename_regex = "^(?P<date>\d{4}_\d{2}_\d{2}).*.tif$"  # ^(?P<date>\d{8}_\d{6})_.{2}_.{4}_(3B_*).*"
    date_format = "%Y_%m_%d"
    is_image = True
    separate_files = False
    #setup is specific for planet bands, which are layed in an unorthodox way as follows:
    #B1:coastal blue B2:blue B3:green_i  B4:green    B5:yellow   B6:red  B7:rededge  B8:NIR
    #currently only using RGB and NIR
    all_bands = ["1", "2", "3", "4"] #, "5","6","7","8"
    rgb_bands = ["3", "2", "1"]

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
    filename_glob = "**/*.tif" #"elevationLayer*.tif"
    filename_regex = ".*.tif$" #"^elevation.*"

    is_image = True
    separate_files = False
    all_bands = ["1"]

#NEED AN NDVI DATASET CLASS
class NDVIData(RasterDataset):

    #transforms = transform
    filename_glob = "**/NDVI_*.tif"
    filename_regex = "^NDVI*.tif$"

    is_image = True
    separate_files = False
    all_bands = ["1"]

#Annotation masks load in here
class PlanetMask(RasterDataset):
    filename_glob = "**/*.tif" #"pan*F.tif"
    filename_regex = ".*.tif$" #"^pan.*"
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



#trial temporal wrapper
import random
class RandomTemporalDataset:
    """Randomly select one temporal dataset that contains the query (UNION version)"""

    def __init__(self, datasets_list):
        self.datasets = datasets_list

        # Use first dataset's properties
        first_ds = datasets_list[0]
        self._crs = first_ds.crs
        self._res = first_ds.res

        # Calculate UNION of all bounds
        all_bounds = [ds.bounds for ds in datasets_list]

        # X dimension (first slice) - UNION
        minx = min(b[0].start for b in all_bounds)  # Min of all mins (leftmost)
        maxx = max(b[0].stop for b in all_bounds)  # Max of all maxs (rightmost)
        x_step = first_ds.bounds[0].step

        # Y dimension (second slice) - UNION
        miny = min(b[1].start for b in all_bounds)  # Min of all mins (bottommost)
        maxy = max(b[1].stop for b in all_bounds)  # Max of all maxs (topmost)
        y_step = first_ds.bounds[1].step

        # Time dimension (third slice) - UNION
        mint = min(b[2].start for b in all_bounds)  # Min of all mins (earliest)
        maxt = max(b[2].stop for b in all_bounds)  # Max of all maxs (latest)
        t_step = first_ds.bounds[2].step

        # Create bounds as tuple of slices
        self._bounds = (
            slice(minx, maxx, x_step),
            slice(miny, maxy, y_step),
            slice(mint, maxt, t_step)
        )

        # Use first dataset's index
        self.index = first_ds.index

    def __getitem__(self, query):
        # Query format: (slice(minx, maxx), slice(miny, maxy), slice(mint, maxt))
        query_minx = query[0].start
        query_maxx = query[0].stop
        query_miny = query[1].start
        query_maxy = query[1].stop
        query_mint = query[2].start
        query_maxt = query[2].stop

        # Find datasets that contain this query
        valid_datasets = []

        for dataset in self.datasets:
            ds_minx = dataset.bounds[0].start
            ds_maxx = dataset.bounds[0].stop
            ds_miny = dataset.bounds[1].start
            ds_maxy = dataset.bounds[1].stop
            ds_mint = dataset.bounds[2].start
            ds_maxt = dataset.bounds[2].stop

            # Check if query is within dataset bounds (spatial AND temporal)
            spatial_match = (query_minx >= ds_minx and
                             query_maxx <= ds_maxx and
                             query_miny >= ds_miny and
                             query_maxy <= ds_maxy)

            temporal_match = (query_mint >= ds_mint and
                              query_maxt <= ds_maxt)

            if spatial_match and temporal_match:
                valid_datasets.append(dataset)

        if not valid_datasets:
            # Provide helpful error message
            raise IndexError(
                f"Query not found in any dataset.\n"
                f"Query: X=[{query_minx}, {query_maxx}], Y=[{query_miny}, {query_maxy}], T=[{query_mint}, {query_maxt}]\n"
                f"Available datasets:\n" +
                "\n".join([f"  Dataset {i}: X=[{ds.bounds[0].start}, {ds.bounds[0].stop}], "
                           f"Y=[{ds.bounds[1].start}, {ds.bounds[1].stop}], "
                           f"T=[{ds.bounds[2].start}, {ds.bounds[2].stop}]"
                           for i, ds in enumerate(self.datasets)])
            )

        # Randomly pick from valid datasets
        dataset = random.choice(valid_datasets)
        return dataset[query]

    @property
    def crs(self):
        return self._crs

    @property
    def res(self):
        return self._res

    @property
    def bounds(self):
        return self._bounds

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)



