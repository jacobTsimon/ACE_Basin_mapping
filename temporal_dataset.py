from torchgeo.datasets import RasterDataset, IntersectionDataset
from torchgeo.samplers import RandomGeoSampler
from torch.utils.data import DataLoader
import torch
import rasterio

def check_dataset_info(filepath):
    """Print spatial info for a raster file"""
    with rasterio.open(filepath) as src:
        print(f"\nFile: {filepath}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        print(f"  Shape: {src.shape}")
        print(f"  Transform: {src.transform}")

# Check your files:
#check_dataset_info('/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/val/2025-10/images/Oct4bStackClip_Val_0.tif')

class TemporalDataset:
    """Wrapper for multiple dates with separate masks"""

    def __init__(self, root, dates, split='train'):
        """
        Args:
            root: Base directory
            dates: List of date strings ['2020-01', '2020-02', '2020-03']
            split: 'train' or 'valid'

        Expected structure:
            root/
            ├── train/
            │   ├── 2020-01/
            │   │   ├── images/
            │   │   └── masks/
            │   ├── 2020-02/
            │   │   ├── images/
            │   │   └── masks/
        """
        self.dates = dates
        self.split = split

        # Create image and mask datasets for each date
        self.date_datasets = {}
        combined = None

        for date in dates:
            img_ds = RasterDataset(f'{root}/{split}/{date}/images/')
            mask_ds = RasterDataset(f'{root}/{split}/{date}/masks/')

            # Combine this date's image and mask
            date_combined = img_ds & mask_ds
            self.date_datasets[date] = {'image': img_ds, 'mask': mask_ds}

            # Add to overall intersection
            if combined is None:
                combined = date_combined
            else:
                combined = combined & date_combined

        self.dataset = combined
        self.dates = dates

    def get_dataloader(self, size=256, length=1000, batch_size=4):
        sampler = RandomGeoSampler(self.dataset, size=size, length=length)
        return DataLoader(self.dataset, sampler=sampler, batch_size=batch_size)

    def parse_batch(self, batch):
        """
        Parse the batch into a cleaner format

        Returns:
            dict with keys like 'images', 'masks' containing temporal stacks
        """
        parsed = {
            'images': [],
            'masks': []
        }

        # IntersectionDataset returns keys like: image, mask, image_1, mask_1, image_2, mask_2
        num_dates = len(self.dates)

        for i in range(num_dates):
            if i == 0:
                img_key = 'image'
                mask_key = 'mask'
            else:
                img_key = f'image_{i}'
                mask_key = f'mask_{i}'

            parsed['images'].append(batch[img_key])
            parsed['masks'].append(batch[mask_key])

        # Stack into tensors [B, T, C, H, W] for images, [B, T, H, W] for masks
        parsed['images'] = torch.stack(parsed['images'], dim=1)
        parsed['masks'] = torch.stack(parsed['masks'], dim=1)

        return parsed


# Usage:
temporal_train = TemporalDataset(
    root='/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips',
    dates=['2025-05', '2025-06', '2025-08','2025-09','2025-10'],
    split='train'
)

train_dl = temporal_train.get_dataloader(size=256, batch_size=4)

# In training loop:
for batch in train_dl:
    parsed = temporal_train.parse_batch(batch)

    images = parsed['images']  # Shape: [B, 3, C, H, W] - 3 dates
    masks = parsed['masks']  # Shape: [B, 3, H, W] - 3 masks

    # Now you can process each date separately or stack them
    jan_img, feb_img, mar_img = images[:, 0], images[:, 1], images[:, 2]
    jan_mask, feb_mask, mar_mask = masks[:, 0], masks[:, 1], masks[:, 2]