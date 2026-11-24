from rasterio.mask import mask
from shapely.geometry import box


def crop_to_overlap(img_path, mask_path, output_img, output_mask):
    """Crop both rasters to their overlapping region"""

    with rasterio.open(img_path) as img:
        with rasterio.open(mask_path) as msk:
            # Calculate overlap
            overlap_bounds = [
                max(img.bounds.left, msk.bounds.left),
                max(img.bounds.bottom, msk.bounds.bottom),
                min(img.bounds.right, msk.bounds.right),
                min(img.bounds.top, msk.bounds.top)
            ]

            overlap_geom = [box(*overlap_bounds)]

            # Crop image
            img_cropped, img_transform = mask(img, overlap_geom, crop=True)
            img_meta = img.meta.copy()
            img_meta.update({
                'height': img_cropped.shape[1],
                'width': img_cropped.shape[2],
                'transform': img_transform
            })

            with rasterio.open(output_img, 'w', **img_meta) as dst:
                dst.write(img_cropped)

            # Crop mask
            mask_cropped, mask_transform = mask(msk, overlap_geom, crop=True)
            mask_meta = msk.meta.copy()
            mask_meta.update({
                'height': mask_cropped.shape[1],
                'width': mask_cropped.shape[2],
                'transform': mask_transform
            })

            with rasterio.open(output_mask, 'w', **mask_meta) as dst:
                dst.write(mask_cropped)

    print(f"✓ Cropped to overlap: {overlap_bounds}")


crop_to_overlap(
    'data/2020-01/images/image.tif',
    'data/2020-01/masks/mask.tif',
    'data/2020-01/images/image_cropped.tif',
    'data/2020-01/masks/mask_cropped.tif'
)