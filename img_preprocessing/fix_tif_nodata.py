import rasterio
import numpy as np
from pathlib import Path


def batch_fix_nodata(root_dir, new_nodata=-9999, backup=True):
    """
    Fix all .tif files in a directory tree

    Args:
        root_dir: Root directory to search
        new_nodata: New nodata value to use
        backup: If True, rename originals to .bak
    """

    tif_files = list(Path(root_dir).rglob('*.tif'))
    print(f"Found {len(tif_files)} .tif files")

    fixed_count = 0

    for tif_file in tif_files:
        try:
            with rasterio.open(tif_file) as src:
                old_nodata = src.nodata

                # Skip if already good
                if old_nodata == new_nodata:
                    continue

                # Read data
                data = src.read()

                # Replace old nodata with new
                if old_nodata is not None:
                    mask = (data == old_nodata) | np.isnan(data) | np.isinf(data)
                    data[mask] = new_nodata

                # Update metadata
                meta = src.meta.copy()
                meta.update({
                    'nodata': new_nodata,
                    'dtype': 'float32',
                    'compress':'lzw'
                })

            # Backup original
            if backup:
                backup_file = tif_file.with_suffix('.tif.bak')
                tif_file.rename(backup_file)
                output_file = tif_file
            else:
                output_file = tif_file.with_stem(f"{tif_file.stem}_fixed")

            # Write fixed version
            with rasterio.open(output_file, 'w', **meta) as dst:
                dst.write(data.astype('float32'))

            print(f"✓ Fixed: {tif_file.name} (old nodata: {old_nodata})")
            fixed_count += 1

        except Exception as e:
            print(f"❌ Error with {tif_file}: {e}")

    print(f"\n✓ Fixed {fixed_count} files")


# Run it on your data directory:
batch_fix_nodata('/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/mask3mRes32up/', new_nodata=-999, backup=True)