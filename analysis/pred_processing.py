import rasterio
import numpy as np
from pathlib import Path
import os

def mask_predictions(
    prediction_path,
    elevation_mask_path,
    udm_path,
    output_path,
    elev_nodata= -999,
    pred_nodata= -999,
    udm_clear_value=1,  # value in UDM that means "usable/clear" - check PlanetLabs docs for your UDM version
):
    # Apply elevation and cloud masks to a prediction raster.
    #
    # Args:
    #     prediction_path: Path to binary prediction raster (0=background, 1=saltpan, 255=nodata)
    #     elevation_mask_path: Path to binary elevation mask (0=valid elevation, 1=mask out)
    #     udm_path: Path to PlanetLabs UDM2 raster
    #     output_path: Path to save masked prediction
    #     elev_nodata: Nodata value in elevation mask
    #     pred_nodata: Nodata value in prediction raster
    #     udm_clear_value: Bit or value in UDM indicating clear/usable pixels

    with rasterio.open(prediction_path) as pred_src:
        pred = pred_src.read(1).astype(np.int16)  # read out raster
        #gather needed metadata
        profile = pred_src.profile.copy()
        pred_transform = pred_src.transform
        pred_crs = pred_src.crs
        pred_shape = pred_src.shape

    # --- Elevation mask ---
        with rasterio.open(elevation_mask_path) as elev_src:

            # Reproject/resample elevation mask to match prediction if needed
            from rasterio.warp import reproject, Resampling
            elev_mask = np.zeros(pred_shape, dtype=np.int16)
            reproject(
                source=rasterio.band(elev_src, 1),
                destination=elev_mask,
                src_transform=elev_src.transform,
                src_crs=elev_src.crs,
                dst_transform=pred_transform,
                dst_crs=pred_crs,
                resampling=Resampling.nearest
            )

    # --- UDM cloud mask ---
    # PlanetLabs UDM2 band meanings (check your specific UDM version):
    # Band 1: clear (1=clear, 0=not clear)
    # Band 2: snow
    # Band 3: shadow
    # Band 4: light haze
    # Band 5: heavy haze
    # Band 6: cloud
    # Band 8: unusable
    # For UDM (not UDM2), band 1 bit 0 = unusable data
            with rasterio.open(udm_path) as udm_src:
                udm = np.zeros(pred_shape, dtype=np.int16)
                reproject(
                    source=rasterio.band(udm_src, 1),  # band 1 = clear map in UDM2
                    destination=udm,
                    src_transform=udm_src.transform,
                    src_crs=udm_src.crs,
                    dst_transform=pred_transform,
                    dst_crs=pred_crs,
                    resampling=Resampling.nearest
                )

                # Build combined valid pixel mask
                # True where pixel should be KEPT
                elev_valid = elev_mask == 1          # within valid elevation range
                cloud_clear = udm == udm_clear_value  # clear sky pixels


                #set pixels as valid IF they have valid elev and udm vals
                valid = elev_valid & cloud_clear

                pred[~valid] = pred_nodata  # set invalid to nodata
                pred[pred_src.read(1) == pred_nodata] = pred_nodata  # preserve original nodata



                # Apply: set invalid pixels to nodata
                # pred[~valid] = pred_nodata
                # # Also preserve any nodata that was already in the prediction
                # pred[pred_src.read(1) == pred_nodata] = pred_nodata

                # Save
                profile.update(dtype='int16', nodata=pred_nodata, compress='lzw')
                #nodata check
                print(f"Saved with nodata: {pred_nodata}")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(pred.astype(np.int16), 1)

                n_valid = valid.sum()
                n_saltpan = ((pred == 1) & valid).sum()
                print(f"  Valid pixels: {n_valid:,}")
                print(f"  Saltpan pixels: {n_saltpan:,} ({n_saltpan/max(n_valid,1)*100:.2f}%)")
                print(f"  Masked out: {(~valid).sum():,}")

    return output_path


def batch_mask_predictions(
    prediction_dir,
    elevation_mask_path,
    udm_dir,           # directory of UDMs matching prediction filenames
    output_dir,
    udm_suffix="_udm2.tif",   # suffix to find matching UDM for each prediction
    pred_suffix="_predicted.tif",
    pred_nodata = 255,
    **kwargs
):
    # Apply masks to all prediction rasters in a directory.
    # Matches predictions to UDMs by filename stem.
    #
    # Filename convention expected: 24a8_2023_03_19_15_4b_stack_predicted.tif
    # Matching UDM expected:        24a8_2023_03_19_15_udm2.tif  (or similar)

    pred_dir = Path(prediction_dir)
    udm_dir = Path(udm_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob(f"*{pred_suffix}"))
    print(f"Found {len(pred_files)} prediction rasters")

    results = {}
    for pred_path in pred_files:
        # Parse scene ID from filename: 24a8_2023_03_19_15_47_30_4b_stack_predicted.tif
        # Extract the scene prefix (everything before _4b_stack)
        stem = pred_path.stem.replace("_predicted", "")
        chop = stem.split("_")  # returns a list
        scene_id = chop[1]+chop[2]+chop[3]+"_"+chop[4]+chop[5]+chop[6]+"_[0-9][0-9]_"+chop[0]+udm_suffix

        # Find matching UDM - FORMATTED AS 20230316_150904_09_24c1_3B_udm2.tif
        udm_candidates = list(udm_dir.glob(scene_id))
        if len(udm_candidates) == 0:
            print(f"WARNING: No UDM found for {pred_path.name} (looked for {scene_id})")
            continue
        udm_path = udm_candidates[0]

        output_path = output_dir / (pred_path.stem + "_masked.tif")

        print(f"\nProcessing: {pred_path.name}")
        print(f"  UDM: {udm_path.name}")

        try:
            mask_predictions(
                prediction_path=str(pred_path),
                elevation_mask_path=elevation_mask_path,
                udm_path=str(udm_path),
                output_path=str(output_path),
                pred_nodata = pred_nodata,
                **kwargs
            )
            results[scene_id] = str(output_path)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nComplete: {len(results)}/{len(pred_files)} processed successfully")
    return results


if __name__ == "__main__":
    PREDICTION_DIR = "/home/hopkinsonlab/Desktop/ACE_Basin_mapping/predicting/predictions/pred23boundsfix/"
    ELEVATION_MASK = "/home/hopkinsonlab/Desktop/ACE_Basin_mapping/analysis/elevmask/elev_mask_0to1point5.tif"
    UDM_DIR = "/home/hopkinsonlab/Desktop/ACE_Basin_mapping/analysis/udm/ortho_udm/"
    OUTPUT_DIR = "/home/hopkinsonlab/Desktop/ACE_Basin_mapping/analysis/pred23boundsfix_masked/"

    results = batch_mask_predictions(
        prediction_dir=PREDICTION_DIR,
        elevation_mask_path=ELEVATION_MASK,
        udm_dir=UDM_DIR,
        output_dir=OUTPUT_DIR,
        pred_suffix="_predicted.tif",
        udm_suffix="_3B_udm2.tif",
        udm_clear_value=1,  # adjust based on your UDM version
        pred_nodata = -999
    )