import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import os
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms import v2
from torchgeo.transforms import AppendNDWI, AppendNDVI
import warnings


# Import your model
import sys
sys.path.append('/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/')
from UNet_model import AttU_Net, U_Net
from helper_functions import MinMaxNormalize


class MSAVI(torch.nn.Module):

    def __init__(self, indexNIR: int, indexRED: int):
        super().__init__()
        self.indexNIR = indexNIR
        self.indexRED = indexRED

    def forward(self, input: torch.Tensor):
        NIR = input[:, self.indexNIR, :, :]
        R = input[:, self.indexRED, :, :]

        NIR2 = (2 * NIR) + 1
        discriminant = NIR2 ** 2 - (8 * (NIR - R))
        discriminant = torch.clamp(discriminant, min=0)

        out = (NIR2 - torch.sqrt(discriminant)) / 2
        out = torch.clamp(out, min=-1, max=1)

        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

        return torch.cat([input, out.unsqueeze(dim=1)], dim=1)


def predict_raster(
        model_path,
        input_raster_path,
        elevation_raster_path,
        slope_raster_path,
        creekprox_raster_path,
        nwi_raster_path,
        output_raster_path,
        device='cuda',
        patch_size=128,
        overlap=32,
        batch_size=16,
        num_classes=2,
        in_ch = 6,
        use_msavi=True,
        use_normalization=False,
        use_elev = True,
        stats=None,
        class_to_predict='SP',
        modtype="UNET",
        nodata_value=255
):
    # Run inference on a full raster image using sliding window approach
    #
    # Args:
    #     model_path: Path to trained model weights (.pth)
    #     input_raster_path: Path to input 4-band raster (R,G,B,NIR)
    #     elevation_raster_path: Path to elevation raster (optional, can be None)
    #     output_raster_path: Path to save output prediction raster
    #     device: 'cuda' or 'cpu'
    #     patch_size: Size of patches to process (should match training)
    #     overlap: Overlap between patches in pixels (helps reduce edge artifacts)
    #     batch_size: Number of patches to process at once
    #     num_classes: Number of output classes
    #     use_msavi: Use MSAVI instead of NDVI
    #     use_normalization: Whether to normalize input
    #     stats: [TMax, TMin] if using normalization
    #     class_to_predict: 'SP' or 'WP' - which class to extract as foreground
    #     nodata_value: Value to use for nodata pixels in output (default 255)
    #
    # Returns:
    #     str: Path to output raster

    print("=" * 80)
    print("RASTER INFERENCE")
    print("=" * 80)
    print(f"Input raster: {input_raster_path}")
    print(f"Elevation raster: {elevation_raster_path}")
    print(f"Output raster: {output_raster_path}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Overlap: {overlap} pixels")
    print(f"Batch size: {batch_size}")
    print("=" * 80 + "\n")

    # Load model
    print("Loading model...")
    print(modtype)
    if modtype == 'ATTUNET':
        model = AttU_Net(img_ch=in_ch, output_ch=num_classes)  # 6 channels: R,G,B,NIR,MSAVI/NDVI,NDWI
    if modtype == 'UNET':
        model = U_Net(in_channels = in_ch,out_channels = num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!\n")

    # Setup transforms
    print("Setting up transforms...")
    if use_normalization and stats is not None:
        TMax, TMin = stats
        if use_msavi:
            transforms = v2.Compose([
                MinMaxNormalize(min=TMin, max=TMax),
                MSAVI(indexNIR=3, indexRED=2),
                AppendNDWI(index_nir=3, index_green=1)
            ])
        else:
            transforms = v2.Compose([
                MinMaxNormalize(min=TMin, max=TMax),
                AppendNDVI(index_nir=3, index_red=2),
                AppendNDWI(index_nir=3, index_green=1)
            ])
    else:
        if use_msavi:
            transforms = v2.Compose([
                MSAVI(indexNIR=3, indexRED=2),
                AppendNDWI(index_nir=3, index_green=1)
            ])
        else:
            transforms = v2.Compose([
                AppendNDVI(index_nir=3, index_red=2),
                AppendNDWI(index_nir=3, index_green=1)
            ])
    print("Transforms configured.\n")

    # Open input raster
    print("Opening input raster...")
    with rasterio.open(input_raster_path) as src:
        img_height = src.height
        img_width = src.width
        img_crs = src.crs
        img_transform = src.transform
        img_profile = src.profile.copy()

        print(f"  Size: {img_width} x {img_height}")
        print(f"  Bands: {src.count}")
        print(f"  CRS: {img_crs}")
        print(f"  Dtype: {src.dtypes[0]}")

    # Open elevation raster if provided
    if elevation_raster_path is not None:
        print("\nOpening elevation raster...")
        elev_src = rasterio.open(elevation_raster_path)
        print(f"  Size: {elev_src.width} x {elev_src.height}")
    else:
        elev_src = None
        print("\nNo elevation raster provided - using zeros")
    #nwi
    if nwi_raster_path is not None:
        print("\nOpening nwi raster...")
        nwi_src = rasterio.open(nwi_raster_path)
        print(f"  Size: {nwi_src.width} x {nwi_src.height}")
    else:
        nwi_src = None
        print("\nNo nwi raster provided - using zeros")
    
    #slope
    if slope_raster_path is not None:
        print("\nOpening slope raster...")
        slope_src = rasterio.open(slope_raster_path)
        print(f"  Size: {slope_src.width} x {slope_src.height}")
    else:
        slope_src = None
        print("\nNo slope raster provided - using zeros")       
    #creekprox
    if creekprox_raster_path is not None:
        print("\nOpening creekprox raster...")
        creekprox_src = rasterio.open(creekprox_raster_path)
        print(f"  Size: {creekprox_src.width} x {creekprox_src.height}")
    else:
        creekprox_src = None
        print("\nNo creekprox raster provided - using zeros")

    # Calculate stride (step size between patches)
    stride = patch_size - overlap

    # Calculate number of patches needed
    n_patches_x = int(np.ceil((img_width - overlap) / stride))
    n_patches_y = int(np.ceil((img_height - overlap) / stride))
    total_patches = n_patches_x * n_patches_y

    print(f"\nProcessing strategy:")
    print(f"  Stride: {stride} pixels")
    print(f"  Patches (X): {n_patches_x}")
    print(f"  Patches (Y): {n_patches_y}")
    print(f"  Total patches: {total_patches}")
    print(f"  Batches: {int(np.ceil(total_patches / batch_size))}")

    # Create output arrays
    print("\nInitializing output arrays...")
    prob_map = np.zeros((img_height, img_width), dtype=np.float32)  # accumulates SP probability
    count_map = np.zeros((img_height, img_width), dtype=np.float32)  # float for proper averaging
    nodata_map = np.zeros((img_height, img_width), dtype=bool)  #
    # Process in batches
    print("\nRunning inference...")
    print("-" * 80)

    with torch.no_grad():
        patch_idx = 0
        batch_patches = []
        batch_positions = []

        with rasterio.open(input_raster_path) as img_src:
            #get the raster nodata value
            src_nodata = img_src.nodata

            for y_idx in range(n_patches_y):
                for x_idx in range(n_patches_x):

                    # Calculate patch boundaries
                    x_start = x_idx * stride
                    y_start = y_idx * stride

                    # Ensure we don't go out of bounds
                    x_end = min(x_start + patch_size, img_width)
                    y_end = min(y_start + patch_size, img_height)

                    # Adjust start if we're at the edge
                    if x_end == img_width and x_end - x_start < patch_size:
                        x_start = max(0, img_width - patch_size)
                    if y_end == img_height and y_end - y_start < patch_size:
                        y_start = max(0, img_height - patch_size)

                    x_end = x_start + patch_size
                    y_end = y_start + patch_size

                    # Read image patch (4 bands: R,G,B,NIR)
                    window = Window(x_start, y_start, patch_size, patch_size)
                    img_patch = img_src.read(window=window)  # Shape: [4, H, W]

                    # Read elevation patch if available
                    if elev_src is not None:
                        elev_patch = elev_src.read(1, window=window)  # Shape: [H, W]
                        elev_patch = np.expand_dims(elev_patch, axis=0)  # Shape: [1, H, W]
                    else:
                        elev_patch = np.zeros((1, patch_size, patch_size), dtype=np.float32)

                    #slope
                    if slope_src is not None:
                        slope_patch = slope_src.read(1, window=window)  # Shape: [H, W]
                        slope_patch = np.expand_dims(slope_patch, axis=0)  # Shape: [1, H, W]
                    else:
                        slope_patch = np.zeros((1, patch_size, patch_size), dtype=np.float32)

                    #creekprox
                    if creekprox_src is not None:
                        creekprox_patch = creekprox_src.read(1, window=window)  # Shape: [H, W]
                        creekprox_patch = np.expand_dims(creekprox_patch, axis=0)  # Shape: [1, H, W]
                    else:
                        creekprox_patch = np.zeros((1, patch_size, patch_size), dtype=np.float32)

                    #nwi
                    if nwi_src is not None:
                        nwi_patch = nwi_src.read(1, window=window)  # Shape: [H, W]
                        nwi_patch = np.expand_dims(nwi_patch, axis=0)  # Shape: [1, H, W]
                    else:
                        nwi_patch = np.zeros((1, patch_size, patch_size), dtype=np.float32)

                    # Check for nodata value in raster
                    if np.isnan(src_nodata):
                        img_patch = torch.nan_to_num(torch.from_numpy(img_patch),nan=nodata_value)  # reassign nans to input nodata val
                        img_patch = img_patch.numpy()


                    if src_nodata is not None and not np.isnan(src_nodata):
                        is_nodata = (img_patch[0] == src_nodata)  # check first band
                    else:
                        # Fall back to all-zero check if no nodata value set
                        is_nodata = (img_patch[0:3] <= 0).all(axis=0)
                        # Skip if mostly nodata
                    if is_nodata.sum() > (patch_size * patch_size * 0.9):
                        patch_idx += 1

                        continue

                    # Combine image and elevation: [5, H, W]
                    combined_patch = np.concatenate([img_patch, elev_patch,slope_patch,creekprox_patch,nwi_patch], axis=0)

                    # Convert to tensor and add to batch
                    patch_tensor = torch.from_numpy(combined_patch).float().unsqueeze(0)  # [1, 5, H, W]

                    batch_patches.append(patch_tensor)
                    batch_positions.append((x_start, y_start, x_end, y_end, is_nodata))

                    # Process batch when full or at end
                    if len(batch_patches) == batch_size or patch_idx == total_patches - 1:

                        # Stack batch
                        batch_tensor = torch.cat(batch_patches, dim=0)  # [B, 5, H, W]

                        # Apply transforms
                        batch_tensor = transforms(batch_tensor)

                        if use_elev != True:

                            # Remove elevation channel (keep R,G,B,NIR,MSAVI/NDVI,NDWI)
                            batch_tensor = torch.cat(
                                (batch_tensor[:, :4, :, :], batch_tensor[:, 5:, :, :]),
                                dim=1
                            )  # [B, 6, H, W]

                        # Move to device
                        batch_tensor = batch_tensor.to(device)

                        # Forward pass
                        outputs = model(batch_tensor)  # [B, num_classes, H, W]

                        # Get predictions
                        probs = outputs.softmax(dim=1)[:, 1, :, :].cpu().numpy()  # SP class probability [B, H, W]

                        # Place predictions in output map
                        for i, (x_start, y_start, x_end, y_end, is_nodata) in enumerate(batch_positions):
                            prob = probs[i]

                            # Accumulate probabilities only for valid pixels
                            valid = ~is_nodata

                            prob_map[y_start:y_end, x_start:x_end] += prob * valid
                            count_map[y_start:y_end, x_start:x_end] += valid.astype(np.float32)
                            nodata_map[y_start:y_end, x_start:x_end] |= is_nodata


                        # Clear batch
                        batch_patches = []
                        batch_positions = []

                        # Update progress
                        progress = (patch_idx + 1) / total_patches * 100
                        print(f"\rProgress: {progress:.1f}% ({patch_idx + 1}/{total_patches} patches)",
                              end='', flush=True)

                    patch_idx += 1

    print("\n" + "-" * 80)
    print("Inference complete!")

    # Remove the old averaging block and replace with:
    print("\nAveraging overlapping predictions and thresholding...")

    # Average probabilities where we have observations
    valid_pixels = count_map > 0
    avg_prob = np.where(valid_pixels, prob_map / count_map, 0.0)

    # Threshold to get binary prediction - use 0.5 or your trained threshold
    THRESHOLD = 0.5  # adjust
    prediction_map = np.full((img_height, img_width), nodata_value, dtype=np.int16)

    # Only write predictions where we had valid (non-nodata) observations
    has_valid_data = valid_pixels & ~nodata_map  # covered by a patch AND not nodata
    prediction_map[has_valid_data] = (avg_prob[has_valid_data] >= THRESHOLD).astype(np.int16)

    # Save the probability map too - useful for downstream analysis
    prob_out = np.where(valid_pixels, (avg_prob * 100).astype(np.int16), nodata_value)

    out_profile = img_profile.copy()
    out_profile.update({'count': 1, 'dtype': 'int16', 'compress': 'lzw', 'nodata': nodata_value})

    os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)

    with rasterio.open(output_raster_path, 'w', **out_profile) as dst:
        dst.write(prediction_map, 1)

    # Optionally save probability map alongside binary prediction
    prob_path = output_raster_path.replace('.tif', '_prob.tif')
    with rasterio.open(prob_path, 'w', **out_profile) as dst:
        dst.write(prob_out, 1)

    # Close elevation raster if opened
    if elev_src is not None:
        elev_src.close()
    if slope_src is not None:
        slope_src.close()
    if creekprox_src is not None:
        creekprox_src.close()
    if nwi_src is not None:
        nwi_src.close()
    
    # Print statistics
    print("\n" + "=" * 80)
    print("OUTPUT STATISTICS")
    print("=" * 80)
    unique, counts = np.unique(prediction_map[prediction_map != nodata_value], return_counts=True)
    total_valid = counts.sum()

    print(f"Total valid pixels: {total_valid:,}")
    for class_id, count in zip(unique, counts):
        print(f"  Class {class_id}: {count:,} pixels ({count / total_valid * 100:.2f}%)")

    nodata_count = (prediction_map == nodata_value).sum()
    print(f"  NoData: {nodata_count:,} pixels")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return output_raster_path


def predict_multiple_rasters(
        model_path,
        input_dir,
        elevation_raster_path,
        slope_raster_path,
        creekprox_raster_path,
        nwi_raster_path,
        output_dir,
        device='cuda',
        in_ch = 7,
        use_elev = True,
        **kwargs
):
    # Run inference on multiple raster files in a directory
    #
    # Args:
    #     model_path: Path to trained model
    #     input_dir: Directory containing input rasters
    #     elevation_raster_path: Path to elevation raster
    #     output_dir: Directory to save outputs
    #     device: 'cuda' or 'cpu'
    #     **kwargs: Additional arguments passed to predict_raster()
    #
    # Returns:
    #     list: Paths to output rasters

    print("=" * 80)
    print("BATCH RASTER INFERENCE")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80 + "\n")

    # Find all .tif files in input directory
    input_path = Path(input_dir)
    raster_files = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")))

    print(f"Found {len(raster_files)} raster files")

    if len(raster_files) == 0:
        print("No raster files found!")
        return []

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    output_paths = []


    for i, raster_file in enumerate(raster_files):
        print(f"\n{'=' * 80}")
        print(f"Processing file {i + 1}/{len(raster_files)}: {raster_file.name}")
        print(f"{'=' * 80}\n")

        # Generate output path
        output_filename = raster_file.stem + "_predicted.tif"
        output_path = os.path.join(output_dir, output_filename)

        # Run prediction
        try:
            predict_raster(
                model_path=model_path,
                input_raster_path=str(raster_file),
                elevation_raster_path=elevation_raster_path,
                slope_raster_path=slope_raster_path,
                creekprox_raster_path=creekprox_raster_path,
                nwi_raster_path=nwi_raster_path,
                output_raster_path=output_path,
                device=device,
                in_ch = in_ch,
                use_elev = use_elev,
                **kwargs
            )
            output_paths.append(output_path)

        except Exception as e:
            print(f"\nERROR processing {raster_file.name}: {e}")
            continue

    print(f"\n{'=' * 80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Successfully processed: {len(output_paths)}/{len(raster_files)} files")
    print(f"{'=' * 80}\n")

    return output_paths


if __name__ == "__main__":



    # Model configuration
    MODEL_PATH = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/saved_models/UM_L_creekproxUnet_cleanmask_v1_bestmIOU_0.69_epoch45.pth'#collated_SPrun21_bestmIOU_0.79_epoch45.pth'#nonoutlier1_bestmIOU_0.73_epoch36.pth'#'/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/saved_models/T0_7_SPrun37_bestmIOU_0.69_epoch44.pth'
    # collated_SPrun21_bestmIOU_0.79_epoch45.pth
    MODTYPE = "UNET"
    NUM_CLASSES = 2
    CLASS_TO_PREDICT = 'SP'  # 'SP' or 'WP'

    # Input/Output paths
    INPUT_RASTER = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStacks/24fe_2025_09_20_16__stack.tif' #emask_09_20_16_2025.tif'  # 4-band raster (R,G,B,NIR)
    ELEVATION_RASTER = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/elev/ACE_elevation.tif'  # Elevation raster (can be None)
    OUTPUT_RASTER = './predictions/092016_output_mask_NEW.tif'
    
    # OR for batch processing:
    INPUT_DIR = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/predicting/23denseStacks/'
    OUTPUT_DIR = './predictions/pred23boundsfixCollated21/'

    # Processing parameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    PATCH_SIZE = 256  # Should match training
    OVERLAP = 64  # Overlap between patches (reduces edge artifacts)
    BATCH_SIZE = 16  # Number of patches to process at once

    # Transform configuration
    USE_MSAVI = True
    USE_NORMALIZATION = True
    USE_ELEVATION = True
    
    #aux files
    # DEM slope data
    slope_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/slope/ACE_elevation_slope.tif'
    # NWI id raster
    nwi_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/roi_shpfiles/wetlands_inv/usable/NWI_functionalRaster.tif'
    # creek proximity raster
    creekproxdata = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/creekprox/creek_prox_fixed.tif'
    
    # Load normalization stats if needed
    #[4.00949414e-02, 5.41604300e-02, 6.20345067e-02, 1.87154680e-01,6.87157345e+00, 8.00000000e+00, 9.16900063e+00, 5.80343994e+03]), array([0., 0., 0., 0., 0., 0., 0., 0.]
    if USE_NORMALIZATION:
        TMax = torch.tensor([4.00949414e-02, 5.41604300e-02, 6.20345067e-02, 1.87154680e-01,6.87157345e+00, 8.00000000e+00, 9.16900063e+00, 5.80343994e+03])#torch.from_numpy(np.load('./allTMax.npy')).float()
        TMin = torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])#torch.from_numpy(np.load('./allTMin.npy')).float()
        STATS = [TMax, TMin]
    else:
        STATS = None

    # ========== RUN INFERENCE ==========

    # Option 1: Single raster
    # print("Running inference on single raster...")
    #
    # output_path = predict_raster(
    #     model_path=MODEL_PATH,
    #     input_raster_path=INPUT_RASTER,
    #     elevation_raster_path=ELEVATION_RASTER,
    #     output_raster_path=OUTPUT_RASTER,
    #     device=DEVICE,
    #     patch_size=PATCH_SIZE,
    #     overlap=OVERLAP,
    #     batch_size=BATCH_SIZE,
    #     num_classes=NUM_CLASSES,
    #     in_ch = 6,
    #     use_msavi=USE_MSAVI,
    #     use_normalization=USE_NORMALIZATION,
    #     stats=STATS,
    #     class_to_predict=CLASS_TO_PREDICT,
    #     modtype=MODTYPE,
    #     nodata_value=255
    # )
    #
    # print(f"\nOutput saved to: {output_path}")


    print("Running inference on multiple rasters...")

    output_paths = predict_multiple_rasters(
        model_path=MODEL_PATH,
        modtype=MODTYPE,
        input_dir=INPUT_DIR,
        elevation_raster_path=ELEVATION_RASTER,
        slope_raster_path=slope_data,
        creekprox_raster_path=creekproxdata,
        nwi_raster_path=nwi_data,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        in_ch = 9,
        use_msavi=USE_MSAVI,
        use_normalization=USE_NORMALIZATION,
        use_elev=USE_ELEVATION,
        stats=STATS,
        class_to_predict=CLASS_TO_PREDICT,
        nodata_value=-999
    )

    print(f"\nProcessed {len(output_paths)} rasters")

