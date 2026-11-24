import rasterio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from rasterio.warp import reproject, Resampling
import pandas as pd
from datetime import datetime
import re
import os

def parse_date_from_filename(filename):

    # Parse date from filename convention: 24a8_2023_03_19_15_4b_stack_predicted_masked.tif
    # Returns datetime object.
    #
    # Match YYYY_MM_DD pattern
    match = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
    if match:
        year, month, day = match.groups()
        return datetime(int(year), int(month), int(day))
    raise ValueError(f"Could not parse date from filename: {filename}")


def build_timeseries_stack(
    masked_pred_dir,
    reference_raster_path=None,
    nodata_value=255,
    mode="binary"  # "binary" or "probability"
):

    # mode="binary"      : loads 0/1/255 prediction rasters
    # mode="probability" : loads 0-100 probability rasters (*_prob.tif),
    #                      converts to 0.0-1.0 float, NaN for nodata
    #
    if mode == "probability":
        # prob rasters sit alongside binary ones with _prob suffix
        file_pattern = "probs/*_prob_masked.tif"
    else:
        file_pattern = "preds/*_predicted_masked.tif"

    pred_dir = Path(masked_pred_dir)
    pred_files = sorted(pred_dir.glob(file_pattern))

    if len(pred_files) == 0:
        raise FileNotFoundError(f"No files found matching {file_pattern} in {masked_pred_dir}")

    print(f"Found {len(pred_files)} prediction rasters (mode={mode})")

    # Parse dates
    dated_files = []
    for f in pred_files:
        try:
            date = parse_date_from_filename(f.name)
            dated_files.append((date, f))
        except ValueError as e:
            print(f"  WARNING: {e}")

    dated_files.sort(key=lambda x: x[0])
    dates = [d for d, _ in dated_files]
    files = [f for _, f in dated_files]

    print(f"Time range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Reference grid
    ref_path = reference_raster_path if reference_raster_path else str(files[0])
    with rasterio.open(ref_path) as ref_src:
        ref_profile = ref_src.profile.copy()
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_shape = (ref_src.height, ref_src.width)

    # Build stack - dtype depends on mode
    if mode == "probability":
        stack = np.full((len(files), *ref_shape), np.nan, dtype=np.float32)
    else:
        stack = np.full((len(files), *ref_shape), nodata_value, dtype=np.uint8)

    for i, fpath in enumerate(files):
        print(f"  Loading [{i+1}/{len(files)}]: {fpath.name} ({dates[i].strftime('%Y-%m-%d')})")
        with rasterio.open(fpath) as src:
            if mode == "probability":
                # Read as float, convert 0-100 scale to 0.0-1.0, nodata=255 becomes NaN
                raw = np.zeros(ref_shape, dtype=np.float32)
                if src.shape == ref_shape and src.transform == ref_transform:
                    raw = src.read(1).astype(np.float32)
                else:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=raw,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear,  # bilinear ok for probabilities
                        src_nodata=nodata_value,
                        dst_nodata=nodata_value
                    )
                # Convert: nodata pixels (255) → NaN, valid pixels → 0.0-1.0
                nodata_pixels = raw >= nodata_value
                stack[i] = np.where(nodata_pixels, np.nan, raw / 100.0)


            else:
                # Binary mode - same as before
                if src.shape == ref_shape and src.transform == ref_transform:
                    stack[i] = src.read(1)
                else:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=stack[i],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.nearest,
                        src_nodata=nodata_value,
                        dst_nodata=nodata_value
                    )

    valid_mask = ~np.isnan(stack).all(axis=0) if mode == "probability" \
                 else (stack != nodata_value).any(axis=0)

    print(f"Stack shape: {stack.shape}")
    print(f"Valid pixels: {valid_mask.sum():,} / {valid_mask.size:,}")

    return stack, dates, ref_profile, valid_mask

def compute_frequency_map(stack, nodata_value=255, mode="binary"):

    #Binary mode:      frequency = fraction of valid obs where class == 1
    #Probability mode: frequency = mean probability across valid obs (NaN-safe)

    if mode == "probability":
        # Stack contains float32 with NaN for nodata
        obs_count = (~np.isnan(stack)).sum(axis=0)           # [H, W]
        freq_map = np.nanmean(stack, axis=0)                  # NaN where all obs are NaN
        freq_map = np.where(obs_count > 0, freq_map, np.nan)
    else:
        valid = stack != nodata_value
        saltpan = (stack == 1) & valid
        obs_count = valid.sum(axis=0)
        saltpan_count = saltpan.sum(axis=0)
        freq_map = np.where(obs_count > 0, saltpan_count / obs_count, np.nan)

    return freq_map, obs_count

def compute_monthly_composites(stack, dates, nodata_value=255, mode="binary"):
    months = {}
    for i, date in enumerate(dates):
        m = date.month
        if m not in months:
            months[m] = []
        months[m].append(stack[i])

    monthly_maps = {}
    for month, frames in months.items():
        arr = np.array(frames)
        if mode == "probability":
            # frames already have NaN for nodata
            monthly_maps[month] = np.nanmean(arr, axis=0)
        else:
            valid = arr != nodata_value
            count = valid.sum(axis=0)
            sp_count = ((arr == 1) & valid).sum(axis=0)
            monthly_maps[month] = np.where(count > 0, sp_count / count, np.nan)

    return monthly_maps

def load_background_image(
    background_raster_path,
    ref_profile,
    ref_shape,
    ref_transform,
    ref_crs,
    bands=(1, 2, 3),        # which bands to use as RGB
    percentile_stretch=2    # clip to 2-98th percentile for contrast
):
    #
    # Load and reproject a GeoTIFF background image to match the prediction grid.
    # Returns a display-ready RGB array normalized to 0-1.
    #
    # Args:
    #     background_raster_path: Path to background GeoTIFF
    #     ref_profile:  rasterio profile of prediction rasters (reference grid)
    #     ref_shape:    (H, W) of prediction grid
    #     ref_transform: affine transform of prediction grid
    #     ref_crs:      CRS of prediction grid
    #     bands:        which bands in the background raster to use as R,G,B
    #     percentile_stretch: clip percentage for contrast enhancement (0 = no stretch)
    #
    rgb = np.zeros((*ref_shape, 3), dtype=np.float32)

    with rasterio.open(background_raster_path) as src:
        for out_idx, band_idx in enumerate(bands):
            band_buf = np.zeros(ref_shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, band_idx),
                destination=band_buf,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan
            )
            rgb[:, :, out_idx] = band_buf

    # Percentile stretch per band for display
    if percentile_stretch > 0:
        for c in range(3):
            band = rgb[:, :, c]
            valid = band[~np.isnan(band)]
            if len(valid) == 0:
                continue
            lo = np.percentile(valid, percentile_stretch)
            hi = np.percentile(valid, 100 - percentile_stretch)
            rgb[:, :, c] = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)

    # Set nodata pixels (any NaN band) to light grey for display
    nodata_pixels = np.isnan(rgb).any(axis=2)
    rgb[nodata_pixels] = 0.85

    return rgb

def plot_frequency_heatmap(
    freq_map,
    obs_count,
    profile,
    output_path,
    title="Saltpan Detection Frequency",
    min_obs=3,
    background_rgb=None     # [H, W, 3] float array from load_background_image
):
    display_map = freq_map.copy()
    display_map[obs_count < min_obs] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, data, cmap, label, ax_title in [
        (axes[0], display_map, 'YlOrRd',
         'Saltpan Detection Frequency', title),
        (axes[1], obs_count.astype(float), 'Blues',
         'Number of Valid Observations', 'Observation Count')
    ]:
        # Draw background if provided
        if background_rgb is not None:
            ax.imshow(background_rgb, interpolation='bilinear', aspect='auto')

        # Overlay heatmap with transparency where nodata
        if data is display_map:
            # Mask nodata so background shows through
            masked = np.ma.masked_invalid(data)
            im = ax.imshow(
                masked,
                cmap=cmap,
                vmin=0, vmax=1,
                alpha=0.75,          # semi-transparent so background visible
                interpolation='none'
            )
        else:
            # Obs count — zero-count areas show as transparent
            masked = np.ma.masked_where(data == 0, data)
            im = ax.imshow(
                masked,
                cmap=cmap,
                alpha=0.75,
                interpolation='none'
            )

        plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
        ax.set_title(ax_title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved frequency heatmap: {output_path}")


def plot_monthly_heatmaps(monthly_maps, output_path, background_rgb=None):
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    months_present = sorted(monthly_maps.keys())
    n = len(months_present)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, month in enumerate(months_present):
        ax = axes[i]

        if background_rgb is not None:
            ax.imshow(background_rgb, interpolation='bilinear', aspect='auto')

        masked = np.ma.masked_invalid(monthly_maps[month])
        im = ax.imshow(
            masked,
            cmap='YlOrRd',
            vmin=0, vmax=1,
            alpha=0.75,
            interpolation='none'
        )
        ax.set_title(month_names[month - 1])
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Saltpan Detection Frequency by Month', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved monthly heatmaps: {output_path}")


def plot_area_timeseries(area_records, output_dir, mode="binary"):

    # Plot two separate area timeseries:
    # 1. Absolute saltpan area (ha)
    # 2. Saltpan area normalized against valid pixel area (fraction)
    #
    # Args:
    #     area_records: list of (date, saltpan_area_ha, valid_area_ha) tuples
    #
    df = pd.DataFrame(
        area_records,
        columns=['date', 'area_ha', 'valid_area_ha']
    )
    df = df.sort_values('date')
    df['normalized'] = df['area_ha'] / df['valid_area_ha'].replace(0, np.nan)

    suffix = f"_{mode}"
    plot_color = '#c0392b'

    def _base_plot(y_col, ylabel, title, output_path):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df['date'], df[y_col], 'o-', color=plot_color,
                linewidth=1.5, markersize=4)
        ax.fill_between(df['date'], df[y_col], alpha=0.15, color=plot_color)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45, ha='right')
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Annotate valid area on normalized plot so reader knows denominator
        if y_col == 'normalized':
            ax2 = ax.twinx()
            ax2.plot(df['date'], df['valid_area_ha'], color='steelblue',
                     linewidth=1.0, linestyle='--', alpha=0.5,
                     label='Valid area (ha)')
            ax2.set_ylabel('Valid Pixel Area (ha)', color='steelblue')
            ax2.tick_params(axis='y', labelcolor='steelblue')
            ax2.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

    prob_label = ' (Probability-Weighted)' if mode == 'probability' else ''

    # Plot 1: absolute area
    _base_plot(
        y_col='area_ha',
        ylabel='Saltpan Area (ha)',
        title=f'Saltpan Area Over Time{prob_label}',
        output_path=os.path.join(output_dir, f'area_absolute{suffix}.png')
    )

    # Plot 2: normalized area with valid pixel area on secondary axis
    _base_plot(
        y_col='normalized',
        ylabel='Saltpan Fraction of Valid Area',
        title=f'Normalized Saltpan Area Over Time{prob_label}',
        output_path=os.path.join(output_dir, f'area_normalized{suffix}.png')
    )

    return df

def save_frequency_raster(freq_map, profile, output_path, nodata_value=255):
    #Save frequency map as a GeoTIFF for use in GIS
    out_profile = profile.copy()
    out_profile.update(dtype='float32', count=1, nodata=-9999, compress='lzw')

    freq_out = freq_map.copy()
    freq_out[np.isnan(freq_out)] = -9999

    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(freq_out.astype(np.float32), 1)
    print(f"Saved frequency raster: {output_path}")

def compute_stats_streaming(
    masked_pred_dir,
    reference_raster_path=None,
    nodata_value=255,
    mode="binary",
    min_obs=3
):
    #
    # Compute frequency map, monthly composites, and area timeseries
    # by streaming through files one at a time. Never holds full stack
    # in memory.
    #
    # Returns:
    #     freq_map:      [H, W] float32
    #     obs_count:     [H, W] int32
    #     monthly_data:  dict {month: {'sum': [H,W], 'count': [H,W]}}
    #     area_records:  list of (datetime, float) tuples
    #     ref_profile:   rasterio profile
    #     dates:         list of datetimes

    if mode == "probability":
        file_pattern = "probs/*_prob_masked.tif"
    else:
        file_pattern = "preds/*_predicted_masked.tif"

    pred_dir = Path(masked_pred_dir)
    pred_files = sorted(pred_dir.glob(file_pattern))

    if len(pred_files) == 0:
        raise FileNotFoundError(
            f"No files found matching {file_pattern} in {masked_pred_dir}"
        )

    print(f"Found {len(pred_files)} rasters (mode={mode})")

    # Parse and sort dates
    dated_files = []
    for f in pred_files:
        try:
            date = parse_date_from_filename(f.name)
            dated_files.append((date, f))
        except ValueError as e:
            print(f"  WARNING: {e}")
    dated_files.sort(key=lambda x: x[0])
    dates = [d for d, _ in dated_files]
    files = [f for _, f in dated_files]

    print(f"Time range: {dates[0].strftime('%Y-%m-%d')} "
          f"to {dates[-1].strftime('%Y-%m-%d')}")

    # Get reference grid
    ref_path = reference_raster_path if reference_raster_path else str(files[0])
    with rasterio.open(ref_path) as ref_src:
        ref_profile = ref_src.profile.copy()
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_shape = (ref_src.height, ref_src.width)

    pixel_area_m2 = abs(ref_profile['transform'].a * ref_profile['transform'].e)
    pixel_area_ha = pixel_area_m2 / 10000

    # Accumulators - only [H,W] arrays, not [T,H,W]
    sum_map   = np.zeros(ref_shape, dtype=np.float64)  # sum of valid values
    obs_count = np.zeros(ref_shape, dtype=np.int32)    # count of valid obs

    # Monthly accumulators: {month_int: {'sum': [H,W], 'count': [H,W]}}
    monthly_data = {}

    area_records = []

    # Reusable buffer for reprojection - allocate once
    buffer = np.zeros(ref_shape, dtype=np.float32)

    for i, (date, fpath) in enumerate(zip(dates, files)):
        print(f"  [{i+1}/{len(files)}] {fpath.name}")

        with rasterio.open(fpath) as src:
            if src.shape == ref_shape and src.transform == ref_transform:
                raw = src.read(1).astype(np.float32)
            else:
                buffer[:] = 0  # clear buffer
                reproject(
                    source=rasterio.band(src, 1),
                    destination=buffer,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear if mode == "probability"
                               else Resampling.nearest,
                    src_nodata=nodata_value,
                    dst_nodata=nodata_value
                )
                raw = buffer.copy()

        # Build valid mask for this frame
        if mode == "probability":
            valid = raw < nodata_value          # nodata stored as 255
            frame_vals = np.where(valid, raw / 100.0, 0.0)
        else:
            valid = raw != nodata_value
            frame_vals = np.where(valid, raw, 0.0)  # 0 or 1

        # Update global accumulators
        sum_map   += frame_vals                 # valid pixels contribute, nodata=0
        obs_count += valid.astype(np.int32)

        # Update monthly accumulators
        m = date.month
        if m not in monthly_data:
            monthly_data[m] = {
                'sum':   np.zeros(ref_shape, dtype=np.float64),
                'count': np.zeros(ref_shape, dtype=np.int32)
            }
        monthly_data[m]['sum']   += frame_vals
        monthly_data[m]['count'] += valid.astype(np.int32)

        # Area for this timestep
        if mode == "probability":
            area = float(np.sum(frame_vals[valid])) * pixel_area_ha
        else:
            area = float((frame_vals == 1).sum()) * pixel_area_ha
        valid_area_ha = float(valid.sum()) * pixel_area_ha
        area_records.append((date, area, valid_area_ha))

        # Explicitly free the frame
        del raw, frame_vals, valid

    # Compute final frequency map
    freq_map = np.where(
        obs_count >= min_obs,
        sum_map / np.maximum(obs_count, 1),
        np.nan
    ).astype(np.float32)

    del sum_map  # free the accumulator

    return freq_map, obs_count, monthly_data, area_records, ref_profile, dates


def monthly_maps_from_data(monthly_data, min_obs=1):

    # Convert raw monthly accumulators to frequency/probability maps

    monthly_maps = {}
    for month, data in monthly_data.items():
        monthly_maps[month] = np.where(
            data['count'] >= min_obs,
            data['sum'] / np.maximum(data['count'], 1),
            np.nan
        ).astype(np.float32)
    return monthly_maps

if __name__ == "__main__":

    MASKED_PRED_DIR = "/home/hopkinsonlab/Desktop/ACE_Basin_mapping/analysis/newmod23predictions_masked/"
    BACKGROUND_RASTER = "/home/hopkinsonlab/Desktop/ACE_Basin_mapping/openstreetmap/openstreetmap.tif"
    OUTPUT_DIR = "./timeseries_outputs_newmod_NONOUTLIERS/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    MIN_OBS = 3

    for mode in ["probability"]:
        print(f"\n{'=' * 60}")
        print(f"Running timeseries analysis: mode={mode}")
        print(f"{'=' * 60}")
        suffix = f"_{mode}"

        freq_map, obs_count, monthly_data, area_records, profile, dates = (
            compute_stats_streaming(
                masked_pred_dir=MASKED_PRED_DIR,
                nodata_value=255,
                mode=mode,
                min_obs=MIN_OBS
            ))

        # Load background once - reused across all plots
        print("\nLoading background image...")
        ref_transform = profile['transform']
        ref_crs = profile['crs']
        ref_shape = (profile['height'], profile['width'])

        background_rgb = load_background_image(
            background_raster_path=BACKGROUND_RASTER,
            ref_profile=profile,
            ref_shape=ref_shape,
            ref_transform=ref_transform,
            ref_crs=ref_crs,
            bands=(1, 2, 3),        # adjust if your background is single-band
            percentile_stretch=2
        )

        # Frequency heatmap
        plot_frequency_heatmap(
            freq_map, obs_count, profile,
            output_path=os.path.join(OUTPUT_DIR, f"frequency_heatmap{suffix}.png"),
            title=f"Saltpan {'Mean Probability' if mode == 'probability' else 'Detection Frequency'}",
            min_obs=MIN_OBS,
            background_rgb=background_rgb
        )

        save_frequency_raster(
            freq_map, profile,
            output_path=os.path.join(OUTPUT_DIR, f"frequency_map{suffix}.tif")
        )

        # Monthly heatmaps
        monthly_maps = monthly_maps_from_data(monthly_data, min_obs=1)
        del monthly_data

        plot_monthly_heatmaps(
            monthly_maps,
            output_path=os.path.join(OUTPUT_DIR, f"monthly_heatmaps{suffix}.png"),
            background_rgb=background_rgb
        )
        del monthly_maps

        # Free background from memory before area plots (not needed there)
        del background_rgb

        # Area timeseries - both absolute and normalized
        df = plot_area_timeseries(
            area_records=area_records,
            output_dir=OUTPUT_DIR,
            mode=mode
        )
        df.to_csv(
            os.path.join(OUTPUT_DIR, f"area_timeseries{suffix}.csv"),
            index=False
        )

    print("\nAll outputs complete.")