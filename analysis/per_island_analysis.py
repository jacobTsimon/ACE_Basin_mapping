import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from rasterio.vrt import WarpedVRT
from shapely.geometry import box
import os
import sys
import re
from pathlib import Path
from datetime import datetime


def parse_date_from_filename(filename):

    # Parse date from prediction filename convention:
    # 24df_2023_11_23_16_4b_stack_predicted.tif
    #
    match = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
    if match:
        year, month, day = match.groups()
        return datetime(int(year), int(month), int(day))
    raise ValueError(f"Could not parse date from: {filename}")


def load_shapefile(shapefile_path):

    # Load shapefile and return GeoDataFrame.
    # Assigns sequential index as polygon_id if no meaningful ID exists.
    #
    gdf = gpd.read_file(shapefile_path)
    gdf['polygon_id'] = range(len(gdf))
    print(f"Loaded {len(gdf)} polygons from {shapefile_path}")
    print(f"Shapefile CRS: {gdf.crs}")
    return gdf


def check_overlap(polygon_geom, raster_path, crs):
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        raster_box = box(*raster_bounds)
        src_nodata = src.nodata

    polygon_reprojected = gpd.GeoSeries([polygon_geom], crs=crs)
    try:
        polygon_reprojected = polygon_reprojected.to_crs(raster_crs)
    except Exception as e:
        print(f"    WARNING: CRS reprojection failed: {e}")
        return False, None

    poly_geom_reproj = polygon_reprojected.iloc[0]

    if poly_geom_reproj is None or poly_geom_reproj.is_empty:
        return False, None

    # First check: bounding box
    if not raster_box.intersects(poly_geom_reproj):
        return False, None

    # Second check: does the polygon overlap actual valid data?
    # Read a small overview to check without loading full raster
    if src_nodata is not None:
        print(f"Checking against raster nodata val: {src_nodata}")
        try:
            with rasterio.open(raster_path) as src:
                clipped, _ = rio_mask(
                    src,
                    [poly_geom_reproj],
                    crop=True,
                    nodata=src_nodata,
                    all_touched=False
                )
            has_valid = (clipped[0] != src_nodata).any()
            if not has_valid:
                return False, None
        except Exception:
            pass  # if mask fails, fall through to quantification which will return None

    return True, poly_geom_reproj


def quantify_polygon_in_raster(
    polygon_geom_raster_crs,
    raster_path,
    nodata_value=-999,
    mode="binary"
):

    # Clip a raster to a polygon and compute saltpan statistics.
    #
    # Args:
    #     polygon_geom_raster_crs: shapely geometry already in raster CRS
    #     raster_path: path to prediction raster
    #     nodata_value: nodata value in raster
    #     mode: 'binary' (0/1) or 'probability' (0-100 scaled)
    #
    # Returns:
    #     dict of stats, or None if no valid pixels found
    try:
        with WarpedVRT(rasterio.open(raster_path),dtype='int16',nodata =nodata_value) as src: #warpedvrt lets us alter dtype on the fly
            keywds = src.profile
            keywds['dtype'] = 'int16'
            # Clip raster to polygon
            clipped, clipped_transform = rio_mask(
                src,
                [polygon_geom_raster_crs],
                crop=True,
                nodata=nodata_value,

                all_touched=False   # only pixels whose centre falls inside
            )

            pixel_area_m2 = abs(src.transform.a * src.transform.e)

        data = clipped[0]  # single band [H, W]

        # Valid pixel mask
        valid = data != nodata_value
        n_valid = int(valid.sum())

        if n_valid == 0:
            return None

        pixel_area_ha = pixel_area_m2 / 10000

        if mode == "binary":
            n_saltpan = int((data[valid] == 1).sum())
            n_background = int((data[valid] == 0).sum())
            saltpan_fraction = n_saltpan / n_valid if n_valid > 0 else np.nan
            saltpan_area_ha = n_saltpan * pixel_area_ha
            mean_prob = np.nan

        elif mode == "probability":
            # Data stored as 0-100 integer
            probs = data[valid].astype(np.float32) / 100.0
            mean_prob = float(np.mean(probs))
            # Expected saltpan pixels (sum of probabilities)
            expected_sp_pixels = float(np.sum(probs))
            n_saltpan = np.nan
            n_background = np.nan
            saltpan_fraction = expected_sp_pixels / n_valid
            saltpan_area_ha = expected_sp_pixels * pixel_area_ha

        return {
            'n_valid_pixels':    n_valid,
            'n_saltpan_pixels':  n_saltpan,
            'n_background_pixels': n_background,
            'saltpan_fraction':  saltpan_fraction,
            'saltpan_area_ha':   saltpan_area_ha,
            'valid_area_ha':     n_valid * pixel_area_ha,
            'mean_probability':  mean_prob,
            'pixel_area_m2':     pixel_area_m2
        }

    except Exception as e:
        _,_, exctb = sys.exc_info()
        tb = exctb.tb_lineno
        print(f"    WARNING: error on line {tb}: {e}")
        return None


def quantify_all(
    shapefile_path,
    prediction_dir,
    output_dir,
    shapefile_crs='EPSG:4326',   # CRS of your shapefile - adjust if needed
    file_pattern="*_predicted.tif",
    nodata_value=255,
    mode="binary",
    name = ""
):
    # Main function: iterate over all polygons and all rasters,
    # check overlap, and compute  statistics
    #
    # Saves:
    #     per_observation.csv  - one row per polygon x raster combination
    #     per_polygon.csv      - summary statistics collapsed across all dates
    #
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load polygons
    gdf = gpd.read_file(shapefile_path)
    gdf['polygon_id'] = range(len(gdf))

    # Ensure correct CRS
    if gdf.crs is None:
        print(f"WARNING: Shapefile has no CRS — assuming {shapefile_crs}")
        gdf = gdf.set_crs(shapefile_crs)
    elif str(gdf.crs) != shapefile_crs:
        print(f"Reprojecting shapefile from {gdf.crs} to {shapefile_crs}")
        gdf = gdf.to_crs(shapefile_crs)

    print(f"Loaded {len(gdf)} polygons")

    # Find prediction rasters
    pred_dir = Path(prediction_dir)
    raster_files = sorted(pred_dir.glob(file_pattern))
    print(f"Found {len(raster_files)} prediction rasters")

    if len(raster_files) == 0:
        raise FileNotFoundError(
            f"No rasters found matching {file_pattern} in {prediction_dir}"
        )

    # Parse dates upfront
    dated_rasters = []
    for f in raster_files:
        try:
            date = parse_date_from_filename(f.name)
            dated_rasters.append((date, f))
        except ValueError as e:
            print(f"  WARNING: {e}")
    dated_rasters.sort(key=lambda x: x[0])
    print(f"Date range: {dated_rasters[0][0].strftime('%Y-%m-%d')} "
          f"to {dated_rasters[-1][0].strftime('%Y-%m-%d')}")

    # Main loop
    records = []
    total_combos = len(gdf) * len(dated_rasters)
    processed = 0
    skipped_no_overlap = 0
    skipped_no_valid = 0

    print(f"\nProcessing {len(gdf)} polygons x {len(dated_rasters)} "
          f"rasters = {total_combos} combinations\n")

    for _, poly_row in gdf.iterrows():
        poly_id = poly_row['polygon_id']
        poly_geom = poly_row['geometry']

        print(f"Polygon {poly_id + 1}/{len(gdf)}")

        for date, raster_path in dated_rasters:

            # Check overlap before doing any raster I/O
            overlaps, poly_reproj = check_overlap(poly_geom, raster_path,crs = shapefile_crs)

            if not overlaps:
                skipped_no_overlap += 1
                continue

            # Compute statistics
            stats = quantify_polygon_in_raster(
                polygon_geom_raster_crs=poly_reproj,
                raster_path=str(raster_path),
                nodata_value=nodata_value,
                mode=mode
            )

            processed += 1

            if stats is None:
                skipped_no_valid += 1
                continue

            records.append({
                'polygon_id':           poly_id,
                'date':                 date.strftime('%Y-%m-%d'),
                'year':                 date.year,
                'month':                date.month,
                'raster_file':          raster_path.name,
                **stats
            })

    print(f"\nDone.")
    print(f"  Successful observations:  {len(records)}")
    print(f"  Skipped (no overlap):     {skipped_no_overlap}")
    print(f"  Skipped (no valid pixels): {skipped_no_valid}")

    if len(records) == 0:
        print("WARNING: No valid observations found. Check CRS and file paths.")
        return None, None

    # --- Per-observation CSV ---
    obs_df = pd.DataFrame(records)
    obs_df = obs_df.sort_values(['polygon_id', 'date'])
    obs_path = output_dir / f'{name}_per_observation.csv'
    obs_df.to_csv(obs_path, index=False)
    print(f"\nSaved per-observation CSV: {obs_path}")

    # --- Per-polygon summary CSV ---
    agg_dict = {
        'date':              'count',          # n_observations
        'n_valid_pixels':    ['mean', 'std'],
        'saltpan_fraction':  ['mean', 'std', 'min', 'max'],
        'saltpan_area_ha':   ['mean', 'std', 'min', 'max'],
        'valid_area_ha':     'mean',
    }
    if mode == "probability":
        agg_dict['mean_probability'] = ['mean', 'std']

    poly_df = obs_df.groupby('polygon_id').agg(agg_dict)

    # Flatten column names
    poly_df.columns = ['_'.join(c).strip('_') for c in poly_df.columns]
    poly_df = poly_df.rename(columns={'date_count': 'n_observations'})
    poly_df = poly_df.reset_index()

    poly_path = output_dir / f'{name}_per_polygon.csv'
    poly_df.to_csv(poly_path, index=False)
    print(f"Saved per-polygon CSV:      {poly_path}")

    # Print a quick summary to console
    print(f"\n{'='*55}")
    print(f"SUMMARY")
    print(f"{'='*55}")
    print(f"Total polygons with any valid data: "
          f"{poly_df['polygon_id'].nunique()}")
    print(f"Mean observations per polygon:      "
          f"{poly_df['n_observations'].mean():.1f}")
    print(f"Mean saltpan fraction (all polys):  "
          f"{obs_df['saltpan_fraction'].mean():.3f}")
    print(f"Mean saltpan area ha (all polys):   "
          f"{obs_df['saltpan_area_ha'].mean():.2f} ha")
    print(f"{'='*55}\n")

    return obs_df, poly_df


if __name__ == "__main__":

    SHAPEFILE   = "/home/hopkinsonlab/Desktop/ACE_Basin_mapping/analysis/shapefiles/train_island/island_boundboxes_train.shp"
    PRED_DIR    = "/home/hopkinsonlab/Desktop/ACE_Basin_mapping/analysis/pred23boundsfix_masked/"
    OUTPUT_DIR  = "/home/hopkinsonlab/Desktop/ACE_Basin_mapping/analysis/polygon_quantification/"
    FILENAME = 'ACTUALnonoutliers_train_islands_boundfix'

    # Match your actual file suffix
    FILE_PATTERN = "*_predicted_masked.tif"

    # CRS of your shapefile — check in QGIS if unsure
    SHAPEFILE_CRS = 'EPSG:32617'

    MODE = "binary"   # 'binary' or "probability"

    obs_df, poly_df = quantify_all(
        shapefile_path=SHAPEFILE,
        prediction_dir=PRED_DIR,
        output_dir=OUTPUT_DIR,
        shapefile_crs=SHAPEFILE_CRS,
        file_pattern=FILE_PATTERN,
        nodata_value=-999,
        mode=MODE,
        name=FILENAME
    )