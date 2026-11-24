import glob
import os

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy import ndimage as ndi
from skimage.measure import label as _label, regionprops as _regionprops
import argparse
import rasterio


DEFAULT_BINS = [0, 16, 32, 48, 52, 60, 125, 500, np.inf]

#checking for skimage attributes
def _prop(p, new, old):
    try:
        return getattr(p, new)
    except AttributeError:
        return getattr(p, old)


def inventory_from_array(mask, sp_value=1, pixel_size_m=None, nodata_mask=None,
                         connectivity=2, meta=None):

    # mask:         2D array of class values; pixels == sp_value are salt pan
    # pixel_size_m: ground size of one pixel in meters (None -> area_m2 is NaN)
    # nodata_mask:  optional 2D bool, True where the mask is invalid
    # Returns one row per connected pan.

    sp = mask == sp_value # array of sp values
    if nodata_mask is not None: #mask out nodatas
        sp = sp & ~nodata_mask
    lbl = _label(sp, connectivity=connectivity) # find areas of connectivity in the array
    n = int(lbl.max()) #number of objs found
    meta = meta or {}

    cols = ["label", "area_px", "area_m2", "eq_diam_px", "bbox_h_px", "bbox_w_px", "min_dim_px",
            "major_axis_px", "minor_axis_px", "perimeter_px", "compactness",
            "solidity", "centroid_row", "centroid_col", "touches_edge", "touches_nodata"]
    if n == 0: #no objects found output
        return pd.DataFrame(columns=list(meta) + cols)

    edge_ids = set(np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))) - {0} #first and last obj id
    nd_ids = set()
    if nodata_mask is not None and nodata_mask.any():
        near_nd = binary_dilation(nodata_mask, structure=np.ones((3, 3))) #eight connectivity dilation
        nd_ids = set(np.unique(lbl[near_nd & (lbl > 0)])) - {0} #goal is to find pixels near nodata edges

    px_area_m2 = pixel_size_m ** 2 if pixel_size_m else np.nan #get pixel size in m^2
    rows = []
    for p in _regionprops(lbl): #loop over all of the properties of each object found
        minr, minc, maxr, maxc = p.bbox
        area = int(p.area)
        perim = float(p.perimeter)
        rows.append({
            **meta,
            "label": p.label,
            "area_px": area,
            "area_m2": area * px_area_m2,
            "eq_diam_px": float(p.equivalent_diameter_area if hasattr(p, "equivalent_diameter_area")
                                else p.equivalent_diameter),
            "bbox_h_px": maxr - minr,
            "bbox_w_px": maxc - minc,
            "min_dim_px": min(maxr - minr, maxc - minc),   # robust "width" for tiny objects
            "major_axis_px": float(_prop(p, "axis_major_length", "major_axis_length")),
            "minor_axis_px": float(_prop(p, "axis_minor_length", "minor_axis_length")),
            "perimeter_px": perim,
            "compactness": (4 * np.pi * area / perim ** 2) if perim > 0 else np.nan,  # 1 = circle
            "solidity": float(p.solidity),
            "centroid_row": float(p.centroid[0]),
            "centroid_col": float(p.centroid[1]),
            "touches_edge": p.label in edge_ids,
            "touches_nodata": p.label in nd_ids,
        })
        print("inv_from_array cleared!")
    return pd.DataFrame(rows) #output


def inventory_from_raster(path, sp_value=1, connectivity=2, meta=None):
    #Read a mask GeoTIFF with rasterio into an array and inventory its pans
    import rasterio

    with rasterio.open(path) as src: #read tiff and get metadata if available
        band = src.read(1)
        nodata = src.nodata
        res_x, res_y = src.res
        crs = src.crs

    pixel_size_m = None
    if crs is not None and crs.is_projected:
        try:
            factor = crs.linear_units_factor[1]      # e.g. 0.3048 for US feet
        except Exception:
            factor = 1.0
        pixel_size_m = float(np.sqrt(res_x * res_y) * factor)
        print(f"Pixel size found to be {pixel_size_m} m^2")
    else:
        print(f"  note: {os.path.basename(path)} is not in a projected CRS, area_m2 will be NaN")

    # get file's nodata tag
    nodata_mask = np.zeros(band.shape, bool)
    if band.dtype.kind == "f":
        nodata_mask |= np.isnan(band)
    nodata_mask |= band < 0
    if nodata is not None and not np.isnan(nodata):
        nodata_mask |= band == nodata
    #get files metadata and feed all info to array inv func
    meta = {"file": os.path.basename(path), **(meta or {})}
    print("inv_from_raster cleared!")
    return inventory_from_array(band, sp_value, pixel_size_m, nodata_mask, connectivity, meta)


def inventory_from_folders(folders, patterns=("*.tif", "*.tiff"), sp_value=1, verbose=True):
    #
    # folders: {(split, month): folder_with_mask_tifs}, e.g.
    #          {("train", "2025-05"): May_train_mask, ("val", "2025-06"): June_val_mask}
    #
    frames = []
    for (split, month), folder in folders.items():
        files = sorted(f for pat in patterns for f in glob.glob(os.path.join(folder, pat)))
        if verbose:
            print(f"{split} {month}: {len(files)} mask files")
        for f in files:
            frames.append(inventory_from_raster(f, sp_value, meta={"split": split, "month": month}))
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_inventory(df, bins_px=DEFAULT_BINS, mmu_px=(4, 16, 52, 125, 500), verbose=True):
    #print and return the tables that matter for choosing size thresholds
    if df.empty:
        print("empty inventory")
        return {}
    out = {}
    total_px = df["area_px"].sum()
    have_m2 = df["area_m2"].notna().any()

    if verbose:
        print(f"\n{len(df)} pans from {df['file'].nunique() if 'file' in df else '?'} mask files | "
              f"total labeled area {total_px:,} px" +
              (f" = {df['area_m2'].sum() / 1e6:.2f} km2" if have_m2 else ""))
        print("NOTE: the same ground is masked in several months, so the same physical pan can appear "
              "once per month - counts are not additive across months.")

    # 1. percentiles
    q = [5, 25, 50, 75, 95, 99]
    pct = pd.DataFrame({
        "area_px": np.percentile(df["area_px"], q),
        "eq_diam_px": np.percentile(df["eq_diam_px"], q),
        "min_dim_px": np.percentile(df["min_dim_px"], q),
    }, index=[f"p{x}" for x in q])
    out["percentiles"] = pct
    if verbose:
        print("\nSIZE PERCENTILES"); print(pct.round(1).to_string())

    # 2. size bins: share of count vs share of area
    edges = list(bins_px)
    labels = [f"{int(lo)}-{'inf' if np.isinf(hi) else int(hi)}" for lo, hi in zip(edges[:-1], edges[1:])]
    cut = pd.cut(df["area_px"], bins=edges, labels=labels, right=False)
    g = df.groupby(cut, observed=False)
    bin_tbl = pd.DataFrame({
        "n": g.size(),
        "pct_of_pans": g.size() / len(df) * 100,
        "pct_of_area": g["area_px"].sum() / total_px * 100,
        "pct_touch_edge": g["touches_edge"].mean() * 100,
    })
    out["size_bins"] = bin_tbl
    if verbose:
        print("\nSIZE BINS (px) - share of pans vs share of labeled area")
        print(bin_tbl.round(1).to_string())

    # 3. minimum mapping unit trade-off
    rows = []
    for m in mmu_px:
        keep = df["area_px"] >= m
        rows.append({"min_size_px": m,
                     "pans_kept": int(keep.sum()),
                     "pct_pans_kept": keep.mean() * 100,
                     "pct_area_kept": df.loc[keep, "area_px"].sum() / total_px * 100})
    mmu_tbl = pd.DataFrame(rows).set_index("min_size_px")
    out["mmu"] = mmu_tbl
    if verbose:
        print("\nMINIMUM MAPPING UNIT - what you keep if you only score pans >= N px")
        print(mmu_tbl.round(1).to_string())

    # 4. thin / fragmentary objects
    thin = (df["min_dim_px"] < 3).mean() * 100
    tiny = (df["area_px"] < 4).mean() * 100
    edge = df["touches_edge"].mean() * 100
    if verbose:
        print(f"\n{tiny:.1f}% of pans are < 4 px (likely fragments)  |  "
              f"{thin:.1f}% are narrower than 3 px  |  {edge:.1f}% touch the mask edge "
              f"(true size unknown)")

    # 5. by split / month
    if {"split", "month"} <= set(df.columns):
        gm = df.groupby(["split", "month"])
        by = pd.DataFrame({
            "pans": gm.size(),
            "median_area_px": gm["area_px"].median(),
            "pct_under_16px": gm["area_px"].apply(lambda s: (s < 16).mean() * 100),
            "area_px_total": gm["area_px"].sum(),
        })
        out["by_split_month"] = by
        if verbose:
            print("\nBY SPLIT / MONTH"); print(by.round(1).to_string())
    return out


def plot_inventory(df, save_path=None, show=True):
    #size histogram + cumulative curves
    import matplotlib.pyplot as plt
    if df.empty:
        return None
    a = np.sort(df["area_px"].to_numpy())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    bins = np.logspace(0, np.log10(max(a.max(), 10)), 40)
    ax[0].hist(a, bins=bins, color="steelblue")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("pan area (px)"); ax[0].set_ylabel("number of pans")
    ax[0].set_title("Pan size distribution")
    for e in DEFAULT_BINS[1:-1]:
        ax[0].axvline(e, color="gray", lw=.6, ls="--")

    thr = np.unique(a)
    frac_count = np.array([(a >= t).mean() for t in thr])
    cum_area = np.cumsum(a[::-1])[::-1]                       # area in pans >= each sorted size
    first_idx = np.searchsorted(a, thr, side="left")
    frac_area = cum_area[first_idx] / a.sum()
    ax[1].plot(thr, frac_count, label="share of pans with area >= x")
    ax[1].plot(thr, frac_area, label="share of labeled area in pans >= x")
    ax[1].set_xscale("log"); ax[1].set_xlabel("minimum pan size x (px)")
    ax[1].set_ylim(0, 1.02); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[1].set_title("Minimum-mapping-unit trade-off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def inventory_from_polygons(path, layer=None, class_field=None, class_value=None,
                            crs=None, pixel_size_m=3.0):


    # Reprojects to a metric CRS (UTM estimated from the data unless `crs` is given),
    # splits multipart geometries, and reports true polygon area in m2 plus
    # `area_px_equiv` = area expressed in pixels of `pixel_size_m`, so it can be
    # compared directly with the raster inventory.
    #
    import geopandas as gpd

    gdf = gpd.read_file(path, layer=layer)
    if class_field is not None:
        gdf = gdf[gdf[class_field] == class_value]
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf.to_crs(crs or gdf.estimate_utm_crs())

    geom = gdf.geometry
    area, perim = geom.area, geom.length
    lengths, widths = [], []
    for poly in geom.minimum_rotated_rectangle():
        c = np.array(poly.exterior.coords)[:3]
        s1, s2 = np.hypot(*(c[1] - c[0])), np.hypot(*(c[2] - c[1]))
        lengths.append(max(s1, s2)); widths.append(min(s1, s2))

    return pd.DataFrame({
        "area_m2": area.to_numpy(),
        "area_px_equiv": area.to_numpy() / pixel_size_m ** 2,
        "perimeter_m": perim.to_numpy(),
        "compactness": (4 * np.pi * area / perim ** 2).to_numpy(),
        "length_m": lengths,
        "width_m": widths,
    })


#Following functions are for removing polygons below a certain size threshold
def remove_small_objects(mask, min_size, connectivity=2):
    #Returns (cleaned_mask, n_before, n_after) using a single labeling pass
    structure = ndi.generate_binary_structure(2, connectivity)
    labels, n_before = ndi.label(mask > 0, structure=structure)

    if n_before == 0:
        return mask.copy(), 0, 0

    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False

    cleaned = keep[labels].astype(mask.dtype)
    del labels
    return cleaned, n_before, int(keep.sum())


def clean_raster(input, output, min_size, connectivity=2, band=1):
    with rasterio.open(input) as src:
        mask = src.read(band)
        profile = src.profile
        nodata = src.nodata
        print(f"Raster size: {src.width} x {src.height}, dtype: {mask.dtype}, nodata: {nodata}")

    if nodata is not None:
        mask[mask == nodata] = 0   # in-place, avoids a second full copy

    if mask.min() < 0 or mask.max() > 1:
        raise ValueError(f"Unexpected values: min={mask.min()}, max={mask.max()}")

    cleaned, n_before, n_after = remove_small_objects(mask, min_size, connectivity)
    del mask

    profile.update(count=1, compress="lzw")
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(cleaned, 1)

    print(f"Objects before: {n_before}")
    print(f"Objects after:  {n_after}")
    print(f"Removed:        {n_before - n_after} (< {min_size} px)")
#running

if __name__ == "__main__":
    res3m = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/masks3mRes/'

    May_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-05/masks/'
    May_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-05/masks/'

    June_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-06/masks/'
    June_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-06/masks/'

    Aug_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-08/masks/'
    Aug_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-08/masks/'

    Sept_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-09/masks/'
    Sept_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-09/masks/'

    Oct_train_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/train/2025-10/masks/'
    Oct_val_mask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/val/2025-10/masks/'

    folders = {
        ("train", "2025-05"): res3m
    }

    df = inventory_from_folders(folders)

    # tables = summarize_inventory(df)
    # plot_inventory(df, "3mrespan_inventory.png")
    # df.to_csv("3mrespan_inventory.csv", index=False)

    #clean raster of all pans below 52 pixels
    inmask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/masks3mRes/collatedSPmask3m.tif'
    outmask = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/cleaned_3mrestruthmask_32up.tif'
    clean_raster(input = inmask,output = outmask,min_size=32)