import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import SurfaceDiceMetric, HausdorffDistanceMetric
from skimage.measure import label as cc_label
from scipy.optimize import linear_sum_assignment


def _filter_by_size(bin_img, lo, hi):
    """Keep only connected components whose pixel area is in [lo, hi)."""
    lbl = cc_label(bin_img, connectivity=2)
    if lbl.max() == 0:
        return bin_img
    sizes = np.bincount(lbl.ravel())
    keep = (sizes >= lo) & (sizes < hi)
    keep[0] = False
    return keep[lbl]


@torch.no_grad()
def collect_pred_target_tiles(model, dataloader, prepare_batch, max_tiles=400,
                              stride=1, device="cuda"):
    """
    Run the model over a dataloader and keep binary prediction / target / valid
    tiles as numpy bool arrays (for estimate_pred_label_offset).

    stride: keep every `stride`-th tile, so max_tiles covers more of the study
            area instead of only the first tiles the grid sampler yields.
    Returns (preds, targs, valids), each a list of (H, W) bool arrays.
    """
    model.eval()
    preds, targs, valids = [], [], []
    seen = 0
    for scene in dataloader:
        prepared = prepare_batch(scene)
        if prepared is None:
            continue
        x, y, nodata = prepared
        x2 = torch.cat((x[:, :5, :, :], x[:, 6:, :, :]), dim=1)#REMOVE NWI
        out = model(x2.to(device)).argmax(dim=1).cpu().numpy() == 1
        yt = y.cpu().numpy() == 1
        nd = nodata.cpu().numpy().astype(bool)
        for b in range(out.shape[0]):
            seen += 1
            if (seen - 1) % stride != 0:
                continue
            preds.append(out[b]); targs.append(yt[b]); valids.append(~nd[b])
            if len(preds) >= max_tiles:
                return preds, targs, valids
    return preds, targs, valids


def estimate_pred_label_offset(preds, targs, valids=None, max_shift=4,
                               size_range=None, min_fg=30, verbose=True):
    """
    Estimate whether predictions are systematically displaced from the labels.

    For every integer shift (dy, dx) in [-max_shift, max_shift]^2, compares
    pred[y + dy, x + dx] against target[y, x] and pools intersection / union
    over all tiles. Convention: a best shift of (dy, dx) means the model output
    is displaced `dy` rows DOWN and `dx` columns RIGHT relative to the labels.

    size_range: optional (lo, hi) pixel areas - keep only components in that
                range in BOTH pred and target, e.g. (0, 52) for small pans, so
                you can compare small-pan alignment with large-pan alignment.
    min_fg:     tiles with fewer target pixels (after filtering) are skipped.

    Returns a dict with the pooled IoU grid, best shift, IoU at zero shift vs.
    best shift, per-tile best shifts, and the pooled pred/target area ratio.

    Reading it:
      - sharp peak at (0, 0), best ~= zero-shift IoU     -> no alignment problem
      - sharp peak at a consistent non-zero shift         -> systematic displacement
      - low, flat grid / per-tile shifts scattered widely -> inconsistent registration
        between labels and imagery (varies tile to tile)
    Caveat: a CNN can learn a *consistent* label offset, in which case pred and
    labels agree and the offset is invisible here. Scatter in the per-tile shifts
    is the more reliable signal of misregistration.
    """
    m = int(max_shift)
    assert m >= 1, "max_shift must be >= 1"
    shifts = list(range(-m, m + 1))
    n = len(shifts)
    inter = np.zeros((n, n))
    union = np.zeros((n, n))
    per_tile = []
    pred_px = targ_px = 0

    for t, (p, g) in enumerate(zip(preds, targs)):
        p = np.asarray(p, bool)
        g = np.asarray(g, bool)
        if size_range is not None:
            p = _filter_by_size(p, *size_range)
            g = _filter_by_size(g, *size_range)
        v = np.asarray(valids[t], bool) if valids is not None else np.ones_like(g)

        H, W = g.shape
        gc = g[m:H - m, m:W - m]
        vc = v[m:H - m, m:W - m]
        if (gc & vc).sum() < min_fg:
            continue

        tile_i = np.zeros((n, n))
        tile_u = np.zeros((n, n))
        for iy, dy in enumerate(shifts):
            for ix, dx in enumerate(shifts):
                pc = p[m + dy:H - m + dy, m + dx:W - m + dx]
                tile_i[iy, ix] = np.logical_and(pc, gc)[vc].sum()
                tile_u[iy, ix] = np.logical_or(pc, gc)[vc].sum()
        inter += tile_i
        union += tile_u
        pred_px += (p[m:H - m, m:W - m] & vc).sum()
        targ_px += (gc & vc).sum()

        tile_iou = tile_i / np.maximum(tile_u, 1)
        if tile_iou.max() >= 0.05:      # ignore tiles with essentially no overlap anywhere
            by, bx = np.unravel_index(np.argmax(tile_iou), tile_iou.shape)
            per_tile.append((shifts[by], shifts[bx], tile_iou[by, bx], tile_iou[m, m]))

    if union.sum() == 0:
        print("No usable tiles (check min_fg / size_range).")
        return {}

    grid = inter / np.maximum(union, 1)
    by, bx = np.unravel_index(np.argmax(grid), grid.shape)
    per_tile = np.array(per_tile) if per_tile else np.zeros((0, 4))
    result = {
        "shifts": shifts,
        "grid": grid,
        "best_shift": (shifts[by], shifts[bx]),
        "iou_at_zero": float(grid[m, m]),
        "iou_at_best": float(grid[by, bx]),
        "per_tile": per_tile,           # columns: dy, dx, iou_at_best, iou_at_zero
        "area_ratio": float(pred_px / max(targ_px, 1)),   # >1: predictions larger than labels
        "n_tiles_used": int(len(per_tile)),
    }

    if verbose:
        print(f"\nPooled IoU vs. shift of prediction (rows = dy, cols = dx), size_range={size_range}")
        print("      " + "".join(f"{dx:>7d}" for dx in shifts))
        for iy, dy in enumerate(shifts):
            print(f"{dy:>5d} " + "".join(f"{grid[iy, ix]:>7.3f}" for ix in range(n)))
        gain = (result["iou_at_best"] / max(result["iou_at_zero"], 1e-9) - 1) * 100
        print(f"\nbest shift (dy, dx) = {result['best_shift']}  |  IoU at zero shift = "
              f"{result['iou_at_zero']:.3f}  |  IoU at best shift = {result['iou_at_best']:.3f}  "
              f"(+{gain:.0f}% relative)")
        print(f"pooled pred/target area ratio = {result['area_ratio']:.2f}  (>1 = predictions larger than labels)")
        if len(per_tile):
            dy_t, dx_t = per_tile[:, 0], per_tile[:, 1]
            within1 = np.mean((np.abs(dy_t) <= 1) & (np.abs(dx_t) <= 1))
            print(f"per-tile best shifts ({len(per_tile)} tiles): median (dy, dx) = "
                  f"({np.median(dy_t):.0f}, {np.median(dx_t):.0f}), std = ({dy_t.std():.1f}, {dx_t.std():.1f}), "
                  f"{within1:.0%} within 1 px of (0, 0)")
    return result


def plot_offset_result(result, save_path=None, show=True):
    """Heatmap of pooled IoU vs. shift, plus scatter of per-tile best shifts."""
    import matplotlib.pyplot as plt
    if not result:
        return None
    shifts = result["shifts"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    im = axes[0].imshow(result["grid"], cmap="viridis",
                        extent=[shifts[0] - .5, shifts[-1] + .5, shifts[-1] + .5, shifts[0] - .5])
    axes[0].set_title("Pooled IoU vs. shift of prediction")
    axes[0].set_xlabel("dx (cols, + = right)")
    axes[0].set_ylabel("dy (rows, + = down)")
    axes[0].plot(*result["best_shift"][::-1], "r*", ms=12)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    pt = result["per_tile"]
    if len(pt):
        # jitter slightly so stacked tiles are visible
        rng = np.random.default_rng(0)
        axes[1].scatter(pt[:, 1] + rng.normal(0, .06, len(pt)),
                        pt[:, 0] + rng.normal(0, .06, len(pt)), s=14, alpha=.5)
    axes[1].axhline(0, color="gray", lw=.5); axes[1].axvline(0, color="gray", lw=.5)
    axes[1].set_xlim(shifts[0] - .7, shifts[-1] + .7)
    axes[1].set_ylim(shifts[-1] + .7, shifts[0] - .7)
    axes[1].set_title("Per-tile best shift")
    axes[1].set_xlabel("dx"); axes[1].set_ylabel("dy")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


@torch.no_grad()
def collect_pan_crops(dataloader, prepare_batch, size_range=(0, 16), n_crops=24,
                      window=48, model=None, device="cuda", min_size=4, seed=0):
    """
    Collect a uniform random sample (reservoir sampling over the WHOLE dataloader)
    of true pans whose pixel area falls in size_range, as small image/mask crops.

    Raw (un-normalized) imagery is captured BEFORE prepare_batch runs, since
    normalization may modify the batch in place. If a model is given, its
    predicted mask and the fraction of each pan it covers are stored too.
    Assumes band order B, G, R, NIR in channels 0-3.
    """
    rng = np.random.default_rng(seed)
    if model is not None:
        model.eval()
    reservoir, seen = [], 0

    for scene in dataloader:
        raw = scene["image"].detach().clone().cpu().numpy()      # copy first
        prepared = prepare_batch(scene)
        if prepared is None:
            continue
        x, y, nodata = prepared
        targ = y.cpu().numpy() == 1
        nd = nodata.cpu().numpy().astype(bool)
        pred = None
        if model is not None:
            x2 = torch.cat((x[:, :5, :, :], x[:, 6:, :, :]), dim=1) #remove NWI
            pred = model(x2.to(device)).argmax(dim=1).cpu().numpy() == 1

        for b in range(targ.shape[0]):
            g = targ[b] & ~nd[b]
            lbl = cc_label(g, connectivity=2)
            if lbl.max() == 0:
                continue
            sizes = np.bincount(lbl.ravel())
            H, W = g.shape
            win = min(window, H, W)
            for k in range(1, lbl.max() + 1):
                s = int(sizes[k])
                if s < min_size or not (size_range[0] <= s < size_range[1]):
                    continue
                ys, xs = np.nonzero(lbl == k)
                y0 = int(np.clip(int(ys.mean()) - win // 2, 0, H - win))
                x0 = int(np.clip(int(xs.mean()) - win // 2, 0, W - win))
                sl = (slice(y0, y0 + win), slice(x0, x0 + win))
                crop = {
                    "size": s,
                    "image": raw[b][:, sl[0], sl[1]].astype(np.float32),
                    "target": g[sl],
                    "pred": pred[b][sl] if pred is not None else None,
                    "pred_cover": float(pred[b][lbl == k].mean()) if pred is not None else None,
                }
                seen += 1
                if len(reservoir) < n_crops:
                    reservoir.append(crop)
                else:
                    j = int(rng.integers(0, seen))
                    if j < n_crops:
                        reservoir[j] = crop
    print(f"sampled {len(reservoir)} of {seen} pans with area in {size_range}")
    return reservoir


def plot_pan_crops(crops, bands="rgb", crops_per_row=3, stretch=(2, 98),
                   save_path=None, show=True):
    """
    Preview crops from collect_pan_crops. Each crop gets two panels: the raw
    imagery alone (so the overlay can't hide a faint pan), and the same imagery
    with the label outline (lime) and, if available, the prediction outline (red).
    bands: "rgb" or "cir" (NIR-R-G false colour; vegetation shows red).
    One contrast stretch is shared by all crops so brightness is comparable.
    """
    import matplotlib.pyplot as plt
    if not crops:
        print("no crops to plot")
        return None
    idx = {"rgb": [2, 1, 0], "cir": [3, 2, 1]}[bands]
    imgs = np.stack([c["image"][idx] for c in crops])            # (N, 3, h, w)
    disp = np.zeros_like(imgs)
    for i in range(3):
        band = imgs[:, i]
        vals = band[band > 0]
        lo, hi = (np.percentile(vals, stretch) if vals.size else (0.0, 1.0))
        disp[:, i] = np.clip((band - lo) / max(hi - lo, 1e-9), 0, 1)

    n = len(crops)
    rows = int(np.ceil(n / crops_per_row))
    fig, axes = plt.subplots(rows, crops_per_row * 2, squeeze=False,
                             figsize=(crops_per_row * 4.2, rows * 2.2))
    for a in axes.ravel():
        a.axis("off")
    for i, c in enumerate(crops):
        r, col = divmod(i, crops_per_row)
        raw_ax, ov_ax = axes[r, 2 * col], axes[r, 2 * col + 1]
        rgb = np.transpose(disp[i], (1, 2, 0))
        raw_ax.imshow(rgb, interpolation="nearest")
        ov_ax.imshow(rgb, interpolation="nearest")
        raw_ax.set_title(f"{c['size']} px", fontsize=8)
        if c["target"].any():
            ov_ax.contour(c["target"].astype(float), levels=[0.5], colors="lime", linewidths=2.2)
        if c["pred"] is not None and c["pred"].any():
            ov_ax.contour(c["pred"].astype(float), levels=[0.5], colors="red", linewidths=0.9)
        if c["pred_cover"] is not None:
            ov_ax.set_title(f"covered {c['pred_cover']:.0%}", fontsize=8)
    fig.suptitle(f"{bands.upper()}: label = lime, prediction = red", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig

