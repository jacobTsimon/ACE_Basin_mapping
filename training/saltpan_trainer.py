
import time
from helper_functions import MinMaxNormalize,F1_score,apply_colormap_to_mask
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import torchvision
from torchvision.transforms import v2
from torchgeo.transforms import AppendNDWI,AppendNDVI
import ignite
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.utils import make_grid
from monai.metrics import SurfaceDiceMetric, HausdorffDistanceMetric
from skimage.measure import label as cc_label
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_opening,binary_closing


#introduce colors for output legibility
torch.set_printoptions(threshold=10000) #increase print space
#colors/fonts
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#function for plotting a logged CM
def plot_confusion_matrix(cm, class_names):
    #turn torch tensor to numpy array
    cm = cm.detach().cpu().numpy()
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.astype(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df_cm, annot=True, ax=ax, cmap='Spectral') #, fmt="d"
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

class early_stop:
    def __init__(self, patience = 3, tolerance = 0.1):
        self.patience = patience
        self.tolerance = tolerance #tolerance range before value above min is counted

        self.count = 0
        self.min_loss = float(100000)
    def run(self,val_loss):
        if val_loss < self.min_loss:
            self.min_loss = val_loss    #new minimum achieved
            self.count = 0  #reset timer
        elif val_loss > (self.min_loss + self.tolerance):
            self.count += 1
            if self.count >= self.patience:
                return True
        return False

class MSAVI(nn.Module):
    def __init__(self,indexNIR: int,indexRED: int):
        super().__init__()
        self.indexNIR = indexNIR
        self.indexRED = indexRED
    def forward(self, input: torch.Tensor):
        #gether the relevant indices
        NIR = input[:,self.indexNIR,:,:].numpy()
        R = input[:,self.indexRED,:,:].numpy()
        #calculations
        NIR2 = (2*NIR) + 1
        out = (NIR2 - np.sqrt(NIR2**2 - (8*(NIR-R)))) / 2
        out = torch.from_numpy(out).float()
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"WARNING: MSAVI produced NaN/Inf values!")
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

        return torch.cat([input,out.unsqueeze(dim=1)],dim=1)

#boundary accuracy detection

class BoundaryMetrics:

    #class_thresholds: tolerance for Surface Dice, in PIXELS

    def __init__(self, class_thresholds=(3.0,), num_classes=2):
        self.surface_dice = SurfaceDiceMetric( #docs @ https://monai-dev.readthedocs.io/en/fixes-sphinx/metrics.html#surface-dice
            class_thresholds=list(class_thresholds),
            include_background=False,  # ignore background
            distance_metric="euclidean",
            reduction="mean",
        )
        self.hausdorff95 = HausdorffDistanceMetric( #docs @ https://monai-dev.readthedocs.io/en/fixes-sphinx/metrics.html#hausdorff-distance
            include_background=False,
            distance_metric="euclidean",
            percentile=95,  # raw max Hausdorff is dominated by single outlier pixels
            reduction="mean",
        )
        self.num_classes = num_classes

    def reset(self):
        self.surface_dice.reset()
        self.hausdorff95.reset()

    def update(self, outputs, y, nodata=None):

        #method for taking in raw model outputs and switching to one hot
        #outputs: raw model logits, shape (B, C, H, W)
        #y:       target labels, shape (B, H, W)
        #nodata:  optional bool mask, shape (B, H, W),

        pred_labels = outputs.detach().argmax(dim=1)  # (B, H, W)
        targ_labels = y.detach().clone().long()

        if nodata is not None:
            # force agreement in nodata regions so they can't manufacture
            # spurious boundaries
            pred_labels = pred_labels.clone()
            pred_labels[nodata] = 0
            targ_labels[nodata] = 0

        pred_onehot = F.one_hot(pred_labels, self.num_classes).permute(0, 3, 1, 2).float()
        targ_onehot = F.one_hot(targ_labels, self.num_classes).permute(0, 3, 1, 2).float()

        self.surface_dice(y_pred=pred_onehot, y=targ_onehot)
        self.hausdorff95(y_pred=pred_onehot, y=targ_onehot)

    def compute(self):
        # aggregate() reduces over not-nan values only - batches with no
        # foreground in either pred or target (nan) are skipped rather than
        # poisoning the average
        return {
            "surface_dice": self.surface_dice.aggregate().item(),
            "hausdorff95": self.hausdorff95.aggregate().item(),
        }


def _drop_small_components(lbl, min_size):
    #
    # Remove connected components smaller than min_size pixels from a labeled image and relabel the survivors 1..N.
    # important to apply to BOTH prediction and target so tiny specks are excluded symmetrically
    #
    if min_size <= 1 or lbl.max() == 0:
        return lbl
    sizes = np.bincount(lbl.ravel())
    small_ids = np.where(sizes < min_size)[0]
    small_ids = small_ids[small_ids != 0]
    if len(small_ids) == 0:
        return lbl
    cleaned = lbl.copy()
    cleaned[np.isin(cleaned, small_ids)] = 0
    return cc_label(cleaned > 0, connectivity=2)


def _diagnose_single(pred_bin, targ_bin, iou_thresh, min_size):

    # Numpy core of the diagnostics for one tile (pred_bin / targ_bin are 2D bool
    # arrays). Returns, for every TRUE object, its pixel size and the best IoU
    # achieved against any overlapping predicted component - and the same for
    # every PREDICTED object against the targets (the precision side).
    # Note: best-IoU is not a one-to-one assignment, so one big predicted blob can
    # be the "best match" for several targets; this is slightly optimistic
    # compared with the Hungarian matching used by ObjectMetrics.
    #
    pred_lbl = _drop_small_components(cc_label(pred_bin, connectivity=2), min_size)
    targ_lbl = _drop_small_components(cc_label(targ_bin, connectivity=2), min_size)
    n_pred, n_targ = int(pred_lbl.max()), int(targ_lbl.max())

    def _best_ious(src_lbl, n_src, other_lbl):
        sizes, best = [], []
        for k in range(1, n_src + 1):
            mask = src_lbl == k
            sizes.append(int(mask.sum()))
            b = 0.0
            for j in np.unique(other_lbl[mask]):
                if j == 0:
                    continue
                other = other_lbl == j
                inter = np.logical_and(mask, other).sum()
                union = np.logical_or(mask, other).sum()
                b = max(b, inter / union)
            best.append(b)
        return sizes, best

    target_sizes, best_iou_per_target = _best_ious(targ_lbl, n_targ, pred_lbl)
    pred_sizes, best_iou_per_pred = _best_ious(pred_lbl, n_pred, targ_lbl)

    return {
        "n_pred_components": n_pred,
        "n_targ_components": n_targ,
        "best_iou_per_target": best_iou_per_target,  # one entry per true pan
        "target_sizes": target_sizes,  # pixel area of each true pan
        "best_iou_per_pred": best_iou_per_pred,  # one entry per predicted object
        "pred_sizes": pred_sizes,  # pixel area of each predicted object
        "total_misses": sum(1 for v in best_iou_per_target if v == 0.0),
        "near_misses": sum(1 for v in best_iou_per_target if 0.0 < v < iou_thresh),
        "clean_matches": sum(1 for v in best_iou_per_target if v >= iou_thresh),
    }


#troubleshooting object metrics
def diagnose_object_matches(outputs, y, nodata=None, iou_thresh=0.5, min_pred_size=4):
    #
    # For each image in the batch, reports predicted vs. target component counts,
    # and for every TRUE object, the best IoU achieved against any overlapping
    # predicted component. This distinguishes:
    #   - total miss   (best_iou == 0.0)       -> model never predicted anything there
    #   - near miss    (0.0 < best_iou < thresh) -> model found it, just not tightly enough
    #   - clean match  (best_iou >= thresh)
    #
    pred_labels = outputs.detach().argmax(dim=1)
    targ_labels = y.detach().clone()
    if nodata is not None:
        nodata = nodata.to(pred_labels.device)
        pred_labels = pred_labels.clone()
        pred_labels[nodata] = 0
        targ_labels[nodata] = 0

    pred_np = pred_labels.cpu().numpy()
    targ_np = targ_labels.cpu().numpy()

    return [
        _diagnose_single(pred_np[b] == 1, targ_np[b] == 1, iou_thresh, min_pred_size)
        for b in range(pred_np.shape[0])
    ]



class ObjectMetrics:

    # object-level precision / recall / F1.
    #
    # iou_thresh:    IoU a predicted component must reach against a target
    #                component to count as a correct match
    # min_pred_size: drop predicted components smaller than this many pixels
    #                before matching - filters single-pixel noise from ever
    #                counting as a false positive "found" object


    def __init__(self, iou_thresh=0.5, min_pred_size=4):
        self.iou_thresh = iou_thresh
        self.min_pred_size = min_pred_size
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, outputs, y, nodata=None):
        #same usage as above, transform raw model outputs to one hot encoding
        pred_labels = outputs.detach().argmax(dim=1)
        targ_labels = y.detach().clone()

        if nodata is not None:
            pred_labels = pred_labels.clone()
            pred_labels[nodata] = 0
            targ_labels[nodata] = 0

        pred_np = pred_labels.cpu().numpy()
        targ_np = targ_labels.cpu().numpy()

        for b in range(pred_np.shape[0]):
            tp, fp, fn = self._match_objects(pred_np[b] == 1, targ_np[b] == 1)
            self.tp += tp
            self.fp += fp
            self.fn += fn

    def _match_objects(self, pred_bin, targ_bin): #relies on scikitimage label function, docs @ https://scikit-image.org/docs/0.25.x/api/skimage.measure.html#skimage.measure.label
        #connectivity is the range from a centroid pixel that is considered part of the object
        pred_lbl = _drop_small_components(cc_label(pred_bin, connectivity=2), self.min_pred_size)
        targ_lbl = _drop_small_components(cc_label(targ_bin, connectivity=2), self.min_pred_size)
        #set to 2 it goes 2 "hops", captures all eight ssurrounding pxls
        n_pred, n_targ = int(pred_lbl.max()), int(targ_lbl.max()) #get number of regions identified

        if n_pred == 0 and n_targ == 0:
            return 0, 0, 0
        if n_targ == 0:
            return 0, n_pred, 0
        if n_pred == 0:
            return 0, 0, n_targ

        iou = np.zeros((n_pred, n_targ))
        for i in range(1, n_pred + 1):
            pmask = pred_lbl == i
            if pmask.sum() < self.min_pred_size:
                continue
            # only check target components that actually overlap this
            # predicted component - avoids an O(n_pred * n_targ) full scan
            candidates = np.unique(targ_lbl[pmask])
            for j in candidates:
                if j == 0:
                    continue
                tmask = targ_lbl == j
                inter = np.logical_and(pmask, tmask).sum()
                union = np.logical_or(pmask, tmask).sum()
                iou[i - 1, j - 1] = inter / union

        row_ind, col_ind = linear_sum_assignment(-iou)  # maximize total IoU, docs @ https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        tp = 0
        matched_pred, matched_targ = set(), set()
        for r, c in zip(row_ind, col_ind):
            if iou[r, c] >= self.iou_thresh:
                tp += 1
                matched_pred.add(r)
                matched_targ.add(c)

        fp = n_pred - len(matched_pred)
        fn = n_targ - len(matched_targ)
        return tp, fp, fn

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-9)
        recall = self.tp / (self.tp + self.fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        return {
            "obj_precision": precision,
            "obj_recall": recall,
            "obj_f1": f1,
            "obj_tp": self.tp,
            "obj_fp": self.fp,
            "obj_fn": self.fn,
        }


def size_stratified_report(results, bins=None, iou_thresh=0.5, verbose=True):

    # Aggregate diagnose_object_matches() results into a size-stratified table.
    #
    # results:    list of per-tile dicts (as returned by diagnose_object_matches
    #             / collect_object_diagnostics)
    # bins:       bin edges in PIXELS, e.g. [0, 20, 100, 500, np.inf]. If None,
    #             edges are the 25/50/75th percentiles of the true-object sizes,
    #             so every bin has a reasonable number of objects.
    # iou_thresh: what counts as a clean match. Best IoUs are stored, so you can
    #             call this again with 0.3 without re-running the model.
    #
    # Recall side:    grouped by TRUE object size  -> are small pans the ones missed?
    # Precision side: grouped by PREDICTED object size -> are false positives small specks?
    #

    def _cat(key, dtype):
        arrs = [np.asarray(r[key], dtype=dtype) for r in results if len(r[key]) > 0]
        return np.concatenate(arrs) if arrs else np.array([], dtype=dtype)

    t_sizes, t_best = _cat("target_sizes", int), _cat("best_iou_per_target", float)
    p_sizes, p_best = _cat("pred_sizes", int), _cat("best_iou_per_pred", float)

    if len(t_sizes) == 0:
        print("No true objects found in results.")
        return {}

    if bins is None:
        qs = np.percentile(t_sizes, [25, 50, 75]).astype(int).tolist()
        bins = [0] + sorted(set(q for q in qs if q > 0)) + [np.inf]

    def _rows(sizes, best):
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (sizes >= lo) & (sizes < hi)
            n = int(m.sum())
            if n == 0:
                continue
            b = best[m]
            rows.append({
                "lo": lo, "hi": hi, "n": n,
                "no_overlap": float(np.mean(b == 0)),
                "low_iou": float(np.mean((b > 0) & (b < iou_thresh))),
                "matched": float(np.mean(b >= iou_thresh)),
                "median_best_iou": float(np.median(b)),
            })
        return rows

    recall_rows = _rows(t_sizes, t_best)
    precision_rows = _rows(p_sizes, p_best)

    def _print(title, rows, labels):
        if title:
            print(title)
        print(f"  {'size (px)':>12} {'n':>6} {labels[0]:>10} {labels[1]:>10} {labels[2]:>10} {'med IoU':>8}")
        for r in rows:
            hi = "inf" if np.isinf(r["hi"]) else str(int(r["hi"]))
            print(f"  {str(int(r['lo'])) + '-' + hi:>12} {r['n']:>6} "
                  f"{r['no_overlap']:>10.0%} {r['low_iou']:>10.0%} {r['matched']:>10.0%} "
                  f"{r['median_best_iou']:>8.2f}")

    if verbose:
        print(f"\nRECALL SIDE - by TRUE object size (iou_thresh={iou_thresh}, {len(t_sizes)} objects)")
        _print("", recall_rows, ["miss", "near miss", "clean"])
        print(f"PRECISION SIDE - by PREDICTED object size ({len(p_sizes)} objects)")
        _print("", precision_rows, ["pure FP", "low IoU", "matched"])
        print("(best-IoU is not one-to-one, so recall-side numbers are slightly optimistic)")

    return {"recall": recall_rows, "precision": precision_rows, "bins": bins}

@torch.no_grad #Decorator which makes whole function not track gradients
def collect_object_diagnostics(model, dataloader, prepare_batch, max_batches=None,
                               iou_thresh=0.5, min_pred_size=4, device="cuda"):

    # Run a model over a dataloader and gather diagnose_object_matches() results.
    #
    # prepare_batch: callable(scene) -> (x, y, nodata) or None to skip a batch.
    #     x: model-ready input (already normalized / indices appended), CPU or GPU
    #     y: (B, H, W) labels in {0, 1} (nodata already clamped to 0)
    #     nodata: (B, H, W) bool, True where pixels are invalid
    #
    model.eval()
    results = []
    for b, scene in enumerate(dataloader):
        if max_batches is not None and b >= max_batches:
            break
        prepared = prepare_batch(scene)
        if prepared is None:
            continue
        x, y, nodata = prepared
        x2 = torch.cat((x[:, :5, :, :], x[:, 6:, :, :]), dim=1)
        outputs = model(x2.to(device))
        results.extend(diagnose_object_matches(
            outputs, y.to(device), nodata=nodata.to(device),
            iou_thresh=iou_thresh, min_pred_size=min_pred_size))
    return results


def singleclass_train(model, train_dl, valid_dl, loss_fn, optimizer, epochs=5, batches=1, batchsize=1,
           scheduler=None, modID="defaultID", stats=None, thresholds=torch.tensor([0.5]),cname = "SP",patience = 3,tolerance = 0.1,bf = 0.3):
    start = time.time()
    model.float()
    model.cuda()
    best_mIOU = 0
    best_F1 = 0
    best_Prec = 0

    #set up early stopping tracker
    stop_check = early_stop(patience=patience, tolerance=tolerance)
    # set up loss functions
    Tloss_fn = loss_fn['train']
    Vloss_fn = loss_fn['val']

    # stats
    TMax = torch.tensor(stats[0]).float()
    TMin = torch.tensor(stats[1]).float()
    VMax = torch.tensor(stats[0]).float()
    VMin = torch.tensor(stats[1]).float()

    print(TMax,TMin,VMax,VMin)
    # training transforms

    CJ = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=None, hue=None)
    addIndicesT = v2.Compose([  # currently configured for 4 band images

        MinMaxNormalize(min=TMin, max=TMax),
        #AppendNDVI(index_nir=3, index_red=2),

        MSAVI(indexNIR=3, indexRED=2),
        AppendNDWI(index_nir=3, index_green=1),

    ])
    geometric_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=90)
       # v2.RandomAffine(degrees=(30,70),scale=(0.9,1.0))
    ])
    # validation transforms
    addIndicesV = v2.Compose([

        MinMaxNormalize(min=VMin, max=VMax),
        MSAVI(indexNIR=3, indexRED=2),
        AppendNDWI(index_nir=3, index_green=1)
        #AppendNDVI(index_nir=3, index_red=2),


    ])
    # empty vectors for performance tracking
    train_precision, valid_precision = [], []
    train_recall, valid_recall = [], []
    train_F1, valid_F1 = [], []
    num_skipped = 0
    # set up confusion matrix

    # set up TBwriters
    writers = {}
    for phase in ['train', 'valid']:
        writers[phase] = {
            'loss': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_Loss'),
            'CEloss': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_CELoss'),
            'Boundloss': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_BoundLoss'),
            'precision': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_Precision'),
            'recall': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_Recall'),
            'prcurve': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_prcurve'),
            'F1': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_F1'),
            'macroF1': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_macroF1'),
            'microF1': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_microF1'),
            'Threshprecision': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshPrecision'),
            'Threshrecall': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshRecall'),
            'ThreshmacroF1': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshmacroF1'),
            'ThreshmicroF1': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshmicroF1'),

            'ThreshprecisionBG': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshPrecisionBG'),
            'ThreshrecallBG': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshRecallBG'),

            'ThreshprecisionSP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshPrecisionSP'),
            'ThreshrecallSP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshRecallSP'),

            'mIoU': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_mIoU'),
            'IoU_SP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_IoU_SP'),
            'IoU_BG': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_IoU_BG'),
            'DiceSP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_DiceSP'),
            'PConfMat': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_PConfMat'),
            'RConfMat': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_RConfMat'),
            'originalIMG': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_OriginalIMG'),
            'MASK': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_MASK'),
            'PRED': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_PRED'),
            'NoData': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_NoData'),
            'DATA': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_DATA'),
            'THRESH': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_THRESH'),
            'SurfaceDice': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_SurfaceDice'),
            'Hausdorff95': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_Hausdorff95'),
            'ObjF1': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ObjF1'),
            'ObjPrecision': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ObjPrecision'),
            'ObjRecall': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ObjRecall'),
            'ObjFP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ObjFP'),
            'ObjTP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ObjTP'),
            'ObjFN': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ObjFN')
        }

    # set up confusion matrix logger
    ConfMatrix = ignite.metrics.confusion_matrix.ConfusionMatrix(num_classes=2)
    mIOU = ignite.metrics.mIoU(ConfMatrix)
    IOU = ignite.metrics.IoU(ConfMatrix,ignore_index=0)
    dice = ignite.metrics.DiceCoefficient(ConfMatrix,ignore_index=0)
    PConfMatrix = ignite.metrics.confusion_matrix.ConfusionMatrix(num_classes=2, average='precision')
    RConfMatrix = ignite.metrics.confusion_matrix.ConfusionMatrix(num_classes=2, average='recall')
    #boundary and object metrics
    boundary_metrics = BoundaryMetrics(class_thresholds=[3.0])  # tolerance in px
    object_metrics = ObjectMetrics(iou_thresh=0.3)
    om_test = ObjectMetrics(iou_thresh=0.3)

    # begin train loop
    for epoch in range(epochs):
        print(bcolors.WARNING + 'Epoch {}/{}'.format(epoch, epochs - 1))
        print(bcolors.ENDC)
        print('-' * 10)
        epoch_loss = 0  # for purposes of avoiding errors

        #update loss epoch trackers for ramping
        for fn in (Tloss_fn, Vloss_fn):
            if hasattr(fn, "set_epoch"):
                fn.set_epoch(epoch)

        for phase in ['train', 'valid']:
            if phase == 'train':
                dataloader = train_dl
            else:
                dataloader = valid_dl
            # reset metrics matrices
            dice.reset()
            mIOU.reset()
            IOU.reset()
            ConfMatrix.reset()
            PConfMatrix.reset()
            RConfMatrix.reset()
            boundary_metrics.reset()
            object_metrics.reset()
            running_loss = 0.0
            running_ce = 0.0
            running_boundloss = 0.0

            step = 0
            num_skipped = 0
            allpreds = []
            allprobs = []
            allTpreds = []
            allTpredsBG = []
            allTpredsSP = []
            alltargets = []
            allthreshtargs = []
            alltargetsBG = []
            alltargetsSP = []

            all_results = []  # OBJ MATCH

            # iterate over data
            for batchID, scene in enumerate(dataloader):


                step += 1

                mask = scene["mask"].clamp(min=-1) #set all negatives to -1
                if cname == "WP":
                    mask[mask != 2] = 0 #ZERO OUT ALL NON WP ANNOTATIONS
                    mask[mask == 2] = 1 #CHANGE WP TO 1
                if cname == "SP":
                    mask[mask > 1] = 0 #zero out non SP annotations
                # forward pass & save image for tensorboard
                img = scene["image"][:, [2, 1, 0], :, :]
                x_prime = scene["image"][:, 0:3, :, :]

                # remove truth values where batch images are [0,0,0] in rgb + NIR
                zeros = (x_prime <= 0).all(dim=1)
                mask[zeros] = -1
                scene["image"][:, 4, :, :][
                    zeros] = -1  # mask out any elevation/NWI data that overlaps with blank/no data in imgs
                scene["image"][:, 5, :, :][
                    zeros] = -1
                # print(img.shape)
                nodata = mask == -1

                if torch.sum(nodata).item() >= 60000*batchsize: #THIS NUMBER IS BATCHSIZE  + INPUT SIZE DEPENDENT!
                    print(bcolors.FAIL + "OVER NO DATA THRESHOLD - SKIPPED")
                    print("NoData sum: ", torch.sum(nodata).item())
                    print(bcolors.ENDC)
                    num_skipped += 1
                    continue

                # Calculate percentage of positive class
                valid_pixels = mask != -1
                if valid_pixels.sum() == 0:
                    print(bcolors.FAIL + "EMPTY IMG - SKIPPED")
                    print(bcolors.ENDC)
                    num_skipped += 1
                    continue

                positive_ratio = (mask[valid_pixels] >= 1).float().mean()

                # Skip if <5%  positive
                if positive_ratio < 0.005:
                    num_skipped += 1
                    print(f"Skipping - {positive_ratio:.1%} positive pixels")
                    continue

                if phase == 'train':
                    model.train(True)  # Set training mode = true
                    # zero the gradients
                    optimizer.zero_grad()

                    # print(y.shape)
                    # print("trainmask",y.min(),y.max())
                    # print(x.shape)
                    x1 = scene["image"]
                    x1 = torchvision.tv_tensors.Image(x1)
                    y1 = torchvision.tv_tensors.Mask(mask)
                    x, y = geometric_transforms(x1, y1)

                    #add colorjitter separate
                    rand_bright = torch.empty(1).uniform_((1-bf),(1+bf)).item()
                    x_spectral = x[:, 0:4, :, :] * rand_bright
                    x_cj = torch.cat([x_spectral, x[:, 4:, :, :]], dim=1)

                    x2 = addIndicesT(x_cj)

                    y = y.clamp(min=0)


                    ##print(x.shape)
                    timg = x2[:, [2, 1, 0], :, :]
                    tmask = y
                    elev = x2[:, 4, :, :]
                   # ndvi = x[:, 5, :, :]
                    ndwi = x2[:, 6, :, :]
                    msavi = x2[:, 5, :, :]

                    # convert the 4D tensors to 3D (Batch size, X, Y)
                    # print(y.sum())
                    #no elevation
                    nux = torch.cat((x2[:,:4,:,:],x2[:,5:,:,:]),dim=1)
                    #no raw Green/Blue bands
                    nux2 = nux[:,2:,:,:]
                    #no NWI mask
                    nwix = torch.cat((x2[:,:5,:,:],x2[:,6:,:,:]),dim=1)


                    outputs = model(nwix.cuda()) #CUT OUT NWI MANUALLY FOR NOW
                    #
                    # y1h = torch.nn.functional.one_hot(y, num_classes=2)
                    # yperm = y1h.permute(0,3,1,2)


                    loss = Tloss_fn(outputs, y.cuda())
                    ce_train = getattr(Tloss_fn, "last", {}).get("ce", loss.item())
                    running_ce += ce_train
                    bound_train = getattr(Tloss_fn, "last", {}).get("boundary", loss.item())
                    running_boundloss += bound_train
                    running_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    y = y.cuda()
                    dice.update((outputs, y))
                    mIOU.update((outputs, y))
                    IOU.update((outputs, y))
                    PConfMatrix.update((outputs, y))
                    RConfMatrix.update((outputs, y))
                else:

                    with torch.no_grad():
                        model.train(False)  # Set model to evaluate mode
 #
                        x = addIndicesV(scene["image"])


                        x_prime = scene["image"][:, 0:3, :, :]
                        zeros = (x_prime <= 0).all(dim=1)
                        mask[zeros] = -1
                        scene["image"][:, 4, :, :][zeros] = -1

                        y = mask.clamp(min=0)
                        
                        timg = x[:, [2, 1, 0], :, :]
                        tmask= y
                        elev = x[:, 4, :, :]
                        #ndvi = x[:, 5, :, :]
                        ndwi = x[:, 6, :, :]
                        msavi = x[:, 5, :, :]
                        #no elevation
                        nux = torch.cat((x[:, :4, :, :], x[:, 5:, :, :]),dim=1)
                        nux2 = nux[:, 2:, :, :]
                        # no NWI mask
                        nwix = torch.cat((x[:, :5, :, :], x[:, 6:, :, :]), dim=1)

                        outputs = model(nwix.cuda())

                        # y1h = torch.nn.functional.one_hot(y, num_classes=2)
                        #
                        # yperm = y1h.permute(0, 3, 1, 2)
                        #CHANGED TO SHARE LOSS FN WITH TRAIN
                        loss = Tloss_fn(outputs, y.cuda())  # ['out']
                        ce_val = getattr(Tloss_fn, "last", {}).get("ce", loss.item())
                        running_ce += ce_val
                        bound_val = getattr(Tloss_fn, "last", {}).get("boundary", loss.item())
                        running_boundloss += bound_val
                        running_loss += loss.item()  # * batch_size

                        #TROUBLESHOOT OBJMATCH
                        if batchID<=10:
                            all_results.extend(diagnose_object_matches(outputs, y.cuda(), nodata=nodata))

                        y=y.cuda()
                        dice.update((outputs, y))
                        mIOU.update((outputs, y))
                        IOU.update((outputs, y))
                        PConfMatrix.update((outputs, y))
                        RConfMatrix.update((outputs, y))
                        boundary_metrics.update(outputs,y,nodata = nodata)
                        object_metrics.update(outputs,y,nodata = nodata)
                        om_test.update(outputs=F.one_hot(y, 2).permute(0,3,1,2).float() * 100,y=y,nodata=nodata)

                # stats - unroll predictions to 1D tensor
                y2 = torch.reshape(y, (-1,))
                # print(y2.shape)


                # print(y.shape)
                pred = outputs.softmax(dim=1).argmax(dim=1).detach().cpu()  #
                # print(pred.shape)
                pred = torch.reshape(pred, (-1,))
                # print(pred.shape)
                probs = outputs.softmax(dim=1)
                probsBG, probsSP = probs[:, 0, :, :], probs[:, 1, :, :]
                threshpredsBG = (probsBG >= thresholds)
                threshpredsSP = (probsSP >= thresholds)
                threshpreds = torch.cat([threshpredsBG, threshpredsSP], dim=1)
                # print("ThreshPReds",threshpredsBG.shape)

                allTpredsBG.append(threshpredsBG.reshape(-1, ).cpu())
                allTpredsSP.append(threshpredsSP.reshape(-1, ).cpu())

                #convert preds to binaries for each class
                targsBG = (y == 0)
                targsSP = (y == 1)
                threshtargs = torch.cat([targsBG, targsSP], dim=1)
                # print("TARGS",targsWP.shape)

                alltargetsBG.append(targsBG.reshape(-1, ).cpu())
                alltargetsSP.append(targsSP.reshape(-1, ).cpu())

                allTpreds.append(threshpreds.reshape(-1, ).cpu())
                allthreshtargs.append(threshtargs.reshape(-1, ).cpu())
                allpreds.append(pred)

                probsSP_ele = torch.reshape(probsSP,(-1,)).detach().cpu()

                allprobs.append(probsSP_ele)

                # check tHAT PREDs are valid
                assert not torch.isnan(pred).any()
                assert not torch.isinf(pred).any()
                # print("Targs:",targsBG.shape)
                y2 = y2.detach().cpu()

                alltargets.append(y2)

            if alltargets:

                all_targs_tensor,all_probs_tensor = torch.cat(alltargets), torch.cat(allprobs)
                #find invalid data
                validmask = all_targs_tensor != -1
                #remove
                filtertargs = all_targs_tensor[validmask]
                filterprobs = all_probs_tensor[validmask]

                epoch_loss = running_loss / (step - num_skipped + 0.0000000001)
                epoch_lossCE = running_ce / (step - num_skipped + 0.0000000001)
                epoch_lossBound = running_boundloss / (step - num_skipped + 0.0000000001)
                epoch_prec = precision_score(filtertargs, torch.cat(allpreds)[validmask], average='binary',
                                             labels=[0, 1])
                epoch_rec = recall_score(filtertargs, torch.cat(allpreds)[validmask], average='binary',
                                         labels=[0, 1])
                epoch_F1 = f1_score(filtertargs,torch.cat(allpreds)[validmask], average='binary',
                                    labels = [0, 1])
                epoch_F1micro = f1_score(filtertargs, torch.cat(allpreds)[validmask], average='micro',
                                         labels=[0, 1])
                epoch_F1macro = f1_score(filtertargs, torch.cat(allpreds)[validmask], average='macro',
                                         labels=[0, 1])

                epochT_prec = precision_score(torch.cat(allthreshtargs), torch.cat(allTpreds), average='binary',
                                              labels=[0, 1])
                epochT_rec = recall_score(torch.cat(allthreshtargs), torch.cat(allTpreds), average='binary',
                                          labels=[0, 1])
                epochT_F1micro = f1_score(torch.cat(allthreshtargs), torch.cat(allTpreds), average='binary',
                                          labels=[0, 1])
                epochT_F1macro = f1_score(torch.cat(allthreshtargs), torch.cat(allTpreds), average='binary',
                                          labels=[0, 1])

                # threshold messy metrics
                epochT_precBG = precision_score(torch.cat(alltargetsBG), torch.cat(allTpredsBG), average='binary',
                                                labels=[0, 1])
                epochT_recBG = recall_score(torch.cat(alltargetsBG), torch.cat(allTpredsBG), average='binary',
                                            labels=[0, 1])

                epochT_precSP = precision_score(torch.cat(alltargetsSP), torch.cat(allTpredsSP), average='binary',
                                                labels=[0, 1])
                epochT_recSP = recall_score(torch.cat(alltargetsSP), torch.cat(allTpredsSP), average='binary',
                                            labels=[0, 1])

            if epoch % 2 == 0 and epoch_loss != 0:
                print(
                    bcolors.BOLD + 'Current epoch: {}  Loss: {}   Precision: {} Recall: {} F1: {} AllocMem (Mb): {}'.format(
                        epoch, epoch_loss, epoch_prec, epoch_rec, epoch_F1,
                        torch.cuda.memory_allocated() / 1024 / 1024))
                print(bcolors.ENDC)
            if epoch_loss != 0:
                writers[phase]['loss'].add_scalar("Loss/{}".format(phase), epoch_loss, epoch)
                writers[phase]['CEloss'].add_scalar("CELoss/{}".format(phase), epoch_lossCE, epoch)
                writers[phase]['Boundloss'].add_scalar("BoundLoss/{}".format(phase), epoch_lossBound, epoch)
                writers[phase]['precision'].add_scalar("Precision/{}".format(phase), epoch_prec, epoch)
                writers[phase]['recall'].add_scalar("Recall/{}".format(phase), epoch_rec, epoch)
                writers[phase]['prcurve'].add_pr_curve("prcurve/{}".format(phase), labels=filtertargs, predictions=filterprobs
                                                        ,global_step=epoch)
                #writers[phase]['prcurve'].close()
                #writers[phase]['PR_Curve'].add_pr_curve("PR_Curve/Thresh_{}".format(phase), labels=torch.cat(allthreshtargs), predictions=torch.cat(allTpreds),global_step=epoch)
                writers[phase]['F1'].add_scalar("F1/{}".format(phase), epoch_F1, epoch)
                writers[phase]['microF1'].add_scalar("F1/{}".format(phase), epoch_F1micro, epoch)
                writers[phase]['macroF1'].add_scalar("F1/{}".format(phase), epoch_F1macro, epoch)

                # #threshold
                writers[phase]['Threshprecision'].add_scalar("ThreshPrecision/{}".format(phase), epochT_prec, epoch)
                writers[phase]['Threshrecall'].add_scalar("ThreshRecall/{}".format(phase), epochT_rec, epoch)
                writers[phase]['ThreshmicroF1'].add_scalar("ThreshF1/{}".format(phase), epochT_F1micro, epoch)
                writers[phase]['ThreshmacroF1'].add_scalar("ThreshF1/{}".format(phase), epochT_F1macro, epoch)

                writers[phase]['ThreshprecisionBG'].add_scalar("ThreshPrecisionBG/{}".format(phase), epochT_precBG,
                                                               epoch)
                writers[phase]['ThreshrecallBG'].add_scalar("ThreshRecallBG/{}".format(phase), epochT_recBG, epoch)

                writers[phase]['ThreshprecisionSP'].add_scalar("ThreshPrecisionSP/{}".format(phase), epochT_precSP,
                                                               epoch)
                writers[phase]['ThreshrecallSP'].add_scalar("ThreshRecallSP/{}".format(phase), epochT_recSP, epoch)


                writers[phase]['mIoU'].add_scalar("mIoU/{}".format(phase), mIOU.compute(), epoch)
                writers[phase]['IoU_SP'].add_scalar("IoU/{}".format(phase), IOU.compute()[0], epoch)
                writers[phase]['DiceSP'].add_scalar("Dice/{}".format(phase), dice.compute()[0], epoch)

            if phase == "valid" and epoch <= 100:
                # OBJ MATCH TROUBLESHOOT
                total_misses = sum(r["total_misses"] for r in all_results)
                near_misses = sum(r["near_misses"] for r in all_results)
                clean = sum(r["clean_matches"] for r in all_results)
                avg_n_pred = np.mean([r["n_pred_components"] for r in all_results])
                avg_n_targ = np.mean([r["n_targ_components"] for r in all_results])
                best = np.concatenate([r["best_iou_per_target"] for r in all_results])
                print(np.histogram(best, bins=[0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]))
                sizes = np.concatenate([r["target_sizes"] for r in all_results])

                bins = [0, 20, 100, 500, np.inf]  # pixel areas - adjust to your pan size distribution
                for lo, hi in zip(bins[:-1], bins[1:]):
                    m = (sizes >= lo) & (sizes < hi)
                    if m.sum() == 0:
                        continue
                    print(f"{lo:>4}-{hi:<5} px | n={m.sum():4d} | miss={np.mean(best[m] == 0):.0%} "
                          f"| near={np.mean((best[m] > 0) & (best[m] < 0.5)):.0%} | clean={np.mean(best[m] >= 0.5):.0%}")

                print(f"total misses: {total_misses}, near misses: {near_misses}, clean: {clean}")
                print(
                    f"avg predicted components/tile: {avg_n_pred:.1f}, avg true components/tile: {avg_n_targ:.1f}")

                if phase == 'valid': #log bound/obj accuracies
                    bm = boundary_metrics.compute()
                    om = object_metrics.compute()
                    omtest = om_test.compute()
                    print(bcolors.OKGREEN + "ObjMatch Sanity Check value: {}".format(omtest))
                    print(bcolors.ENDC)
                    writers[phase]['SurfaceDice'].add_scalar("SurfaceDice/{}".format(phase), bm['surface_dice'], epoch)
                    writers[phase]['Hausdorff95'].add_scalar("Hausdorff95/{}".format(phase), bm['hausdorff95'], epoch)
                    writers[phase]['ObjF1'].add_scalar("ObjF1/{}".format(phase), om['obj_f1'], epoch)
                    writers[phase]['ObjPrecision'].add_scalar("ObjPrecision/{}".format(phase), om['obj_precision'],epoch)
                    writers[phase]['ObjRecall'].add_scalar("ObjRecall/{}".format(phase), om['obj_recall'], epoch)
                    writers[phase]['ObjTP'].add_scalar("ObjCounts/{}_TP".format(phase), om['obj_tp'], epoch)
                    writers[phase]['ObjFP'].add_scalar("ObjCounts/{}_FP".format(phase), om['obj_fp'], epoch)
                    writers[phase]['ObjFN'].add_scalar("ObjCounts/{}_FN".format(phase), om['obj_fn'], epoch)

                pCM = plot_confusion_matrix(PConfMatrix.compute(), ["BG", "SP"])  # ["water","land"]
                rCM = plot_confusion_matrix(RConfMatrix.compute(), ["BG", "SP"])  # ["water","land"]
                writers[phase]['PConfMat'].add_figure("prec_confusion_matrix/{}".format(phase), pCM, epoch)
                writers[phase]['RConfMat'].add_figure("rec_confusion_matrix/{}".format(phase), rCM, epoch)
                writers[phase]['NoData'].add_scalar("NumSkipped/{}".format(phase), num_skipped, global_step=epoch)
            # capture images every 5 epochs
            if epoch % 5 == 0 and epoch_loss != 0:
                img_grid = make_grid(img[0:8,:,:,:], nrow=4, normalize=True, scale_each=True, value_range=(0, 0.1))

                transimg_grid = make_grid(timg[0:8,:,:,:], nrow=4, normalize=True, scale_each=True, value_range=(0, 1))
                writers[phase]['originalIMG'].add_image('Original Images/{}'.format(epoch), img_grid, global_step=epoch)
                #print('ndvi: ', ndvi.shape, 'ndwi: ', ndwi.shape)  # 'elev:', elev.shape,
                elev_grid = make_grid(elev[0:8,:,:].unsqueeze(1).float(), nrow=4, normalize=True, value_range=(0, 1))
                #ndvi_grid = make_grid(ndvi.unsqueeze(1).float(), nrow=4, normalize=True, value_range=(-1, 1))
                ndwi_grid = make_grid(ndwi[0:8,:,:].unsqueeze(1).float(), nrow=4, normalize=True, value_range=(-1, 1))
                msavi_grid = make_grid(msavi[0:8,:,:].unsqueeze(1).float(), nrow=4, normalize=True, value_range=(-1, 1))

                tmask_color = apply_colormap_to_mask(tmask.cpu(), num_classes=2)
                mask_color = apply_colormap_to_mask(mask.cpu(), num_classes=2)
                mask_grid = make_grid(mask_color[0:8,:,:,:], nrow=4, normalize=False)  # ,scale_each=True
                tmask_grid = make_grid(tmask_color[0:8,:,:,:], nrow=4, normalize=False)  # ,scale_each=True

                writers[phase]['MASK'].add_image('Masks/{}'.format(epoch), mask_grid, global_step=epoch)
                print("out {} | pred {}".format(outputs.argmax(dim=1).detach().cpu().shape, pred.shape))
                inpred = outputs.argmax(dim=1).detach().cpu().unsqueeze(1).float()
                pred_color = apply_colormap_to_mask(inpred[0:8,:,:,:], num_classes=2)
                print("img {}   mask {}     pred {}".format(img.shape, mask.shape, inpred.shape))

                pred_grid = make_grid(pred_color, nrow=4, normalize=True,scale_each=True)  # ,scale_each=True
                # threshold vis
                TpredBG_grid = make_grid(threshpredsBG[0:8,:,:].unsqueeze(1).float(), nrow=4, normalize=False)
                TpredSP_grid = make_grid(threshpredsSP[0:8,:,:].unsqueeze(1).float(), nrow=4, normalize=False)

                TtargBG_grid = make_grid(targsBG[0:8,:,:].unsqueeze(1).float(), nrow=4, normalize=False)
                TtargSP_grid = make_grid(targsSP[0:8,:,:].unsqueeze(1).float(), nrow=4, normalize=False)
                allthresh = torch.cat(
                    [TpredBG_grid, TtargBG_grid, TpredSP_grid, TtargSP_grid], dim=1)


                all = torch.cat([transimg_grid, tmask_grid, pred_grid], dim=1)
                data = torch.cat([elev_grid,ndwi_grid,msavi_grid], dim=1)  #ndvi_grid,
                print("data", data.shape)
                print("all", all.shape)
                writers[phase]['PRED'].add_image('Preds/{}_{}'.format(phase, epoch), all,
                                                 global_step=epoch)  # FIGURE THIS OUT
                writers[phase]['DATA'].add_image('Data/{}_{}'.format(phase, epoch), data,
                                                 global_step=epoch)  # FIGURE THIS OUT
                writers[phase]['THRESH'].add_image('THRESH/{}_{}'.format(phase, epoch), allthresh, global_step=epoch)

            if epoch_loss != 0:
                print(
                    bcolors.OKBLUE + '{} epoch {} |  Loss: {:.4f}  |  epoch Precision: {} | epoch Recall: {} | epoch F1: {}'.format(
                        phase, epoch, epoch_loss, epoch_prec, epoch_rec, epoch_F1))
                print(bcolors.ENDC)

                train_precision.append(epoch_prec) if phase == 'train' else valid_precision.append(epoch_prec)
                train_recall.append(epoch_rec) if phase == 'train' else valid_recall.append(epoch_rec)
                # attempt to exclude nan values while doing optuna optimization
                if phase == 'train' and np.isnan(epoch_F1micro) == False:
                    train_F1.append(epoch_F1micro)
                if phase == 'valid' and np.isnan(epoch_F1micro) == False:
                    valid_F1.append(epoch_F1micro)

                # save best model
                if phase == 'valid':
                    if mIOU.compute() > best_mIOU:
                        best_mIOU = mIOU.compute()
                        print(bcolors.OKGREEN + "New best model mIoU: {}".format(best_mIOU))
                        print(bcolors.ENDC)
                        torch.save(model.state_dict(), './saved_models/{}_bestmIOU_{}_epoch{}.pth'.format(modID,best_mIOU.round(decimals=2),epoch))  # %%
                        savedpCM_mIOU = PConfMatrix.compute()
                        savedrCM_mIOU = RConfMatrix.compute()
                        print(savedpCM_mIOU,savedrCM_mIOU)
                    if  epoch_F1 > best_F1:
                        best_F1 = epoch_F1
                        print(bcolors.OKGREEN + "New best model saltpan F1: {}".format(best_F1))
                        print(bcolors.ENDC)
                        torch.save(model.state_dict(), './saved_models/{}_bestF1_{}_epoch{}.pth'.format(modID,round(best_F1,2),epoch))  # %%
                        savedpCM_F1 = PConfMatrix.compute()
                        savedrCM_F1 = RConfMatrix.compute()
                        print(savedpCM_F1,savedrCM_F1)
                    if epoch_prec > best_Prec:
                        best_Prec = epoch_prec
                        print(bcolors.OKGREEN + "New best model saltpan Precision: {}".format(best_Prec))
                        print(bcolors.ENDC)
                        torch.save(model.state_dict(), './saved_models/{}_bestPrec_{}_epoch{}.pth'.format(modID,round(best_Prec,2),epoch))  # %%
                        savedpCM_Prec = PConfMatrix.compute()
                        savedrCM_Prec = RConfMatrix.compute()
                        print(savedpCM_Prec,savedrCM_Prec)


            if phase == 'valid':
                if stop_check.run(val_loss = epoch_lossCE):
                    print("VALIDATION LOSS INCREASING - STOPPING EARLY!")
                    break
        if scheduler:
            scheduler.step()
            print("current LR: {}".format(scheduler.get_last_lr()))

    # average F1 scores
    train20 = train_F1[-20:-1]
    print(train20)  # this grabs UP TO the last 20 F1 scores
    valid20 = valid_F1[-20:-1]
    train_F1 = sum(train20) / (len(train20) + 0.01000)
    valid_F1 = sum(valid20) / (len(valid20) + 0.00001)
    # print finishing info
    time_elapsed = time.time() - start
    print(bcolors.OKGREEN + 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(bcolors.OKGREEN + 'Number of empty batches skipped: {}'.format(num_skipped))
    print(bcolors.OKGREEN + "Best model: {}".format(best_mIOU))
    print(bcolors.ENDC)
    print("Best mIOU model CM: \n", savedpCM_mIOU)
    print("Best F1 model CM: \n", savedpCM_F1)

    for phase in ["train", "valid"]:
        w = writers[phase]
        for writer in w:
            w[writer].close()

    return train_precision, valid_precision, train_recall, valid_recall, train_F1, valid_F1, model, best_mIOU


def multiclass_train(model, train_dl,valid_dl,  loss_fn, optimizer, epochs=5,batches=1,batchsize = 1,
          modname = None,scheduler = None,modID = "defaultID", stats = None,thresholds = torch.tensor([0.5])):

    start = time.time()
    model.float()
    model.cuda()
    best_mIOU = 0

    #set up loss functions
    Tloss_fn = loss_fn['train']
    Vloss_fn = loss_fn['val']

    #stats
    TMax = torch.tensor(stats[0]).float()
    TMin = torch.tensor(stats[1]).float()
    VMax = torch.tensor(stats[2]).float()
    VMin = torch.tensor(stats[3]).float()
#training transforms

    CJ = v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=None)
    addIndicesT = v2.Compose([ # currently configured for 8 band images

        MinMaxNormalize(min = TMin,max=TMax),
        AppendNDVI(index_nir=3, index_red=2),
        AppendNDWI(index_nir=3, index_green=1)
    ])
    geometric_transforms = v2.Compose([
                     v2.RandomHorizontalFlip(p=0.5),
                     v2.RandomVerticalFlip(p=0.5),
                     v2.RandomRotation(degrees=90)
                 ])
#validation transforms
    addIndicesV = v2.Compose([

        MinMaxNormalize(min=VMin, max=VMax),
        AppendNDVI(index_nir=3, index_red=2),
        AppendNDWI(index_nir=3, index_green=1)
    ])
    #empty vectors for performance tracking
    train_precision, valid_precision = [], []
    train_recall, valid_recall = [], []
    train_F1, valid_F1 = [], []
    num_skipped = 0
    #set up confusion matrix

#set up TBwriters
    writers = {}
    for phase in ['train', 'valid']:
        writers[phase] = {
            'loss': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_Loss'),
            'precision': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_Precision'),
            'recall': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_Recall'),
            'macroF1': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_macroF1'),
            'microF1': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_microF1'),
            'Threshprecision': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshPrecision'),
            'Threshrecall': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshRecall'),
            'ThreshmacroF1': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshmacroF1'),
            'ThreshmicroF1': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshmicroF1'),

            'ThreshprecisionBG': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshPrecisionBG'),
            'ThreshrecallBG': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshRecallBG'),

            'ThreshprecisionSP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshPrecisionSP'),
            'ThreshrecallSP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshRecallSP'),

            'ThreshprecisionDB': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshPrecisionDB'),
            'ThreshrecallDB': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshRecallDB'),

            'ThreshprecisionWP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshPrecisionWP'),
            'ThreshrecallWP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_ThreshRecallWP'),
            
            
            'mIoU': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_mIoU'),
            'IoU_SP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_IoU_SP'),
            'IoU_DB': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_IoU_DB'),
            'IoU_WP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_IoU_WP'),
            'DiceSP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_DiceSP'),
            'DiceDB': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_DiceDB'),
            'DiceWP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_DiceWP'),
            'PConfMat': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_PConfMat'),
            'RConfMat': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_RConfMat'),
            'originalIMG': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_OriginalIMG'),
            'MASK': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_MASK'),
            'PRED': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_PRED'),
            'NoData': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_NoData'),
            'DATA': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_DATA'),
            'THRESH': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_THRESH')
        }

#set up confusion matrix logger
    ConfMatrix = ignite.metrics.confusion_matrix.ConfusionMatrix(num_classes=4)
    mIOU = ignite.metrics.mIoU(ConfMatrix,ignore_index=-1)
    IOU = ignite.metrics.IoU(ConfMatrix,ignore_index=-1)
    dice = ignite.metrics.DiceCoefficient(ConfMatrix,ignore_index=-1)
    PConfMatrix = ignite.metrics.confusion_matrix.ConfusionMatrix(num_classes=4,average='precision')
    RConfMatrix = ignite.metrics.confusion_matrix.ConfusionMatrix(num_classes=4, average='recall')

#begin train loop
    for epoch in range(epochs):
        print(bcolors.WARNING + 'Epoch {}/{}'.format(epoch, epochs - 1))
        print(bcolors.ENDC)
        print('-' * 10)
        epoch_loss = 0 # for purposes of avoiding errors


        for phase in ['train', 'valid']:
            if phase == 'train':
                dataloader = train_dl
            else:
                dataloader = valid_dl
            #reset metrics matrices
            dice.reset()
            mIOU.reset()
            IOU.reset()
            ConfMatrix.reset()
            PConfMatrix.reset()
            RConfMatrix.reset()
            running_loss = 0.0

            step = 0
            num_skipped = 0
            allpreds = []
            allTpreds = []
            allTpredsBG = []
            allTpredsSP = []
            allTpredsDB = []
            allTpredsWP = []
            alltargets = []
            allthreshtargs = []
            alltargetsBG = []
            alltargetsSP = []
            alltargetsDB = []
            alltargetsWP = []

            # iterate over data
            for batchID, scene in enumerate(dataloader):

                step += 1

                mask = scene["mask"].clamp(min = -1)
                # forward pass & save image for tensorboard
                img = scene["image"][:, [2, 1, 0], :, :]
                x_prime = scene["image"][:, [2, 1, 0, 3], :, :]

                # remove truth values where batch images are [0,0,0] in rgb
                zeros = (x_prime <= 0).all(dim=1)
                mask[zeros] = -1
                scene["image"][:,4,:,:][zeros] = -1 #mask out any elevation data that overlaps with blank/no data in imgs
                #print(img.shape)
                nodata = mask == -1

                if torch.sum(nodata).item() >= 60000:
                    print(bcolors.FAIL +"OVER NO DATA THRESHOLD - SKIPPED")
                    print("NoData sum: ", torch.sum(nodata).item())
                    print(bcolors.ENDC)
                    num_skipped +=1
                    continue

                # Calculate percentage of positive class
                valid_pixels = mask != -1
                if valid_pixels.sum() == 0:
                    print(bcolors.FAIL + "EMPTY IMG - SKIPPED")
                    print(bcolors.ENDC)
                    num_skipped += 1
                    continue

                positive_ratio = (mask[valid_pixels] >= 1).float().mean()

                # Skip if <5%  positive (adjust thresholds as needed)
                if positive_ratio < 0.005:
                    num_skipped += 1
                    print(f"Skipping - {positive_ratio:.1%} positive pixels")
                    continue

                if phase == 'train':
                    model.train(True)  # Set training mode = true
                    # zero the gradients
                    optimizer.zero_grad()


                    #print(y.shape)
                    #print("trainmask",y.min(),y.max())
                    #print(x.shape)
                    x1 = scene["image"]
                    x1 = torchvision.tv_tensors.Image(x1)
                    y1 = torchvision.tv_tensors.Mask(mask)
                    x2,y = geometric_transforms(x1,y1)

                    x = addIndicesT(x2)


                    ##print(x.shape)
                    timg = x[:,[2,1,0],:,:]
                    tmask = y
                    elev = x[:,4,:,:]
                    ndvi = x[:,5,:,:]
                    ndwi = x[:,6,:,:]


                    # convert the 4D tensors to 3D (Batch size, X, Y)
                    # print(y.sum())
                    y = y.cuda()
                    x = x.cuda()

                    outputs = model(x)

                    loss = Tloss_fn(outputs, y)
                    loss.backward()
                    optimizer.step()
                    dice.update((outputs, y))
                    mIOU.update((outputs, y))
                    IOU.update((outputs, y))
                    PConfMatrix.update((outputs,y))
                    RConfMatrix.update((outputs, y))
                    running_loss += loss.item()
                else:
                    with torch.no_grad():
                        model.train(False)  # Set model to evaluate mode
                        y = mask.cuda()
                        x = addIndicesV(scene["image"])
                        timg = x[:, [2, 1, 0], :, :]
                        elev = x[:, 4, :, :]
                        ndvi = x[:, 5, :, :]
                        ndwi = x[:, 6, :, :]
                        x = x.cuda()
                        outputs = model(x)

                        #print(outputs.shape)
                        loss = Vloss_fn(outputs, y) #['out']
                        running_loss += loss.item()  # * batch_size
                        dice.update((outputs, y))
                        mIOU.update((outputs, y))
                        IOU.update((outputs, y))
                        PConfMatrix.update((outputs,y))
                        RConfMatrix.update((outputs, y))


                # stats - unroll predictions to 1D tensor
                y2 = torch.reshape(y, (-1,))
                # print(y2.shape)

                #print(y.shape)
                pred = outputs.softmax(dim=1).argmax(dim=1).detach().cpu()#
                #print(pred.shape)
                pred = torch.reshape(pred,(-1,))
                #print(pred.shape)
                probs = outputs.softmax(dim=1)
                probsBG, probsSP, probsDB, probsWP = probs[:,0,:,:],probs[:,1,:,:],probs[:,2,:,:],probs[:,3,:,:]
                threshpredsBG = (probsBG >= thresholds)
                threshpredsSP = (probsSP >= thresholds)
                threshpredsDB = (probsDB >= thresholds)
                threshpredsWP = (probsWP >= thresholds)
                threshpreds = torch.cat([threshpredsBG,threshpredsSP,threshpredsDB,threshpredsWP],dim=1)
                #print("ThreshPReds",threshpredsBG.shape)

                allTpredsBG.append(threshpredsBG.reshape(-1,).cpu())
                allTpredsSP.append(threshpredsSP.reshape(-1,).cpu())
                allTpredsDB.append(threshpredsDB.reshape(-1,).cpu())
                allTpredsWP.append(threshpredsWP.reshape(-1,).cpu())

                targsBG = (y == 0)
                targsSP = (y == 1)
                targsDB = (y == 2)
                targsWP = (y == 3)
                threshtargs = torch.cat([targsBG,targsSP,targsDB,targsWP],dim=1)
                #print("TARGS",targsWP.shape)
                
                alltargetsBG.append(targsBG.reshape(-1,).cpu())
                alltargetsSP.append(targsSP.reshape(-1,).cpu())
                alltargetsDB.append(targsDB.reshape(-1,).cpu())
                alltargetsWP.append(targsWP.reshape(-1,).cpu())

                allTpreds.append(threshpreds.reshape(-1,).cpu())
                allthreshtargs.append(threshtargs.reshape(-1,).cpu())
                allpreds.append(pred)

                #print("pred shape:",pred.shape,"pred unique:", np.unique(pred.numpy()))
                assert not torch.isnan(pred).any()
                assert not torch.isinf(pred).any()
                #print("Targs:",targsBG.shape)
                y2 = y2.cpu()

                alltargets.append(y2)


            if alltargets:
                epoch_loss = running_loss /(step - num_skipped +0.0000000001)
                epoch_prec = precision_score(torch.cat(alltargets),torch.cat(allpreds),average='micro',labels=[0,1,2,3])
                epoch_rec = recall_score(torch.cat(alltargets),torch.cat(allpreds),average='micro',labels=[0,1,2,3])
                epoch_F1micro = f1_score(torch.cat(alltargets),torch.cat(allpreds),average='micro',labels=[0,1,2,3])
                epoch_F1macro = f1_score(torch.cat(alltargets),torch.cat(allpreds),average='macro',labels=[0,1,2,3])

                epochT_prec = precision_score(torch.cat(allthreshtargs), torch.cat(allTpreds), average='binary',
                                             labels=[0, 1])
                epochT_rec = recall_score(torch.cat(allthreshtargs), torch.cat(allTpreds), average='binary',
                                         labels=[0, 1])
                epochT_F1micro = f1_score(torch.cat(allthreshtargs), torch.cat(allTpreds), average='binary',
                                         labels=[0, 1])
                epochT_F1macro = f1_score(torch.cat(allthreshtargs), torch.cat(allTpreds), average='binary',
                                         labels=[0, 1])
                
                #threshold messy metrics
                epochT_precBG = precision_score(torch.cat(alltargetsBG), torch.cat(allTpredsBG), average='binary',
                                              labels=[0, 1])
                epochT_recBG = recall_score(torch.cat(alltargetsBG), torch.cat(allTpredsBG), average='binary',
                                          labels=[0, 1])

                epochT_precSP = precision_score(torch.cat(alltargetsSP), torch.cat(allTpredsSP), average='binary',
                                              labels=[0, 1])
                epochT_recSP = recall_score(torch.cat(alltargetsSP), torch.cat(allTpredsSP), average='binary',
                                          labels=[0, 1])
                epochT_precDB = precision_score(torch.cat(alltargetsDB), torch.cat(allTpredsDB), average='binary',
                                              labels=[0, 1])
                epochT_recDB = recall_score(torch.cat(alltargetsDB), torch.cat(allTpredsDB), average='binary',
                                          labels=[0, 1])
                epochT_precWP = precision_score(torch.cat(alltargetsWP), torch.cat(allTpredsWP), average='binary',
                                              labels=[0, 1])
                epochT_recWP = recall_score(torch.cat(alltargetsWP), torch.cat(allTpredsWP), average='binary',
                                          labels=[0, 1])

            if epoch % 2 == 0 and epoch_loss != 0:
                print(
                    bcolors.BOLD + 'Current epoch: {}  Loss: {}   Precision: {} Recall: {} F1: {} AllocMem (Mb): {}'.format(
                        epoch, epoch_loss, epoch_prec, epoch_rec, epoch_F1micro, torch.cuda.memory_allocated() / 1024 / 1024))
                print(bcolors.ENDC)
            if epoch_loss != 0:
                writers[phase]['loss'].add_scalar("Loss/{}".format(phase),epoch_loss,epoch)
                writers[phase]['precision'].add_scalar("Precision/{}".format(phase),epoch_prec,epoch)
                writers[phase]['recall'].add_scalar("Recall/{}".format(phase),epoch_rec,epoch)
                writers[phase]['microF1'].add_scalar("F1/{}".format(phase),epoch_F1micro,epoch)
                writers[phase]['macroF1'].add_scalar("F1/{}".format(phase),epoch_F1macro,epoch)

                # #threshold
                writers[phase]['Threshprecision'].add_scalar("ThreshPrecision/{}".format(phase),epochT_prec,epoch)
                writers[phase]['Threshrecall'].add_scalar("ThreshRecall/{}".format(phase),epochT_rec,epoch)
                writers[phase]['ThreshmicroF1'].add_scalar("ThreshF1/{}".format(phase),epochT_F1micro,epoch)
                writers[phase]['ThreshmacroF1'].add_scalar("ThreshF1/{}".format(phase),epochT_F1macro,epoch)

                writers[phase]['ThreshprecisionBG'].add_scalar("ThreshPrecisionBG/{}".format(phase),epochT_precBG,epoch)
                writers[phase]['ThreshrecallBG'].add_scalar("ThreshRecallBG/{}".format(phase),epochT_recBG,epoch)

                writers[phase]['ThreshprecisionSP'].add_scalar("ThreshPrecisionSP/{}".format(phase),epochT_precSP,epoch)
                writers[phase]['ThreshrecallSP'].add_scalar("ThreshRecallSP/{}".format(phase),epochT_recSP,epoch)
                
                writers[phase]['ThreshprecisionDB'].add_scalar("ThreshPrecisionDB/{}".format(phase),epochT_precDB,epoch)
                writers[phase]['ThreshrecallDB'].add_scalar("ThreshRecallDB/{}".format(phase),epochT_recDB,epoch)
                
                writers[phase]['ThreshprecisionWP'].add_scalar("ThreshPrecisionWP/{}".format(phase),epochT_precWP,epoch)
                writers[phase]['ThreshrecallWP'].add_scalar("ThreshRecallWP/{}".format(phase),epochT_recWP,epoch)
                
                writers[phase]['mIoU'].add_scalar("mIoU/{}".format(phase), mIOU.compute(), epoch)
                writers[phase]['IoU_SP'].add_scalar("IoU/{}".format(phase), IOU.compute()[0], epoch)
                writers[phase]['IoU_DB'].add_scalar("IoU/{}".format(phase), IOU.compute()[1], epoch)
                writers[phase]['IoU_WP'].add_scalar("IoU/{}".format(phase), IOU.compute()[2], epoch)
                writers[phase]['DiceSP'].add_scalar("Dice/{}".format(phase), dice.compute()[0], epoch)
                writers[phase]['DiceDB'].add_scalar("Dice/{}".format(phase), dice.compute()[1], epoch)
                writers[phase]['DiceWP'].add_scalar("Dice/{}".format(phase), dice.compute()[2], epoch)


                pCM = plot_confusion_matrix(PConfMatrix.compute(),["BG","SP","DB","WP"])#["water","land"]
                rCM = plot_confusion_matrix(RConfMatrix.compute(), ["BG", "SP", "DB", "WP"])  # ["water","land"]
                writers[phase]['PConfMat'].add_figure("prec_confusion_matrix/{}".format(phase),pCM,epoch)
                writers[phase]['RConfMat'].add_figure("rec_confusion_matrix/{}".format(phase), rCM, epoch)
                writers[phase]['NoData'].add_scalar("NumSkipped/{}".format(phase), num_skipped, global_step=epoch)
            #capture images every 5 epochs
            if epoch % 5 == 0 and epoch_loss != 0:
                img_grid = make_grid(img, nrow=4, normalize=True,scale_each=True,value_range=(0,0.1))

                transimg_grid = make_grid(timg, nrow=4, normalize=True,scale_each=True,value_range=(0,1))
                writers[phase]['originalIMG'].add_image('Original Images/{}'.format(epoch), img_grid, global_step=epoch)
                print('ndvi: ',ndvi.shape,'ndwi: ',ndwi.shape) #'elev:', elev.shape,
                elev_grid = make_grid(elev.unsqueeze(1).float(),nrow=4,normalize=True,value_range=(0,1))
                ndvi_grid = make_grid(ndvi.unsqueeze(1).float(),nrow=4,normalize=True,value_range=(-1,1))
                ndwi_grid = make_grid(ndwi.unsqueeze(1).float(),nrow=4,normalize=True,value_range=(-1,1))

                tmask_color = apply_colormap_to_mask(tmask.cpu(), num_classes=4)
                mask_color = apply_colormap_to_mask(mask.cpu(), num_classes=4)
                mask_grid = make_grid(mask_color, nrow=4,normalize=False) #,scale_each=True
                tmask_grid = make_grid(tmask_color, nrow=4,normalize=False) #,scale_each=True

                writers[phase]['MASK'].add_image('Masks/{}'.format(epoch), mask_grid, global_step=epoch)
                print("out {} | pred {}".format(outputs.argmax(dim=1).detach().cpu().shape,pred.shape))
                inpred = outputs.argmax(dim=1).detach().cpu().unsqueeze(1).float()
                pred_color = apply_colormap_to_mask(inpred, num_classes=4)
                print("img {}   mask {}     pred {}".format(img.shape,mask.shape,inpred.shape))


                pred_grid = make_grid(pred_color, nrow=4,normalize=False)#,scale_each=True
                #threshold vis
                TpredBG_grid = make_grid(threshpredsBG.unsqueeze(1).float(), nrow=4,normalize=False)
                TpredSP_grid = make_grid(threshpredsSP.unsqueeze(1).float(), nrow=4,normalize=False)
                TpredDB_grid = make_grid(threshpredsDB.unsqueeze(1).float(), nrow=4,normalize=False)
                TpredWP_grid = make_grid(threshpredsWP.unsqueeze(1).float(), nrow=4,normalize=False)

                TtargBG_grid = make_grid(targsBG.unsqueeze(1).float(), nrow=4,normalize=False)
                TtargSP_grid = make_grid(targsSP.unsqueeze(1).float(), nrow=4,normalize=False)
                TtargsDB_grid = make_grid(targsDB.unsqueeze(1).float(), nrow=4,normalize=False)
                TtargsWP_grid = make_grid(targsWP.unsqueeze(1).float(), nrow=4,normalize=False)
                allthresh = torch.cat([TpredBG_grid,TtargBG_grid,TpredSP_grid,TtargSP_grid,TpredDB_grid,TtargsDB_grid,TpredWP_grid,TtargsWP_grid], dim=1)

                if phase == 'train':
                    all = torch.cat([transimg_grid, tmask_grid, pred_grid], dim=1)
                else:
                    all = torch.cat([transimg_grid, mask_grid, pred_grid], dim=1)
                data = torch.cat([elev_grid,ndvi_grid, ndwi_grid], dim=1) #
                print("data",data.shape)
                print("all", all.shape)
                writers[phase]['PRED'].add_image('Preds/{}_{}'.format(phase,epoch), all, global_step=epoch)  #FIGURE THIS OUT
                writers[phase]['DATA'].add_image('Data/{}_{}'.format(phase,epoch), data, global_step=epoch)  #FIGURE THIS OUT
                writers[phase]['THRESH'].add_image('THRESH/{}_{}'.format(phase,epoch), allthresh, global_step=epoch)


            if epoch_loss != 0:
                print(bcolors.OKBLUE + '{} epoch {} |  Loss: {:.4f}  |  epoch Precision: {} | epoch Recall: {} | epoch F1: {}'.format(phase,epoch, epoch_loss, epoch_prec,epoch_rec,epoch_F1micro))
                print(bcolors.ENDC)

                train_precision.append(epoch_prec) if phase == 'train' else valid_precision.append(epoch_prec)
                train_recall.append(epoch_rec) if phase == 'train' else valid_recall.append(epoch_rec)
                #attempt to exclude nan values while doing optuna optimization
                if phase == 'train' and np.isnan(epoch_F1micro) == False:
                    train_F1.append(epoch_F1micro)
                if phase == 'valid' and np.isnan(epoch_F1micro) == False:
                    valid_F1.append(epoch_F1micro)

                #save best model
                if phase == 'valid' and mIOU.compute() > best_mIOU:
                    best_mIOU = mIOU.compute()
                    print(bcolors.OKGREEN + "New best model mIoU: {}".format(best_mIOU))
                    print(bcolors.ENDC)
                    torch.save(model.state_dict(),'./saved_models/{}.pth'.format(modID)) #%%
                    savedpCM = PConfMatrix
                    print(savedpCM.compute())


        if scheduler:
            scheduler.step()
            print("current LR: {}".format(scheduler.get_last_lr()))


    # average F1 scores
    train20 = train_F1[-20:-1]
    print(train20) #this grabs UP TO the last 20 F1 scores
    valid20 = valid_F1[-20:-1]
    train_F1 = sum(train20) / (len(train20) + 0.01000)
    valid_F1 = sum(valid20) / (len(valid20) + 0.00001)
#print finishing info
    time_elapsed = time.time() - start
    print(bcolors.OKGREEN + 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(bcolors.OKGREEN + 'Number of empty batches skipped: {}'.format(num_skipped))
    print(bcolors.OKGREEN + "Best model: {}".format(best_mIOU))
    print(bcolors.ENDC)
    print("Best model CM: \n",savedpCM.compute())
    for phase in ["train","valid"]:
        w = writers[phase]
        for writer in w:
            w[writer].close()

    return train_precision, valid_precision, train_recall, valid_recall,train_F1,valid_F1, model,best_mIOU