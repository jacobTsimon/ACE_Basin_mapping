
"""
Script for evaluating trained U-Net model on test dataset
Computes: mIoU, Dice score, F1, Precision, Recall, Confusion Matrix
"""
import sys
sys.path.append("./training")

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomBatchGeoSampler
from torchgeo.datasets import stack_samples
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import ignite.metrics
from tqdm import tqdm
import pandas as pd
from torchvision.transforms import v2
from torchgeo.transforms import AppendNDWI, AppendNDVI
import numpy
import torch.nn as nn


# Import model and dataset classes
from UNet_model import AttU_Net, U_Net
from PanNet_dataset import PlanetScope, PlanetMask, ElevationData, RandomTemporalDataset,NWIData,SlopeData,CreekProxData
from helper_functions import MinMaxNormalize, apply_colormap_to_mask,getstats
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')


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
        out = (NIR2 - numpy.sqrt(NIR2**2 - (8*(NIR-R)))) / 2
        out = torch.from_numpy(out).float()
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"WARNING: MSAVI produced NaN/Inf values!")
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

        return torch.cat([input,out.unsqueeze(dim=1)],dim=1)


def calculate_metrics_per_class(conf_matrix, num_classes, class_names):
    """
    Calculate per-class metrics from confusion matrix

    Args:
        conf_matrix: Confusion matrix [num_classes, num_classes]
        num_classes: Number of classes
        class_names: List of class names

    Returns:
        dict: Per-class metrics
    """
    metrics = {}

    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        TN = conf_matrix.sum() - TP - FP - FN

        if (TP+FP+FN+TN) != conf_matrix.sum():
            print("MISMATCH")
            print(TP,FP,FN,TN)
            print(conf_matrix)
        print("CLASS {}:\n".format(i),"TP:",TP,"FP:", FP,"FN:", FN,"TN:", TN)

        # Precision
        precision = TP / (TP + FP + 1e-8)

        # Recall
        recall = TP / (TP + FN + 1e-8)

        # F1
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        # IoU
        iou = TP / (TP + FP + FN + 1e-8)

        # Dice
        dice = 2 * TP / (2 * TP + FP + FN + 1e-8)

        metrics[class_names[i]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'dice': dice,
            'support': int(TP + FN)  # Number of true samples
        }

    return metrics


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize to percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations with both counts and percentages
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(cm_percent, annot=annot, fmt='', ax=ax, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")
    plt.close()

    return fig


def plot_per_class_metrics(metrics_dict, class_names, save_path='per_class_metrics.png'):
    """Plot per-class metrics as bar chart"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    metric_names = ['precision', 'recall', 'f1', 'iou', 'dice']

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        values = [metrics_dict[cls][metric] for cls in class_names]

        bars = ax.bar(class_names, values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(class_names))))

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)

        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.set_title(f'{metric.upper()} per Class', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

    # Hide extra subplot
    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved per-class metrics to {save_path}")
    plt.close()

    return fig


def evaluate_model(model_path, test_dataloader, device='cuda', num_classes=2,inchan = 7,
                   class_names=['BG', 'SP'], use_msavi=True,use_elev = True,use_GB=True,use_NWI=True, use_normalization=False,
                   stats=None, class_to_evaluate='SP', save_dir='./evaluation_results/',modtype = 'UNET',thresh = 0.5):
    # Evaluate trained model on test dataset
    #
    # Args:
    #     model_path: Path to saved model weights (.pth file)
    #     test_dataloader: DataLoader for test data
    #     device: 'cuda' or 'cpu'
    #     num_classes: Number of output classes
    #     class_names: List of class names
    #     use_msavi: Whether to use MSAVI instead of NDVI
    #     use_normalization: Whether to apply normalization
    #     stats: [TMax, TMin] if using normalization
    #     class_to_evaluate: Which class to focus on for binary classification ('SP' or 'WP')
    #     save_dir: Directory to save results
    #
    # Returns:
    #     dict: Evaluation metrics
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Num classes: {num_classes}")
    print(f"Class names: {class_names}")
    print("=" * 80 + "\n")

    # Load model
    print("Loading model...")
    if modtype == 'UNET':
        model = U_Net(in_channels=inchan, out_channels=num_classes)  # 6 channels: R,G,B,NIR,MSAVI,NDWI (no elevation)
    if modtype == 'ATTUNET':
        model = AttU_Net(img_ch=inchan, output_ch=num_classes)  # 6 channels: R,G,B,NIR,MSAVI,NDWI (no elevation)


    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Setup transforms
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

    # Initialize metrics
    all_preds = []
    all_targets = []
    all_probs = []
    all_threshpreds = []

    # Setup confusion matrix
    ConfMatrix = ignite.metrics.confusion_matrix.ConfusionMatrix(num_classes=num_classes)
    ThreshConfMatrix = ignite.metrics.confusion_matrix.ConfusionMatrix(num_classes=num_classes)


    mIOU_metric = ignite.metrics.mIoU(ConfMatrix)
    dice_metric = ignite.metrics.DiceCoefficient(ConfMatrix, ignore_index=0)

    TmIOU_metric = ignite.metrics.mIoU(ThreshConfMatrix)
    Tdice_metric = ignite.metrics.DiceCoefficient(ThreshConfMatrix, ignore_index=0)

    num_batches = 0
    num_skipped = 0

    print("Evaluating model on test data...")
    print("-" * 80)

    with torch.no_grad():
        for batch_idx, scene in enumerate(tqdm(test_dataloader, desc="Processing batches")):

            # Process mask
            mask = scene["mask"].clamp(min=-1)

            # Convert mask based on class to evaluate
            if class_to_evaluate == 'WP':
                mask[mask != 2] = 0  # Zero out non-WP
                mask[mask == 2] = 1  # Change WP to 1
            elif class_to_evaluate == 'SP':
                mask[mask > 1] = 0  # Zero out non-SP

            # Remove nodata pixels
            x_prime = scene["image"][:, 0:3, :, :]
            zeros = (x_prime <= 0).all(dim=1)
            mask[zeros] = -1

            nodata = mask == -1

            # Skip if too much nodata
            if torch.sum(nodata).item() >= 15000 * scene["image"].shape[0]:
                num_skipped += 1
                continue

            # Skip if no valid pixels
            valid_pixels = mask != -1
            if valid_pixels.sum() == 0:
                num_skipped += 1
                continue

            # Apply transforms
            x = transforms(scene["image"])

            if not use_elev:
                x = torch.cat((x[:, :4, :, :], x[:, 5:, :, :]), dim=1)
            if not use_GB:
                x = x[:, 2:, :, :]
            if not use_NWI:
                x = torch.cat((x[:, :5, :, :], x[:, 6:, :, :]), dim=1)

            # Move to device
            x = x.type(torch.float32).to(device)
            y = mask.to(device)

            # Forward pass
            outputs = model(x)

            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            #Get thresh preds
            threshmask = probs >= thresh
            threshprobs = probs*threshmask



            #update thresh metric trackers
            ThreshConfMatrix.update((threshprobs,y))
            TmIOU_metric.update((threshprobs,y))
            Tdice_metric.update((threshprobs,y))

            # Update confusion matrix
            ConfMatrix.update((outputs, y))
            mIOU_metric.update((outputs, y))
            dice_metric.update((outputs, y))

            # Flatten and store (exclude nodata)
            preds_flat = torch.reshape(preds, (-1,)).cpu()
            targets_flat = torch.reshape(y, (-1,)).cpu()
            probs_flat = probs[:, 1, :, :].reshape(-1).cpu()  # Probability of positive class
            threshpreds_flat = threshprobs.argmax(dim=1).reshape(-1,).cpu()

            valid_mask = targets_flat != -1

            all_preds.append(preds_flat[valid_mask])
            all_threshpreds.append(threshpreds_flat[valid_mask])
            all_targets.append(targets_flat[valid_mask])
            all_probs.append(probs_flat[valid_mask])

            num_batches += 1

    print(f"\nProcessed {num_batches} batches (skipped {num_skipped})")
    print("=" * 80 + "\n")

    # Concatenate all predictions
    all_preds = torch.cat(all_preds).numpy()
    all_threshpreds = torch.cat(all_threshpreds)
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()

    print(f"Total pixels evaluated: {len(all_targets):,}")
    print(f"Class distribution:")
    for i, name in enumerate(class_names):
        count = (all_targets == i).sum()
        print(f"  {name}: {count:,} pixels ({count / len(all_targets) * 100:.2f}%)")
    print()

    # Calculate metrics
    print("=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)

    # Precision, Recall, F1 (micro and macro)
    precision_micro = precision_score(all_targets, all_preds, average='micro', labels=list(range(num_classes)))
    recall_micro = recall_score(all_targets, all_preds, average='micro', labels=list(range(num_classes)))
    f1_micro = f1_score(all_targets, all_preds, average='micro', labels=list(range(num_classes)))

    precision_macro = precision_score(all_targets, all_preds, average='macro', labels=list(range(num_classes)))
    recall_macro = recall_score(all_targets, all_preds, average='macro', labels=list(range(num_classes)))
    f1_macro = f1_score(all_targets, all_preds, average='macro', labels=list(range(num_classes)))

    precision_binary = precision_score(all_targets, all_preds, average='binary', labels=list(range(num_classes)))
    recall_binary = recall_score(all_targets, all_preds, average='binary', labels=list(range(num_classes)))
    f1_binary = f1_score(all_targets, all_preds, average='binary', labels=list(range(num_classes)))

    Tprecision_binary = precision_score(all_targets, all_threshpreds, average='binary', labels=list(range(num_classes)))
    Trecall_binary = recall_score(all_targets, all_threshpreds, average='binary', labels=list(range(num_classes)))
    Tf1_binary = f1_score(all_targets, all_threshpreds, average='binary', labels=list(range(num_classes)))


    print(f"Micro-averaged:")
    print(f"  Precision: {precision_micro:.4f}")
    print(f"  Recall:    {recall_micro:.4f}")
    print(f"  F1 Score:  {f1_micro:.4f}")
    print()
    print(f"Macro-averaged:")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall:    {recall_macro:.4f}")
    print(f"  F1 Score:  {f1_macro:.4f}")
    print()
    print(f"Binary averaged:")
    print(f"  Precision: {precision_binary:.4f}")
    print(f"  Recall:    {recall_binary:.4f}")
    print(f"  F1 Score:  {f1_binary:.4f}")
    print()

    print()
    print(f"Thresh: Binary averaged:")
    print(f"  Precision: {Tprecision_binary:.4f}")
    print(f"  Recall:    {Trecall_binary:.4f}")
    print(f"  F1 Score:  {Tf1_binary:.4f}")
    print()

    # mIoU and Dice from confusion matrix
    cm_tensor = ConfMatrix.compute()
    mIoU = mIOU_metric.compute().item()
    dice_scores = dice_metric.compute()
    #same for threshold
    Tcm_tensor = ThreshConfMatrix.compute()
    TmIoU = TmIOU_metric.compute().item()
    Tdice_scores = Tdice_metric.compute()

    print(f"Mean IoU (mIoU): {mIoU:.4f}")
    if num_classes > 1:
        for i, name in enumerate(class_names[1:]):  # Skip background
            print(f"  Dice Score ({name}): {dice_scores[i]:.4f}")
    print()

    # Per-class metrics
    cm_numpy = cm_tensor.cpu().numpy()
    per_class_metrics = calculate_metrics_per_class(cm_numpy, num_classes, class_names)
    #thresh
    Tcm_numpy = Tcm_tensor.cpu().numpy()
    Tper_class_metrics = calculate_metrics_per_class(Tcm_numpy, num_classes, class_names)

    print("=" * 80)
    print("PER-CLASS METRICS")
    print("=" * 80)

    for cls_name in class_names:
        metrics = per_class_metrics[cls_name]
        Tmetrics = Tper_class_metrics[cls_name]
        print(f"\n{cls_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  IoU:       {metrics['iou']:.4f}")
        print(f"  Dice:      {metrics['dice']:.4f}")

        print(f"\n{cls_name}:")
        print(f"  ThreshPrecision: {Tmetrics['precision']:.4f}")
        print(f"  ThreshRecall:    {Tmetrics['recall']:.4f}")
        print(f"  ThreshF1 Score:  {Tmetrics['f1']:.4f}")
        print(f"  Thresh IoU:       {Tmetrics['iou']:.4f}")
        print(f"  Thresh Dice:      {Tmetrics['dice']:.4f}")

        print(f"  Support:   {metrics['support']:,} pixels")

    print("\n" + "=" * 80)

    # Plot confusion matrix
    plot_confusion_matrix(cm_numpy, class_names,
                          save_path=f'{save_dir}/confusion_matrix.png')

    plot_confusion_matrix(Tcm_numpy, class_names,
                          save_path=f'{save_dir}/confusion_matrix_Thresh.png')

    # Plot per-class metrics
    plot_per_class_metrics(per_class_metrics, class_names,
                           save_path=f'{save_dir}/per_class_metrics.png')

    plot_per_class_metrics(Tper_class_metrics, class_names,
                           save_path=f'{save_dir}/per_class_metrics_Thresh.png')
    # Save metrics to CSV
    results_df = pd.DataFrame(per_class_metrics).T
    results_df.to_csv(f'{save_dir}/per_class_metrics.csv')
    print(f"\nSaved per-class metrics to {save_dir}/per_class_metrics.csv")

    # Save summary
    summary = {
        'model_path': model_path,
        'num_pixels': len(all_targets),
        'num_batches': num_batches,
        'num_skipped': num_skipped,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'mIoU': mIoU,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f'{save_dir}/evaluation_summary.csv', index=False)
    print(f"Saved evaluation summary to {save_dir}/evaluation_summary.csv")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return {
        'summary': summary,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm_numpy,
        'all_predictions': all_preds,
        'all_targets': all_targets,
        'all_probabilities': all_probs
    }


if __name__ == "__main__":
    """
    Example usage
    """

    # ========== CONFIGURATION ==========

    # Model configuration
    modfile = 'UM_L_creekproxUnet_cleanmask32res3m_v1_bestmIOU_0.69_epoch75' #('nonoutlier_NWI2_bestF1_0.71_epoch119')#'collated_SPrun21_bestmIOU_0.79_epoch45'
    MODEL_PATH = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/saved_models/{}.pth'.format(modfile)  # Path to  trained model
    modtype = 'UNET'
    NUM_CLASSES = 2
    NUM_INPUT = 9
    CLASS_NAMES = ['Background', 'Salt Pan']
    CLASS_TO_EVALUATE = 'SP'  # 'SP' or 'WP'

    #cleaned mask
    cleanmask ='/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/mask3mRes32up/'

    # Data configuration
    #May
    May_test_data = PlanetScope('/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/test/2025-05/images/')
    May_test_mask = PlanetMask(cleanmask)
    #June
    June_test_data = PlanetScope('/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/test/2025-06/images/')
    June_test_mask = PlanetMask(cleanmask)
    #Sept
    Sept_test_data = PlanetScope('/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/test/2025-09/images/')
    Sept_test_mask = PlanetMask(cleanmask)
    #Oct
    Oct_test_data = PlanetScope('/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/test/2025-10/images/')
    Oct_test_mask = PlanetMask(cleanmask)

    #elevation
    ELEV_DATA_DIR = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/elev/'
    elev = ElevationData(ELEV_DATA_DIR)

    #NWI
    nwi_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/roi_shpfiles/wetlands_inv/usable/'
    nwi = NWIData(nwi_data)

    # DEM slope data
    slope_data = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/slope/'
    slope = SlopeData(slope_data)

    # creek proximity raster
    creekproxdata = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/creekprox/'
    creekprox = CreekProxData(creekproxdata)

    #Join imgs and masks
    MAYtest = (May_test_data & May_test_mask) & (elev & nwi) & (slope & creekprox)
    JUNEtest = (June_test_data & June_test_mask) & (elev & nwi) & (slope & creekprox)
    SEPTtest = (Sept_test_data & Sept_test_mask) & (elev & nwi) & (slope & creekprox)
    OCTtest = (Oct_test_data & Oct_test_mask) & (elev & nwi) & (slope & creekprox)

    #Build random time sampler
    test_dataset = RandomTemporalDataset([MAYtest, JUNEtest, SEPTtest,OCTtest]) #MAYtest,


    # Processing configuration
    BATCH_SIZE = 16
    PATCH_SIZE = 256
    NUM_SAMPLES = 1600  # Number of random samples to evaluate
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Transform configuration
    USE_MSAVI = True  # Use MSAVI instead of NDVI
    USE_NORMALIZATION = True  # Set to True if you want to use normalization
    USE_ELEV = True
    USE_GB = True

    # If using normalization, load stats
    if USE_NORMALIZATION:
        # MAY

        MAYT_Img_Stats = getstats('/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/test/2025-05/images/', bands=4, imgcount=May_test_data.__len__())
        MAYTMax = MAYT_Img_Stats["AvgMax"]
        MAYTMin = MAYT_Img_Stats["AvgMin"]
        # June
        JuneT_Img_Stats = getstats('/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/test/2025-06/images/', bands=4, imgcount=June_test_data.__len__())
        JuneTMax = JuneT_Img_Stats["AvgMax"]
        JuneTMin = JuneT_Img_Stats["AvgMin"]
       
        # Sept
        SeptT_Img_Stats = getstats('/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/test/2025-09/images/', bands=4, imgcount=Sept_test_data.__len__())
        SeptTMax = SeptT_Img_Stats["AvgMax"]
        SeptTMin = SeptT_Img_Stats["AvgMin"]
        # Oct
        OctT_Img_Stats = getstats('/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/unmasked4bClips/test/2025-10/images/', bands=4, imgcount=Oct_test_data.__len__())
        OctTMax = OctT_Img_Stats["AvgMax"]
        OctTMin = OctT_Img_Stats["AvgMin"]
        # calc outb avg
        AllTMax = np.array([MAYTMax, SeptTMax, JuneTMax,  OctTMax])
        TMax = np.max(AllTMax, axis=0)
        print(TMax)

        AllTMin = np.array([MAYTMin, SeptTMin, JuneTMin, OctTMin])
        TMin = np.min(AllTMin, axis=0)
        print(TMin)


        # get elevation and slope stats
        ElevStats = getstats(ELEV_DATA_DIR, bands=1, imgcount=1)
        emax = float(ElevStats['Max'].item())
        print(emax)
        emin = float(ElevStats['Min'].item())
        SlopeStats = getstats(slope_data, bands=1, imgcount=1)
        smax = float(SlopeStats['Max'].item())
        smin = float(SlopeStats['Min'].item())
        CreekProxStats = getstats(creekproxdata, bands=1, imgcount=1)
        cpmax = float(CreekProxStats['Max'].item())
        cpmin = float(CreekProxStats['Min'].item())

        # Add in those values, plus NWI range
        TMax, TMin= np.append(TMax, [emax, 8.0, smax, cpmax], axis=0), np.append(TMin,[emin, 0.0, smin, cpmin],axis=0)

        #
        # TMax = torch.tensor([0.1,0.1,0.1,0.25,1.5,8.0])#torch.from_numpy(np.load('/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/allTMax.npy')).float()
        # TMin = torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0]) #torch.from_numpy(np.load('/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/allTMin.npy')).float()
        STATS = [torch.from_numpy(TMax), torch.from_numpy(TMin)]
        print(STATS)
    else:
        STATS = None

    # Output directory
    SAVE_DIR = './evaluation_results/{}_cleanover48/'.format(modfile)



    print(f"Test dataset: {test_dataset}")
    print(f"Number of files: {len(test_dataset)}")

    # Create sampler and dataloader
    test_sampler = RandomBatchGeoSampler(
        test_dataset,
        size=PATCH_SIZE,
        length=NUM_SAMPLES,
        batch_size=BATCH_SIZE
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=stack_samples
    )

    print(f"Test dataloader created: {len(test_dataloader)} batches\n")

    # ========== RUN EVALUATION ==========

    results = evaluate_model(
        model_path=MODEL_PATH,
        modtype = modtype,
        test_dataloader=test_dataloader,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        inchan = NUM_INPUT,
        class_names=CLASS_NAMES,
        use_msavi=USE_MSAVI,
        use_elev=USE_ELEV,
        use_GB=USE_GB,
        use_NWI=False,
        use_normalization=USE_NORMALIZATION,
        stats=STATS,
        class_to_evaluate=CLASS_TO_EVALUATE,
        thresh=0.7,
        save_dir=SAVE_DIR
    )

    print("\nResults saved to:", SAVE_DIR)
    print("Files generated:")
    print("  - confusion_matrix.png")
    print("  - per_class_metrics.png")
    print("  - per_class_metrics.csv")
    print("  - evaluation_summary.csv")