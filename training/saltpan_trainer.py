
import time
from helper_functions import MinMaxNormalize,F1_score
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from torchvision.transforms import v2
from torchgeo.transforms import AppendNDWI,AppendNDVI
import ignite
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.utils import make_grid
import warnings



warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')


#introduce colors for output legibility
torch.set_printoptions(threshold=10000)
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


def plot_confusion_matrix(cm, class_names):
    cm = cm.detach().cpu().numpy()
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.astype(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df_cm, annot=True, ax=ax, cmap='Spectral') #, fmt="d"
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

def train(model, train_dl,valid_dl,  loss_fn, optimizer, epochs=5,batches=1,batchsize = 1,
          modname = None,scheduler = None,modID = "defaultID", stats = None):

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
        # v2.RandomHorizontalFlip(p=0.5),
        # v2.RandomVerticalFlip(p=0.5),
        # v2.RandomRotation(degrees=90),

        AppendNDVI(index_nir=4, index_red=3),
        AppendNDWI(index_nir=4, index_green=2)
    ])
#validation transforms
    addIndicesV = v2.Compose([

        MinMaxNormalize(min=VMin, max=VMax),
        AppendNDVI(index_nir=4, index_red=3),
        AppendNDWI(index_nir=4, index_green=2)
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
            'mIoU': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_mIoU'),
            'DiceSP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_DiceSP'),
            'DiceDB': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_DiceDB'),
            'DiceWP': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_DiceWP'),
            'PConfMat': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_PConfMat'),
            'RConfMat': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_RConfMat'),
            'TransIMG': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_TransIMG'),
            'MASK': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_MASK'),
            'PRED': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_PRED'),
            'NoData': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_NoData'),
            'DATA': SummaryWriter(f'TBruns/{modID}/{modID}_{phase}_DATA')
        }
#set up confusion matrix logger
    ConfMatrix = ignite.metrics.confusion_matrix.ConfusionMatrix(num_classes=4)
    mIOU = ignite.metrics.mIoU(ConfMatrix,ignore_index=0)
    dice = ignite.metrics.DiceCoefficient(ConfMatrix,ignore_index=0)
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
            dice.reset()
            mIOU.reset()
            PConfMatrix.reset()
            RConfMatrix.reset()



            running_loss = 0.0


            step = 0
            num_skipped = 0
            allpreds = []
            alltargets = []

            # iterate over data
            for batchID, scene in enumerate(dataloader):

                step += 1
                mask = scene["mask"].clamp(min = -1)
                #print(img.shape)
                nodata = mask == -1

                if torch.sum(nodata).item() >= 100:
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




                # forward pass & save image for tensorboard
                img = scene["image"][:,[2,1,0],:,:]
                if phase == 'train':
                    model.train(True)  # Set training mode = true
                    # zero the gradients
                    optimizer.zero_grad()


                    y = scene["mask"].clamp(min=-1)
                    #print(y.shape)
                    #print("trainmask",y.min(),y.max())



                    x = scene["image"]
                    #print(x.shape)
                    x = addIndicesT(x)
                    # xRGB = x[:,:3,:,:]
                    # xtra = x[:,3:,:,:]
                    # xRGBjitter = CJ(xRGB)
                    #x = torch.cat([xRGBjitter,xtra],dim = 1)
                    #print(x.shape)
                    timg = x[:,[2,1,0],:,:]
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
                    PConfMatrix.update((outputs,y))
                    RConfMatrix.update((outputs, y))
                    running_loss += loss.item()
                else:
                    with torch.no_grad():
                        model.train(False)  # Set model to evaluate mode
                        y = scene["mask"].clamp(min=-1).cuda()
                        x = addIndicesV(scene["image"])
                        timg = x[:, [2, 1, 0], :, :]
                        elev = x[:, [4], :, :]
                        ndvi = x[:, [5], :, :]
                        ndwi = x[:, [6], :, :]
                        x = x.cuda()
                        outputs = model(x)

                        #print(outputs.shape)
                        loss = Vloss_fn(outputs, y) #['out']
                        running_loss += loss.item()  # * batch_size
                        dice.update((outputs, y))
                        mIOU.update((outputs, y))
                        PConfMatrix.update((outputs,y))
                        RConfMatrix.update((outputs, y))


                # stats - whatever is the phase
                y2 = torch.reshape(y, (-1,))
                # print(y2.shape)

                #print(y.shape)
                pred = outputs.argmax(dim=1).detach().cpu()#
                pred = torch.reshape(pred,(-1,))
                allpreds.append(pred)

                #print("pred shape:",pred.shape,"pred unique:", np.unique(pred.numpy()))
                assert not torch.isnan(pred).any()
                assert not torch.isinf(pred).any()
                y2 = y2.cpu()

                alltargets.append(y2)


            if alltargets:
                epoch_loss = running_loss /(step - num_skipped +0.0000000001)
                epoch_prec = precision_score(torch.cat(alltargets),torch.cat(allpreds),average='micro',labels=[0,1,2,3])
                epoch_rec = recall_score(torch.cat(alltargets),torch.cat(allpreds),average='micro',labels=[0,1,2,3])
                epoch_F1micro = f1_score(torch.cat(alltargets),torch.cat(allpreds),average='micro',labels=[0,1,2,3])
                epoch_F1macro = f1_score(torch.cat(alltargets),torch.cat(allpreds),average='macro',labels=[0,1,2,3])


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


                writers[phase]['mIoU'].add_scalar("mIoU/{}".format(phase), mIOU.compute(), epoch)
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
                writers[phase]['TransIMG'].add_image('Transformed Images/{}'.format(epoch), transimg_grid, global_step=epoch)
                print('elev:', elev.shape,'ndvi: ',ndvi.shape,'ndwi: ',ndwi.shape)
                elev_grid = make_grid(elev.squeeze(1).float(),nrow=4)#,normalize=True,value_range=(0,1)
                ndvi_grid = make_grid(ndvi.squeeze(1).float(),nrow=4)#,normalize=True,value_range=(-1,1)
                ndwi_grid = make_grid(ndwi.squeeze(1).float(),nrow=4) #,normalize=True,value_range=(-1,1)

                mask_grid = make_grid(mask.unsqueeze(1).float(), nrow=4) #,scale_each=True
                writers[phase]['MASK'].add_image('Masks/{}'.format(epoch), mask_grid, global_step=epoch)
                print("out {} | pred {}".format(outputs.argmax(dim=1).detach().cpu().shape,pred.shape))
                inpred = outputs.argmax(dim=1).detach().cpu().unsqueeze(1).float()
                print("img {}   mask {}     pred {}".format(img.shape,mask.shape,inpred.shape))


                pred_grid = make_grid(inpred, nrow=4)#,scale_each=True
                all = torch.cat([img_grid, mask_grid, pred_grid], dim=1)
                data = torch.cat([elev_grid, ndvi_grid, ndwi_grid], dim=1)
                print("data",data.shape)
                print("all", all.shape)
                writers[phase]['PRED'].add_image('Preds/{}_{}'.format(phase,epoch), all, global_step=epoch)  #FIGURE THIS OUT
                writers[phase]['DATA'].add_image('Data/{}_{}'.format(phase,epoch), data, global_step=epoch)  #FIGURE THIS OUT


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
    train_F1 = sum(train20) / len(train20)
    valid_F1 = sum(valid20) / len(valid20)
#print finishing info
    time_elapsed = time.time() - start
    print(bcolors.OKGREEN + 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(bcolors.OKGREEN + 'Number of empty batches skipped: {}'.format(num_skipped))
    print(bcolors.OKGREEN + "Best model: {}".format(best_mIOU))
    print(bcolors.ENDC)
    print("Best model CM: \n",savedpCM.compute())


    return train_precision, valid_precision, train_recall, valid_recall,train_F1,valid_F1, model,best_mIOU