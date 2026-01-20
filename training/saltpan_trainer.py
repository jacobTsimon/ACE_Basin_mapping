
import time
from helper_functions import MinMaxNormalize,F1_score
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from torchvision import transforms
from torchgeo.transforms import AppendNDWI,AppendNDVI
import ignite

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


def train(model, train_dl,valid_dl,  loss_fn, optimizer, epochs=5,batches=1,
          modname = None,scheduler = None,modID = "defaultID", stats = None):
    start = time.time()
    model.float()
    model.cuda()
    best_F1 = 0

    #set up transforms

    #stats
    TMax = torch.tensor(stats[0]).float()
    TMin = torch.tensor(stats[1]).float()
    VMax = torch.tensor(stats[2]).float()
    VMin = torch.tensor(stats[3]).float()
#training transforms
    addIndicesT = transforms.Compose([

        MinMaxNormalize(min = TMin,max=TMax),
        AppendNDVI(index_nir=4, index_red=3),
        AppendNDWI(index_nir=4, index_green=2)
    ])
#validation transforms
    addIndicesV = transforms.Compose([

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
    trainMatrix = ignite.metrics.confusion_matrix.ConfusionMatrix(num_classes=4)
    valMatrix = ignite.metrics.confusion_matrix.ConfusionMatrix(num_classes=4)

#begin train loop
    for epoch in range(epochs):
        print(bcolors.WARNING + 'Epoch {}/{}'.format(epoch, epochs - 1))
        print(bcolors.ENDC)
        print('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set training mode = true
                dataloader = train_dl
                Llogdir = 'TBruns/{}_train_Loss'.format(modID) #%%
                Plogdir = 'TBruns/{}_train_Precision'.format(modID)
                Rlogdir = 'TBruns/{}_train_Recall'.format(modID)
                Flogdir = 'TBruns/{}_train_F1'.format(modID)
                Lwriter = SummaryWriter(log_dir=Llogdir)
                Pwriter = SummaryWriter(log_dir=Plogdir)
                Rwriter = SummaryWriter(log_dir=Rlogdir)
                Fwriter = SummaryWriter(log_dir=Flogdir)

            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl
                Llogdir = 'TBruns/{}_val_Loss'.format(modID)
                Plogdir = 'TBruns/{}_val_Precision'.format(modID)
                Rlogdir = 'TBruns/{}_val_Recall'.format(modID)
                Flogdir = 'TBruns/{}_val_F1'.format(modID)
                Lwriter = SummaryWriter(log_dir=Llogdir)
                Pwriter = SummaryWriter(log_dir=Plogdir)
                Rwriter = SummaryWriter(log_dir=Rlogdir)
                Fwriter = SummaryWriter(log_dir=Flogdir)


            running_loss = 0.0
            running_P = 0.0
            running_R = 0.0
            running_F1 = 0.0

            step = 0

            # iterate over data
            for batchID, scene in enumerate(dataloader):
                x1 = scene["image"].cuda()
                print(x1.shape)

                y = scene["mask"]
                #print(y.shape)
                y = y[:,0,:,:] # convert the 4D tensors to 3D (Batch size, X, Y)
                #print(y.sum())
                y = y.cuda()
                step += 1

                # forward pass
                if phase == 'train':
                    x = addIndicesT(scene["image"])
                    x = x.cuda()

                    outputs = model(x)

                    loss = loss_fn(outputs, y)

                    # zero the gradients
                    optimizer.zero_grad()

                    g1 = list(model.parameters())[0].clone()
                    # print(g1)
                    loss.backward()
                    optimizer.step()
                    g2 = list(model.parameters())[0].clone()
                    # print(g2)
                    # scheduler.step()
                    trainMatrix.update((outputs,y))

                else:
                    with torch.no_grad():
                        x = addIndicesV(scene["image"])
                        x = x.cuda()
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long()) #['out']
                        valMatrix.update((outputs,y))
                #check grads updating - seem to be working normally
                print(torch.equal(g1.data,g2.data))

                # stats - whatever is the phase
                a = torch.reshape(outputs.argmax(dim=1).detach().cpu(),(-1,))
                #print("pred shape:",a.shape,"pred unique:", np.unique(a.numpy()))
                assert not torch.isnan(a).any()
                assert not torch.isinf(a).any()
                #print("mask shape", y.shape, "y0 unique:",np.unique(y.cpu().numpy()))
                b = torch.reshape(y.cpu(),(-1,))
                #print("mask re shape", b.shape, "b unique:",np.unique(b.cpu().numpy()))

                precision = precision_score(y_pred = a,y_true = b,average = "micro")
                recall = recall_score(y_pred = a,y_true = b,average = "micro")
                f1 = F1_score(precision,recall)
                skf1 = f1_score(y_true = b,y_pred = a,average = "micro")

                running_loss += loss #* batch_size
                running_P += precision
                running_R += recall
                running_F1 += skf1
                print(step)

                if step % 10 == 0:
                    # clear_output(wait=True)
                    print(bcolors.BOLD + 'Current step: {}  Loss: {}   Precision: {} Recall: {} F1: {} AllocMem (Mb): {}'.format(step, loss, precision,recall,f1,torch.cuda.memory_allocated() / 1024 / 1024))
                    print(bcolors.ENDC)
            if scheduler:
                scheduler.step()
                print("current LR: {}".format(scheduler.get_last_lr()))

            epoch_loss = running_loss / batches

            epoch_prec = running_P/(batches - num_skipped)
            epoch_rec = running_R/(batches - num_skipped)
            epoch_F1 = running_F1/(batches - num_skipped)

            Lwriter.add_scalar("Loss/{}".format(phase),epoch_loss,epoch)
            Pwriter.add_scalar("Precision/{}".format(phase),epoch_prec,epoch)
            Rwriter.add_scalar("Recall/{}".format(phase),epoch_rec,epoch)
            Fwriter.add_scalar("F1/{}".format(phase),epoch_F1,epoch)


            num_skipped = 0

            print(bcolors.OKBLUE + '{} epoch {} | epoch Loss: {:.4f} |   epoch Precision: {} | epoch Recall: {} | epoch F1: {}'.format(phase,epoch, epoch_loss, epoch_prec,epoch_rec,epoch_F1))
            print(bcolors.ENDC)

            train_precision.append(epoch_prec) if phase == 'train' else valid_precision.append(epoch_prec)
            train_recall.append(epoch_rec) if phase == 'train' else valid_recall.append(epoch_rec)
            #attempt to exclude nan values while doing optuna optimization
            if phase == 'train' and np.isnan(epoch_F1) == False:
                train_F1.append(epoch_F1)
            if phase == 'valid' and np.isnan(epoch_F1) == False:
                valid_F1.append(epoch_F1)

            #save best model
            if phase == 'valid' and epoch_F1 > best_F1:
                best_F1 = epoch_F1
                print(bcolors.OKGREEN + "New best model F1: {}".format(epoch_F1))
                print(bcolors.ENDC)
                torch.save(model.state_dict(),'./saved_models/{}.pth'.format(modID)) #%%
                print(trainMatrix.compute())
                print(valMatrix.compute())


    # average F1 scores
    train20 = train_F1[-20:-1]
    print(train20) #this grabs UP TO the last 20 F1 scores (in early trials there are  often less)
    valid20 = valid_F1[-20:-1]
    train_F1 = sum(train20) / len(train20)
    valid_F1 = sum(valid20) / len(valid20)
#print finishing info
    time_elapsed = time.time() - start
    print(bcolors.OKGREEN + 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(bcolors.OKGREEN + 'Number of empty batches skipped: {}'.format(num_skipped))
    print(bcolors.OKGREEN + "Best model: {}".format(best_F1))
    print(bcolors.ENDC)
    print(trainMatrix.compute())
    print(valMatrix.compute())

    return train_precision, valid_precision, train_recall, valid_recall,train_F1,valid_F1, model,best_F1