import glob
import os
import shutil
from pathlib import Path
import fnmatch
#script for moving image files identified to have outlier acquisition metadata

#load relevant outlier log files
outlierfile = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/analysis/band analysis/outliers2023'
mark = '20 files with more than 2 outlier acquisition variables\n'

with open(outlierfile, 'r') as ofile:
    contents = ofile.readlines()
    ind = contents.index(mark)
    outliers = contents[ind+1:]

#gather the outlier file names
IDs = []
for line in outliers:
    chunks = line.split('.')
    IDs.append(chunks[0])

#predictions directory
pred_dir = "/home/hopkinsonlab/Desktop/ACE_Basin_mapping/analysis/pred23boundsfix_masked"
pred_suffix = ["*_predicted_masked.tif","*_predicted_prob_masked.tif"]
#directory to move outlier imgs to
movedir = "/home/hopkinsonlab/Desktop/ACE_Basin_mapping/analysis/pred23boundsfix_masked_nonoutlier/"
os.makedirs(movedir, exist_ok=True)

for suff in pred_suffix:
    #loop over binary and probabilities
    pred_files = sorted(glob.glob(f"{pred_dir}**/{suff}"))
    print(f"Found {len(pred_files)} {suff} rasters")

    for pred_string in pred_files:
        pred_path = Path(pred_string)
        #Parse scene ID from filename: 24a8_2023_03_19_15_47_30_4b_stack_predicted.tif
        #Extract the scene prefix (everything before _4b_stack)
        stem = pred_path.stem.replace("_predicted", "")
        chop = stem.split("_")  # returns a list
        scene_id = chop[1] + chop[2] + chop[3] + "_" + chop[4] + chop[5] + chop[6] + "_[0-9][0-9]_" + chop[0]
        IDpatt = scene_id + "_metadata"
        matches = fnmatch.filter(IDs, IDpatt)
        if len(matches) > 0:
            print(f"Found an outlier: {pred_string}")
            name = pred_string.split('/')[-1]
            print(f"Moving file {name}")
            shutil.move(pred_string, movedir + name)



