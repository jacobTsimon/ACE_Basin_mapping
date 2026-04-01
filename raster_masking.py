#!!this file is copied into QGIS built-in python terminal!!
#aim here is to isolate marsh pixels by applying a basic elevation threshold mask to training imgs
import processing
import os
from qgis.core import *
import qgis.utils
import re

# get current layer - imagery goes here
elevation = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/elev_mask_0to1point5.tif'
imgfolder = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/4bStackClips/'

alg_ID = "gdal:rastercalculator"
extent = "527400.0000,575259.0000,3573738.0000,3615723.0000 [EPSG:32617]"
for dir, _, files in os.walk(imgfolder):
    for file in files:
        print(file)
        if file == "20230628_151127_55_2440_3B_AnalyticMS_SR_8b.tif":
            print('SKIPPED PROB FILE')
            next
        if re.search('.*tif$',file):
            p = os.path.join(dir, file)
            p_stem, ext = os.path.splitext(file)
            print(p)
            exp = ""

            save = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/img_preprocessing/masked4bClips/{}elevmask.tif'.format(
                p_stem)
            print(save)
            params = {
                "INPUT_A": p,
                "BAND_A": 1,
                "INPUT_B": elevation,
                "BAND_B": 1,
                "EXPRESSION": "B*A",
                "RTYPE": 5,
                "OUTPUT": save,
                "PROJWIN": extent,
                "EXTENT": "union",
                "EXTRA": "--allBands A"
            }
            processing.run(alg_ID, params)