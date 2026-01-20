#!!this file is copied into QGIS built-in python terminal!!
#aim here is to isolate marsh pixels by applying a basic elevation threshold mask to training imgs
import processing
import os
from qgis.core import *
import qgis.utils

# get current layer - imagery goes here
elevation = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/elev_mask_0to1point5.tif'
imgfolder = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/training/imgs/scenes/'

alg_ID = "gdal:rastercalculator"
extent = "527400.0000,575259.0000,3573738.0000,3615723.0000 [EPSG:32617]"

for dir, _, files in os.walk(imgfolder):
    for file in files:
        print(file)
        if file == "20230628_151127_55_2440_3B_AnalyticMS_SR_8b.tif":
            print('SKIPPED PROB FILE')
            next
        p = os.path.join(dir, file)
        p_stem, ext = os.path.splitext(file)
        print(p)
        exp = ""
        for band in range(1, 9):
            print(band)

            save = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/img_preprocessing/masked/{}elevthresh_band{}.tif'.format(
                p_stem, band)
            print(save)
            params = {
                "INPUT_A": p,
                "BAND_A": band,
                "INPUT_B": elevation,
                "BAND_B": 1,
                "EXPRESSION": "B*A",
                "RTYPE": 5,
                "OUTPUT": save,
                "PROJWIN": extent
            }
            processing.run(alg_ID, params)