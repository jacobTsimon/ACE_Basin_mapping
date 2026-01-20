#used only for copy + paste into QGIS built in python console
import processing
from qgis.core import *
from qgis.core import QgsVectorLayer

#get .shp ROI layer
shapefile = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/val_rois.shp'

outfile = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/val_extents23.txt'

write = open(outfile,'w')

veclayer = QgsVectorLayer(shapefile)
it = veclayer.getFeatures()


#write out each min/max in a txt file for later
t = 0
for feature in it:
    t  +=1
    bbox = feature.geometry().boundingBox()
    print(t)
    write.write("{}\t{}\t{}\t{}\t{}\t\n".format(t,bbox.xMinimum(), bbox.xMaximum(), bbox.yMinimum(), bbox.yMaximum()))
write.close()