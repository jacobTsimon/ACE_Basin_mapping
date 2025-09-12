import processing
from qgis.core import *
import qgis.utils
from qgis.core import QgsVectorLayer
#get current layer - imagery goes here
shapefile = '/Users/jts30437/Downloads/MAD Fellowship/research/train_rois.shp'
inraster = '/Users/jts30437/Downloads/MAD Fellowship/research/ElevThresh_1m50cm.tif'
inlayer = QgsRasterLayer(inraster)
inextent = inlayer.extent()
print(inextent)
veclayer = QgsVectorLayer(shapefile, 'borders', 'ogr')
it = veclayer.getFeatures()
alg_ID = "gdal:cliprasterbyextent"
params = {
    "INPUT" : inlayer,
    "EXTENT" : None
}
t = 0
for feature in it:
    t  +=1
    bbox = feature.geometry().boundingBox()
    bbox_extent = [bbox.xMinimum(), bbox.xMaximum(), bbox.yMinimum(), bbox.yMaximum()]
    print(bbox_extent)
    print(t)
    rect = QgsRectangle(bbox.xMinimum(), bbox.yMinimum(), bbox.xMaximum(),bbox.yMaximum())
    if inextent.contains(rect):
        print("INTERSECTS!")
        params["EXTENT"] = bbox_extent
        #params["PROJWIN"] = bbox_extent
        processing.run(alg_ID, params)
    print(params["EXTENT"])