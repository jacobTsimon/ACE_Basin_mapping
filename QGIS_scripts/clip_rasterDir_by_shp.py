
#needs to be run from inside the QGIS python console

import processing
from qgis.core import QgsVectorLayer

import os
import sys
import processing
from qgis.core import (
    QgsApplication,
    QgsVectorLayer,
    QgsFeature,
    QgsProcessingFeedback,
)


def init_qgis(qgis_prefix_path=None):
    # qgis_prefix_path: path to your QGIS install root.
    #     Windows (OSGeo4W):  r"C:\OSGeo4W\apps\qgis"
    #     macOS:               "/Applications/QGIS.app/Contents/MacOS"
    #     Linux:                "/usr"   (often auto-detectable)

    if qgis_prefix_path is None:
        # Try common environment variable first
        qgis_prefix_path = os.environ.get("QGIS_PREFIX_PATH", "/usr")

    QgsApplication.setPrefixPath(qgis_prefix_path, True)
    qgs = QgsApplication([], False)  # False = no GUI
    qgs.initQgis()

    # Make the 'processing' plugin's algorithms available
    sys.path.append(os.path.join(qgis_prefix_path, "python", "plugins"))
    from processing.core.Processing import Processing
    Processing.initialize()
    import processing as proc_module  # re-import now that it's initialized

    return qgs


def clip_rasters_by_mask_features(
    input_dir,
    output_dir,
    mask_layer_path,
    id_field=None,
    raster_ext=(".tif", ".tiff"),
    nodata_value=None,
    crop_to_cutline=True,
    output_fname = 'train'
):
    #
    # Loop through a directory of rasters AND through each feature in a mask
    # layer, clipping every raster against every individual feature

    # Parameters
    # ----------
    # input_dir : str
    #     Folder containing raster images to clip.
    # output_dir : str
    #     Folder where clipped rasters will be saved.
    # mask_layer_path : str
    #     Path to the vector mask layer (shapefile, GeoPackage, etc).
    # id_field : str or None
    #     Field name used to label output files per feature (e.g. "name" or
    #     "id"). If None, the feature's FID is used instead.
    # raster_ext : tuple
    #     File extensions treated as rasters.
    # nodata_value : float or None
    #     NoData value for outputs.
    # crop_to_cutline : bool
    #     Whether to crop extent to each feature's geometry.
    #
    # Returns
    # -------
    # list of str
    #     Paths to successfully created output rasters.
    #
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    mask_layer = QgsVectorLayer(mask_layer_path, "mask", "ogr")
    if not mask_layer.isValid():
        raise ValueError(f"Could not load mask layer: {mask_layer_path}")

    if id_field and id_field not in [f.name() for f in mask_layer.fields()]:
        raise ValueError(f"Field '{id_field}' not found in mask layer")

    raster_files = [
        f for f in os.listdir(input_dir) if f.lower().endswith(raster_ext)
    ]
    if not raster_files:
        print(f"No rasters found in {input_dir}")
        return []

    outputs = []
    feedback = QgsProcessingFeedback()

    for feature in mask_layer.getFeatures():
        # Build a temporary single-feature memory layer for this feature
        feature_layer = QgsVectorLayer(
            f"{mask_layer.geometryType().name}?crs={mask_layer.crs().authid()}",
            "feature_mask",
            "memory",
        )
        feature_layer.dataProvider().addAttributes(mask_layer.fields())
        feature_layer.updateFields()

        new_feat = QgsFeature(feature_layer.fields())
        new_feat.setGeometry(feature.geometry())
        new_feat.setAttributes(feature.attributes())
        feature_layer.dataProvider().addFeature(new_feat)
        feature_layer.updateExtents()

        # Label for this feature's outputs
        if id_field:
            label = str(feature[id_field]).replace(" ", "_")
        else:
            label = f"fid{feature.id()}"

        for filename in raster_files:
            input_path = os.path.join(input_dir, filename)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(
                output_dir, f"{name}_{label}_{output_fname}_clipped{ext}"
            )

            params = {
                "INPUT": input_path,
                "MASK": feature_layer,
                "SOURCE_CRS": None,
                "TARGET_CRS": None,
                "TARGET_EXTENT": None,
                "NODATA": nodata_value,
                "ALPHA_BAND": False,
                "CROP_TO_CUTLINE": crop_to_cutline,
                "KEEP_RESOLUTION": True,
                "SET_RESOLUTION": False,
                "X_RESOLUTION": None,
                "Y_RESOLUTION": None,
                "MULTITHREADING": False,
                "OPTIONS": "",
                "DATA_TYPE": 0,
                "EXTRA": "",
                "OUTPUT": output_path,
            }

            try:
                result = processing.run(
                    "gdal:cliprasterbymasklayer", params, feedback=feedback
                )
                outputs.append(result["OUTPUT"])
                print(f"Clipped: {filename} x feature {label} -> {result['OUTPUT']}")
            except Exception as e:
                print(f"Failed on {filename} x feature {label}: {e}")

    return outputs



#function call setup
clip_rasters_by_mask_features(
    input_dir= '/media/hopkinsonlab/JS_LabDrive/ACOLITE/processed_imgs/uptide4bstacks/',
    output_dir= '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/img_preprocessing/',
    mask_layer_path='/home/hopkinsonlab/Desktop/ACE_Basin_mapping/roi_shpfiles/merged_trainrois2025.shp',
    raster_ext= '.tif',
    nodata_value= -10,
    crop_to_cutline=True,
    output_fname= 'train'
)