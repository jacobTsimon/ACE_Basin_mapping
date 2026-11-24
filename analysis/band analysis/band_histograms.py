import rasterio
import matplotlib.pyplot as plt
import numpy as np


def plot_band_histograms(tif_file_path,title = 'default title',index =False):

    with rasterio.open(tif_file_path) as src:
        num_bands = src.count
        print(f"Number of bands found: {num_bands}")
        if index:
            b = input("index abbr. :")
            c = input("clor code :")
            planetbands = [b]
            colors = [c]
        else:
            planetbands = ["Coastal Blue","Blue","Green I", "Green","Yellow","Red","Red Edge","NIR"]
            colors = ['c','b','mediumseagreen','g','y','r','m','darkred']

        # Determine the number of rows/columns for the subplot grid
        # A simple grid of 2 rows is used if there are more than 1 band
        rows = 2 if num_bands > 1 else 1
        cols = (num_bands + 1) // rows if num_bands > 1 else 1

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False,sharex=True)
        axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

        for i in range(1, num_bands + 1):
            band_data = src.read(i)
            ax = axes[i - 1]

            # Filter out no-data values if they exist
            no_data = src.nodata
            if no_data is not None:
                valid_data = band_data[band_data != no_data]
            else:
                valid_data = band_data.flatten()  # Flatten the array for histogram calculation
            #exclude 0s
            valid_data = valid_data[valid_data != 0]
            # Use numpy to calculate histogram bins
            perc98 = np.percentile(valid_data, 98)
            #exclude outliers from stats
            valid_data = valid_data[valid_data<=perc98]
            # get stats
            avg = np.mean(valid_data)
            std = np.std(valid_data)
            top = np.max(valid_data)
            text = "Mean: {:.2f}\nStdDev: {:.2f}".format(avg, std)


            ax.hist(valid_data, bins=50, color=colors[i-1], alpha=0.7,range = (-1,perc98))
            bottom,topy = ax.get_ylim()
            ax.set_title(f'{planetbands[i-1]} Histogram')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.5)
            ax.text((top-(top*0.15)),topy-(topy*0.15),text)


        # Hide any unused subplots
        for j in range(num_bands, len(axes)):
            fig.delaxes(axes[j])

        #plt.tight_layout()
        plt.suptitle(title)
        plt.show()


dirpath = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping'
tif_path = dirpath + '/whole317814WP.tif'
title = "Wrack pixels(img 317814, Sept 2025)"
plot_band_histograms(tif_path,title = title)#,index = True