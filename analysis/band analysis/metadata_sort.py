from datetime import datetime, time
import re
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def getPlanetmeta(jsondir = os.getcwd(),pattern = '**/*.json',outfile = './metadata.json'):
    dirpath = Path(jsondir)
    jsons = [p.resolve() for p in Path(dirpath).glob(pattern)]

    if len(jsons) == 0:
        raise FileNotFoundError(f"No JSON files found - check pattern matching ")
    saves = {}

    for i in jsons:

        with open(i) as j:
            dict = json.load(j)
            try:
                print(dict['properties'])
            except KeyError as k:
                print('irregular json, lacks following key: {}'.format(k))
                continue

            ID,time = i.name, dict['properties']['acquired']
            sun_az,sun_elev = dict['properties']['sun_azimuth'], dict['properties']['sun_elevation']
            sat_az,view_ang = dict['properties']['satellite_azimuth'], dict['properties']['view_angle']
            saves[ID] = {'time':time,'sun_az':sun_az,'sun_elev':sun_elev,'sat_az':sat_az,'view_ang':view_ang}

    with open(outfile, 'w') as ofile:
        json.dump(saves, ofile)

    return saves



def visualize_acquisition_stats(data,outfile = 'acquisition_geometry_distributions',title= "Acquisition Geometry Statistics"):
    # Visualize distributions of continuous variables from nested dict.
    #
    # Args:
    #     data: dict of form {timestamp: {'sun_az': x, 'sun_elev': x,
    #                                     'sat_az': x, 'view_ang': x}, ...}

    # --- Extract variables ---
    continuous_vars = ['sun_az', 'sun_elev', 'sat_az', 'view_ang']
    var_labels = {
        'sun_az': 'Sun Azimuth (°)',
        'sun_elev': 'Sun Elevation (°)',
        'sat_az': 'Satellite Azimuth (°)',
        'view_ang': 'View Angle (°)'
    }
    # Expected ranges for context lines on plots
    var_ranges = {
        'sun_az': (75, 175),
        'sun_elev': (40, 80),
        'sat_az': (0, 360),
        'view_ang': (0, 10)
    }

    # Pull values into arrays, skipping any missing keys
    values = {var: [] for var in continuous_vars}
    times = []
    for ID, obs in data.items():
        #get acquistions time from ID
        chunks = ID.split('_')
        acq = chunks[1]
        acqT = datetime.strptime(acq, '%H%M%S')
        Tval = acqT.hour + acqT.minute/60 + acqT.second/3600

        times.append(Tval)
        for var in continuous_vars: # gather continuous vars
            if var in obs and obs[var] is not None:
                values[var].append(obs[var])
    values = {var: np.array(vals) for var, vals in values.items()}
    times = np.array(times)
    print(times)

    # --- Compute summary stats ---
    stats = {}
    for var, arr in values.items():
        if len(arr) == 0:
            continue
        stats[var] = {
            'mean': np.mean(arr),
            'median': np.median(arr),
            'std': np.std(arr),
            'min': np.min(arr),
            'max': np.max(arr),
            'n': len(arr)
        }

    # set up plot layoiut
    fig = plt.figure(figsize=(14, 14))
    fig.suptitle('{}'.format(title),
                 fontsize=15, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(
        len(continuous_vars) + 1, 2,
        width_ratios=[3, 1],  # histogram wider than boxplot
        hspace=0.45,
        wspace=0.25
    )

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    for row, (var, color) in enumerate(zip(continuous_vars, colors)):
        arr = values[var]
        if len(arr) == 0:
            print(f"WARNING: No data for {var}, skipping")
            continue


        s = stats[var]
        label = var_labels[var]

        # --- Histogram ---
        ax_hist = fig.add_subplot(gs[row, 0])

        n_bins = min(30, max(10, len(arr) // 5))  # scale bins to sample size
        ax_hist.hist(arr, bins=n_bins, color=color, alpha=0.75, edgecolor='white')

        # Mean and ±1 std lines
        ax_hist.axvline(s['mean'], color='black', linewidth=1.5,
                        linestyle='-', label=f"Mean: {s['mean']:.1f}°")
        ax_hist.axvline(s['mean'] - s['std'], color='black', linewidth=1.0,
                        linestyle='--', label=f"±1 SD: {s['std']:.1f}°")
        ax_hist.axvline(s['mean'] + s['std'], color='black', linewidth=1.0,
                        linestyle='--')
        ax_hist.axvline(s['median'], color='red', linewidth=1.5,
                        linestyle=':', label=f"Median: {s['median']:.1f}°")

        ax_hist.set_xlabel(label)
        ax_hist.set_ylabel('Count')
        ax_hist.set_title(f"{label}  (n={s['n']})")
        ax_hist.legend(fontsize=8, loc='upper right')
        ax_hist.grid(True, alpha=0.3, axis='y')

        # Set x range to known physical range for context
        vmin, vmax = var_ranges[var]
        ax_hist.set_xlim(vmin, vmax)

        # --- Boxplot ---
        ax_box = fig.add_subplot(gs[row, 1])
        bp = ax_box.boxplot(
            arr,
            vert=True,
            patch_artist=True,
            widths=0.5,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor=color,
                            markersize=4, alpha=0.5)
        )
        ax_box.set_ylabel(label)
        ax_box.set_ylim(vmin, vmax)
        ax_box.set_xticks([])
        ax_box.grid(True, alpha=0.3, axis='y')

        # Annotate with stats
        stats_text = (
            f"mean:   {s['mean']:.1f}°\n"
            f"median: {s['median']:.1f}°\n"
            f"std:    {s['std']:.1f}°\n"
            f"min:    {s['min']:.1f}°\n"
            f"max:    {s['max']:.1f}°"
        )
        ax_box.text(
            1.08, 0.5, stats_text,
            transform=ax_box.transAxes,
            fontsize=8, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    #time plots
    time_hist = fig.add_subplot(gs[len(continuous_vars), :])

    n_bins = min(30, max(5, len(times) // 5))  # scale bins to sample size - no more than 30, no less than 5
    time_hist.hist(times, bins=n_bins, color='purple', alpha=0.75, edgecolor='white')



    time_hist.set_xlabel('Hours')
    time_hist.set_ylabel('Count')
    time_hist.set_title(f"{'Acquisition times'}  (n={len(times)})")
    time_hist.grid(True, alpha=0.3, axis='y')

    # Set x range to known physical range for context
    vmin, vmax = 0,24
    time_hist.set_xlim(vmin, vmax)
    time_hist.set_xticks(range(0,24))


    plt.savefig('{}.png'.format(outfile),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: {}.png".format(outfile))
    return stats


def print_stats_table(stats):
    #Print a clean summary table to console
    header = f"{'Variable':<12} {'N':>5} {'Mean':>8} {'Median':>8} " \
             f"{'Std':>8} {'Min':>8} {'Max':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for var, s in stats.items():
        print(f"{var:<12} {s['n']:>5} {s['mean']:>8.1f} {s['median']:>8.1f} "
              f"{s['std']:>8.1f} {s['min']:>8.1f} {s['max']:>8.1f}")
    print("=" * len(header) + "\n")

def find_outliers(data,stats,outfile,thresh = 2):
    #find outlier images based on stats
    valid_range = {}
    vars = []
    for var,vals in stats.items():
        minval,maxval = vals['mean'] - vals['std'], vals['mean'] + vals['std']
        valid_range[var] = minval,maxval
        vars.append(var)
    print(valid_range)
    outliers = {}
    magnitude = {}
    for id, vals in data.items():
        outliers[id] = {}
        magnitude[id] = {}
        for var in vars:
            vr = valid_range[var]
            val = vals[var]
            if vr[0] < val < vr[1]:
                outliers[id][var] = 1
                if val > vr[0]:
                    magnitude[id][var] = val - vr[1]
                else:
                    magnitude[id][var] = val - vr[0]
            else:
                outliers[id][var] = 0
    #save the outlier IDs
    with open(outfile, 'w') as ofile:
        json.dump(outliers, ofile,sort_keys=True, indent=4)
    count = 0
    #print findings
    anom = []
    for id in outliers.keys():

        outsum = sum(outliers[id].values())
        if outsum > thresh:
            print(f"{id}: {outsum} | {magnitude[id]}")
            anom.append(f"\n{id}: {outsum} | {magnitude[id]}")

            count += 1
    with open(outfile, 'a') as ofile:
        ofile.write(f"\n{count} files with more than {thresh} outlier acquisition variables")
        ofile.writelines(anom)


    print(f"{count} files with more than {thresh} outlier acquisition variables")

    return outliers

if __name__ == "__main__":
    directory = '/home/hopkinsonlab/Desktop/ACE_Basin_mapping/analysis/2025metadata/'

    dat = getPlanetmeta(directory)

    save = '2025_acqData'
    title = '2025 Acquisition Geometry — Variable Distributions'

    stats = visualize_acquisition_stats(dat,outfile = save,title = title)

    find_outliers(dat,stats,outfile = "outliers2025",thresh = 2)