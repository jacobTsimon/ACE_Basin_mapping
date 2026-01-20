import os
import json
import geojson
from tqdm import tqdm
from datetime import datetime
from NOAA_API_pull import tide_pull

#create a list of tide periods based on parameters given

#date format "YYYYMMDD"
start = "20200131"
end = "20250820"
thresh = 1.0 #maximum acceptable tide levels
station = "charleston"

#call on the tide script function to get tide windows
tide_pull(thresh=thresh,begindate=start,endate=end,station=station)

#bring in img geojson
with open("./data/collection_ortho_analytic_8b_sr.geojson") as f:
    imgs = geojson.load(f)

#add tides to csv files
tides = open("lowtide_periods.csv","r")
chosenfew = open("ideal_imgs.txt","w")
count = 0
#get the datetime periods:
for line in tqdm(tides):
    begin, end = line.split(sep=" - ")

    bDT = datetime.strptime(begin,"%Y-%m-%d %H:%M:%S")
    bDT.strftime("%Y-%m-%d %H:%M:%S.%f")
    eDT = datetime.strptime(end,"%Y-%m-%d %H:%M:%S ") #hidden space for some reason??
    eDT.strftime("%Y-%m-%d %H:%M:%S.%f")


    #loop over imgaes and compare dates
    for feat in imgs["features"]:
        #go over the metadata in features
        id = feat["id"]
        time = feat["properties"]["acquired"]
        dtime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ") #format into workable time
        dtime.strftime("%Y-%m-%d %H:%M:%S.%f")

        #print(id)


        if bDT < dtime and dtime < eDT:
            count += 1
            chosenfew.write("{}\n".format(id))
        else:
            continue

print(count)

