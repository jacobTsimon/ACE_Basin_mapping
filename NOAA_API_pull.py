#COOPS wrapper from: https://github.com/GClunies/noaa_coops
from noaa_coops import Station

#establish station codes in a dictionary
ACE_tide_stations = {
    'charleston' : '8665530', #for some reason ONLY charleston works currently?
    'mosquito_creek' : '8667209',
    'AC_cutoff_ICWW' : '8667482'
}


#begindate and enddate set the timeframe desired, thresh is the maximum acceptable tide level in meters
def tide_pull(thresh = 1.5,begindate = "20200131",endate = "20240131",station = "charleston"):
    ACE_tide_stations = {
        'charleston': '8665530',
        'mosquito_creek': '8667209',
        'AC_cutoff_ICWW': '8667482'
    }
    ID = ACE_tide_stations[station]
    print(ID)
    # select the station desired
    station = Station(id=ID)

    print(station.get_data_inventory)
    #grab data according to set params
    bigpull = station.get_data(begin_date=begindate,
        end_date=endate,
        product="hourly_height",
        datum="MLLW",
        units="metric",
        time_zone="gmt")


    bigpull=bigpull.drop(["s","f"], axis =1)
    lowtides = bigpull.mask(bigpull["v"] >= thresh,1000.0) #mark times above the threshhold with arbitrary value, used 1000.0 here
    #record the times in a csv
    lowtides.to_csv("test_tides.csv", sep=" ")

    #make coherent lowtide periods:
    outfile = open("lowtide_periods.csv","w")
    #somewhat convoluted loop, but it works...
    mark = True
    for index,line in lowtides.iterrows():
        if index == 0:
            continue
        if line["v"] != 1000.0 and mark == True: #identifies the start of a low tide period
            first_DT = line.name
            mark = False
        if line["v"] != 1000.0 and mark == False: #current time in the period
            last_DT = line.name
        if line["v"] == 1000.0 and mark == False: # detects when times stop and markers begin (1000.0), records the interval and resets
            print("FIRST: {} \n LAST: {}".format(first_DT,last_DT))
            outfile.write("{} - {}\n".format(first_DT,last_DT))
            mark = True
        if line["v"] == 1000.0 and mark ==True: #skips high tide periods (1000.0)
            continue

    outfile.close()