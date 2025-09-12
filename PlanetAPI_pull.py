import os
import json
import requests
import yaml
from dateutil.parser import parse

# Helper function to printformatted JSON using the json module
def p(data):
    print(json.dumps(data, indent=2))

#load in API key from dedicated yaml file (github privacy)
f = open("PL_API_KEY.yaml","r")
keyF = yaml.safe_load(f)
key = keyF["key"]


# if your Planet API Key is not in a yaml file, you can paste it below
if keyF:
    API_KEY = key
else:
    API_KEY = "paste key here"

 # construct authorization tuple for use in the requests library
BASIC_AUTH = (API_KEY, '')

# Setup Planet Data API base URL
URL = "https://api.planet.com/data/v1"

# Setup the session
session = requests.Session()

# Authenticate
session.auth = (API_KEY, "")

# Make a GET request to the Planet Data API
res = session.get(URL)

# Response status code
res.status_code

# Print formatted JSON response to check validity
p(res.json())

# Print the value of the item-types key from _links
print(res.json()["_links"]["item-types"])

# Setup the stats URL
stats_url = "{}/stats".format(URL)

# Print the stats URL
print(stats_url)


# Create filter object for all imagery captured between 2013-01-01 and present.
#whole filter includes the DateRange, type of asset desired, a geometry filter to only find images
# within my ROI and a cloud cover filter set to <= 10%
whole_filter = {
   "type":"AndFilter",
   "config":[
       {
           "type": "DateRangeFilter",  # Type of filter -> Date Range
           "field_name": "acquired",  # The field to filter on: "acquired" -> Date on which the "image was taken"
           "config": {
               "gte": "2013-01-01T00:00:00.000Z",  # "gte" -> Greater than or equal to
           }
       },
       {
            "type":"AssetFilter",
            "config":[
               "ortho_analytic_8b_sr"
            ]
         },
       {
           "type": "GeometryFilter",
           "field_name": "geometry",
           "config": {
               "type": "Polygon",
               "coordinates": [
                   [
                       [
                           -80.35116161549101,
                           32.63351701860242
                       ],
                       [
                           -80.34391918342268,
                           32.64266523177115
                       ],
                       [
                           -80.37590659172612,
                           32.67772471402955
                       ],
                       [
                           -80.66982862651581,
                           32.57809952715948
                       ],
                       [
                           -80.61369977798283,
                           32.54503608508628
                       ],
                       [
                           -80.60042198585731,
                           32.535877901081875
                       ],
                       [
                           -80.57507347361661,
                           32.52722764750513
                       ],
                       [
                           -80.56783104154754,
                           32.50687082419701
                       ],
                       [
                           -80.56058860947921,
                           32.48854574165249
                       ],
                       [
                           -80.55032849738151,
                           32.47327198791791
                       ],
                       [
                           -80.55153556939342,
                           32.46257881821019
                       ],
                       [
                           -80.57024518557056,
                           32.45748638635327
                       ],
                       [
                           -80.56783104154754,
                           32.4534122335854
                       ],
                       [
                           -80.55153556939342,
                           32.45595860065674
                       ],
                       [
                           -80.53161888120438,
                           32.44984719877952
                       ],
                       [
                           -80.52497998514083,
                           32.45188437879749
                       ],
                       [
                           -80.50385622494066,
                           32.44730065898996
                       ],
                       [
                           -80.48212892873414,
                           32.44679134239681
                       ],
                       [
                           -80.47609356867694,
                           32.443735382391395
                       ],
                       [
                           -80.46945467261416,
                           32.43405749214398
                       ],
                       [
                           -80.46583345658004,
                           32.420812379898834
                       ],
                       [
                           -80.45979809652205,
                           32.411132029392064
                       ],
                       [
                           -80.45496980847676,
                           32.40960345806043
                       ],
                       [
                           -80.44953798442513,
                           32.40552714127041
                       ],
                       [
                           -80.4646263845681,
                           32.389729674999444
                       ],
                       [
                           -80.4785077127,
                           32.38055568441176
                       ],
                       [
                           -80.48876782479766,
                           32.378007188358325
                       ],
                       [
                           -80.49480318485486,
                           32.372909980617266
                       ],
                       [
                           -80.48152539272857,
                           32.35047885147617
                       ],
                       [
                           -80.45798748850538,
                           32.33773141290274
                       ],
                       [
                           -80.45315920045928,
                           32.33824134492707
                       ],
                       [
                           -80.43263897626467,
                           32.378516893319954
                       ],
                       [
                           -80.42056825615029,
                           32.39584515110457
                       ],
                       [
                           -80.33607321534802,
                           32.476326946020436
                       ],
                       [
                           -80.34633332744573,
                           32.49363641685501
                       ],
                       [
                           -80.3306413912964,
                           32.49210924453996
                       ],
                       [
                           -80.31676006316454,
                           32.499744846855776
                       ],
                       [
                           -80.35116161549101,
                           32.63351701860242
                       ]
                   ]
               ]
           }
       },
        {
           "type":"RangeFilter",
           "field_name":"cloud_cover",
           "config":{
              "lte":0.1
           }
        }
   ]
}


# Specify the sensors/satellites or "item types" to include in our results + the assets from those items
item_types = ["PSScene"]
asset_types = ["ortho_analytic_8b_sr"]

#formulate the request
request2 = {
    "asset_types" : asset_types,
    "item_types" : item_types,
    "filter" : whole_filter
}

# Setup the quick search endpoint url
quick_url = "{}/quick-search".format(URL)

# Send the POST request to the API quick search endpoint
res2 = session.post(quick_url, json=request2)

# Assign the response to a variable
geojson = res2.json()

#function for getting next link taken from their git repo:
#https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/Analytics-API/quickstart/02_fetching_feed_results.ipynb
#I modified it since the syntax they used seemed to be out of date (uses "rel" and links without the _)
def get_next_link(results_json):
    """Given a response json from one page of subscription results,
    get the url for the next page of results.

    Args:
        results_json (dict): The response JSON containing the subscription results.

    Returns:
        str or None: The URL for the next page of results if available, None otherwise.
    """
    for link in results_json['_links']:

        if link == "_next":
            return results_json['_links']["_next"]
    return None

#prep geojson file for comparison
filedump = "ortho_analytic_8b_sr"

#set up collection dict
feature_collection = {'type': 'FeatureCollection', 'features': []}
# initial link = quick search
next_link = quick_url
#post the request + get the results
geojson = session.post(next_link, json=request2).json()
results = session.get(geojson["_links"]["_first"])

#set up page var
page = 0
# Fetch features iteratively until there are no more next links (this code also from their git repo, posted above)
while next_link:
    if page == 0:
        next_features = results.json()['features']
        #move past the first page
        page += 1
        # get the difference in dates and print them
        latest_feature_creation = parse(next_features[0]["properties"]['acquired']).date()
        earliest_feature_creation = parse(next_features[-1]["properties"]['acquired']).date()
        print('Fetched {} features fetched ({}, {})'.format(
            len(next_features), earliest_feature_creation, latest_feature_creation))
        #add features
        feature_collection['features'].extend(next_features)
        next_link = get_next_link(results.json())
        print(page)
    if page != 0:
        #get next page + features
        results = session.get(next_link)
        next_features = results.json()['features']
        # Check if there are next features available
        if next_features:
            # get the difference in dates and print them
            latest_feature_creation = parse(next_features[0]["properties"]['acquired']).date()
            earliest_feature_creation = parse(next_features[-1]["properties"]['acquired']).date()
            print('Fetched {} features fetched ({}, {})'.format(
                len(next_features), earliest_feature_creation, latest_feature_creation))
            # add features
            feature_collection['features'].extend(next_features)
            next_link = get_next_link(results.json())
            page += 1
            print(page)

        # No next features available
        else:
            next_link = None
#give a total feature count at the end
print('Total features: {}'.format(len(feature_collection['features'])))
#make a folderto put the resulting file in
os.makedirs('Planet_img_metadata', exist_ok=True)
filename = 'Planet_img_metadata/collection_{}.geojson'.format(filedump)
with open(filename, 'w') as file:
    json.dump(feature_collection, file)

