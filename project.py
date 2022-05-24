#!/usr/bin/env python
# coding: utf-8

# # FIT5196 Assessment 3
# 
# #### Student Name: Ibrahim Al-Hindi
# #### Date Created: 10/10/2021
# #### Date Last Edited: 1/11/2021
# 
# #### Environment Details:
# - Python: 3.8.10
# - Anaconda: 4.10.3
# 
# #### Libraries Used:
# - numpy
# - pandas
# - re
# - shapefile
# - matplotlib
# - ast
# - haversine
# - datetime
# - urllib
# - bs4
# - sklearn
# - statsmodels
# - math
# - scipy
# 
# ## Table of Contents:
# 1. Introduction
# 2. Load and Parse Data Files
# 3. Integrate Column Values
#     - 3.1 Suburb
#     - 3.2 LGA
#     - 3.3 closest_train_station_id and distance_closest_train_station
#     - 3.4 travel_min_to_MC
#     - 3.5 direct_journey_flag
#     - 3.6 COVID Cases
# 4. Data Reshaping
#     - 4.1 Data Cleaning and Exploration
#     - 4.2 Initial Model
#     - 4.3 Normalisation
#         - 4.3.1 Standardisation
#         - 4.3.2 Minmax Normalisation
#     - 4.4 Transformation
#         - 4.4.1 Root Transformation
#         - 4.4.2 Square Power Transformation
#         - 4.4.3 Log Transformation
#         - 4.4.4 Box Cox Transformation
#     - 4.5 Final Model
# 5. Conclusion
# 6. References
# 
# ## 1. Introduction
# 
# This assignment is concerned with techniques used to integrate several datasets of varying forms into a single schema. The final schema will include a list of unique residential properties and their attributes, such as their suburb, LGA, closest train station, and others. It will also include COVID-19 figures that will be retrieved through webscraping. The second part of the assignment will explore several normalisation and transformation methods in an attempt to produce the best linear model in relation to the COVID-19 columns.
# 
# ## 2. Load and Parse Data Files
# 
# The property data is contained in two files: a json file and an xml file. We will read both of them in and then concatenate the two dataframes

# In[ ]:


import numpy as np
import pandas as pd
import re
import shapefile
from matplotlib.patches import Polygon
from ast import literal_eval
from haversine import haversine, Unit
import datetime as dt
from urllib.request import urlopen
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing
import math
from scipy import stats


# In[ ]:


# load json file
json_df = pd.read_json("data/jsonfile.json")
json_df.head()


# In[ ]:


# load xml file
with open('data/xmlfile.xml','r') as infile:
    xml_text = infile.read()
    
xml_text.split("\n")[:30]


# Looking at the first few properties of the xml file, we can see that each property has a root `<property>` tag, followed by `<property>`, `<lat>`, `<lng>`, and `<addr_street>` tags

# To ascertain the total number of properties in the file, regex will be used to count the number of times `<property>` appears in the file

# In[ ]:


len(re.findall(r"<property>", xml_text))


# We have 1,024 properties in the file.
# 
# For each of the tags in the file, a list will be created that holds all the values corresponding to each tag. This will be done using regex where for each tag, all the characters between the opening and closing tags will be matched as so:
# 
# `<tag>(.*)?</tag>`
# 
# The list corresponding to each tag will then be used to build a dataframe

# In[ ]:


# create regex patterns
id_pattern = r"<property_id>(.*?)</property_id>"
lat_pattern = r"<lat>(.*?)</lat>"
lng_pattern = r"<lng>(.*?)</lng>"
addr_pattern = r"<addr_street>(.*?)</addr_street>"

# build dataframe
xml_df = pd.DataFrame({"property_id": [prop_id for prop_id in re.findall(id_pattern, xml_text)],
                       "lat": [lat for lat in re.findall(lat_pattern, xml_text)],
                       "lng": [lng for lng in re.findall(lng_pattern, xml_text)],
                       "addr_street": [addr for addr in re.findall(addr_pattern, xml_text)]})

xml_df.head()


# The process of concatenating the two dataframes will now be undertaken. First we need to ensure the column types for both dataframes are the same 

# In[ ]:


print("json_df:")
json_df.info()
print("\n")
print("xml_df:")
xml_df.info()


# The two dataframes have different data types. We will convert `xml_df` to be the same as `json_df`

# In[ ]:


xml_df = xml_df.astype({"property_id": "int64", "lat": "float64", "lng": "float64"})

xml_df.info()


# We will now concatenate the two dataframes along the rows, we will also set `ignore_index` to **True** so that the indexes are not repeated from both dataframes. We will also change the type of `property_id` to object, as well as round the `lat` and `lng` values to seven decimal places to avoid rounding discrepancies

# In[ ]:


prop_df = pd.concat([json_df, xml_df], ignore_index = True)
prop_df["property_id"] = prop_df["property_id"].astype("object")
prop_df["lat"] = prop_df["lat"].round(7)
prop_df["lng"] = prop_df["lng"].round(7)
prop_df.head()


# Let's take a look at the summary statistics of the data

# In[ ]:


prop_df.describe(include = "all")


# We can see there are duplicates in `property_id` and `addr_street`. Let's dive deeper into these duplicates. First we'll extract the rows where all the values are the same

# In[ ]:


prop_df[prop_df.duplicated(keep = False)].sort_values("addr_street")


# We will drop these duplicates

# In[ ]:


prop_df.drop_duplicates(inplace = True)
prop_df.describe(include = "O")


# We still have duplicates in `addr_street`

# In[ ]:


prop_df[prop_df.duplicated("addr_street", keep = False)].sort_values("addr_street")


# Searching online indicates that identical street names do exist in different suburbs, which is also indicated by the different `lat` and `lng` for each duplicated address, therefore the "duplicates" are acceptable

# Our final dataframe contains the following statistics

# In[ ]:


prop_df.describe(include = "all")


# ## 3. Integrate Column Values
# 
# In this section we will integrate data from multiple sources into our dataframe.
# 
# ### 3.1 Suburb
# 
# The first column we will integrate is the `suburb` column which will hold the suburb name of each property. To determine the suburb of each property in `prop_df`, a shapefile of the suburb boundaries will be loaded. Then, for each property in `prop_df`,
# the suburb name from the loaded shapefile will be returned where the `lat` and `lng` from `prop_df` are inside the boundary coordinates of the suburb in the shapefile. Reading and utlising the shapefile will be done using the `shapefile` package and the `Polygon` module from the `matplotlib.patches` package.

# In[ ]:


# load shapefile
sf = shapefile.Reader("data/vic_suburb_bounadry/VIC_LOCALITY_POLYGON_shp")
# read records and shapes
shaperecs = sf.shapeRecords()


# For each record-shape combination in `shaperecs`, the records are generated using the `record` attribue while the shape is accessed using the `shape` attribute. The suburb is the 7th element in each record. This is illustrated below

# In[ ]:


print(shaperecs[0].record, "\n")
print("suburb:", shaperecs[0].record[6])


# A dictionary `subs_bounds` will be created where keys are the suburb names and the values are the polygons for each suburb

# In[ ]:


subs_bounds = {}

for rs in shaperecs:
    sub = rs.record[6]
    shape = rs.shape
    ptchs = []
    pts = np.array(shape.points)
    prt = shape.parts
    par = list(prt) + [pts.shape[0]]
    
    for pij in range(len(prt)):
         ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))
            
    subs_bounds[sub] = ptchs[0]
    
subs_bounds


# A function called `loc_sub` will be defined where a longitude and latitude will be checked against the polygons in `subs_bounds`, and returns the suburb name that holds the specified longitude and latitude using the `contains_point` function. The function will take a row as an argument. This function will then be applied to `prop_df` to create the `suburb` column

# In[ ]:


def loc_sub(row):
    # retrieve row latitude
    lat = row["lat"]    
    # retrieve row longitude
    lng = row["lng"]
    
    # check if any suburb contains the point
    for sub, poly in subs_bounds.items():
        if poly.contains_point((lng, lat)):
            return sub  
        
    # if no suburb contains the point    
    return "not available"


# In[ ]:


# create suburb column by applying loc_sub
prop_df["suburb"] = prop_df.apply(loc_sub, axis = 1)
prop_df.head()


# The following addresses do not contain a suburb name

# In[ ]:


prop_df[prop_df["suburb"] == "not available"]


# ### 3.2 LGA
# 
# The next column to be integrated is the `lga` column. Each `lga` contains multiple suburbs, therefore the `lga` to be returned for each row will be the `lga` that holds the specified property's `suburb`. 
# 
# The list of LGA's and their corresponding suburbs is held in a pdf file `lga_to_suburb`. `pdf_miner` will be used to convert the pdf file into a text file

# In[ ]:


# convert pdf to text file using pdf_miner
get_ipython().system('pdf2txt.py -o lga_to_suburb.txt data/lga_to_suburb.pdf')


# In[ ]:


# read in the text file
with open("lga_to_suburb.txt", "r") as infile:
    lga_text = infile.readlines()


# Examine the first and last five lines in the text file

# In[ ]:


# first lines
lga_text[:5]


# In[ ]:


# last lines
lga_text[-5:]


# The last two lines in the file will be removed

# In[ ]:


lga_text = lga_text[:-2]


# A dictionary `lga_dict` will be created where the keys are the names of the LGA's and the values are the list of the suburbs that LGA holds. These items will be extracted from the `lga_text` file. The LGA and the list of suburbs it contains are separated by ":", which will be used to split them apart

# In[ ]:


lga_dict = {}

for line in lga_text:
    if line != "\n":
        # extract LGA and suburb from each line
        lga, suburbs = line.strip().split(" : ")
        # convert string that holds the list of suburbs to a pure list using literal_eval from the ast package
        suburbs = literal_eval(suburbs)
        # convert suburbs to all capital letters to match suburb name in prop_df
        suburbs = [suburb.upper() for suburb in suburbs]
        lga_dict[lga] = suburbs
        
lga_dict


# A function `find_lga` will be defined where a given suburb will be checked against the suburbs in `lga_dict` and returns the name of the `lga` that suburb is in. The function will then be applied to the `suburb` column in `prop_df` to create the `lga` column

# In[ ]:


def find_lga(suburb):
    for lga, suburbs in lga_dict.items():
        if suburb in suburbs:
            return lga
        
    return "not available"


# In[ ]:


# apply find_lga
prop_df["lga"] = prop_df["suburb"].apply(find_lga)
prop_df.head()


# The following properties do not contain a `lga` name

# In[ ]:


prop_df[prop_df["lga"] == "not available"]


# ### 3.3 closest_train_station_id and distance_to_closest_train_station
# 
# The next columns to be added are `closest_train_station_id` and `distance_to_closest_train_station`.
# 
# The file containing the stop id's and their coordinates will be read in. In order to maximise the accuracy of the desired columns, the file will be read as a text file rather than as a csv to maintain the maximum possible decimal points for each stop's latitude and longitude.

# In[ ]:


with open("data/Vic_GTFS_data/metropolitan/stops.txt", "r") as infile:
    stops = infile.readlines()


# The first five lines of the file will be examined

# In[ ]:


stops[:5]


# In[ ]:


# remove top row containing field names
stops = stops[1:]
stops[:5]


# A dictionary `stop_dict` will be created where the keys will be the station id and the values will be the latitude and longitude coordinates of that stop.
# 
# Each line in `stops` will be split on ",". The stop id is the 1st element of the resulting list, while the latitude and longitude are the 4th and 5th elements of the list respectively

# In[ ]:


stops_dict = {}

for stop in stops:
    stop_info = stop.split(",")
    stop_id = int(literal_eval(stop_info[0]))
    stop_lat = float(literal_eval(stop_info[3]))
    stop_lng = float(literal_eval(stop_info[4]))
    
    stops_dict[stop_id] = (stop_lat, stop_lng)
    
stops_dict


# To find the closest train station, a function `closest_station` will be defined that accepts a row as input, it then calculates the haversine distance using the `haversine` package between each property and all the stops, and returns the stop with the shortest distance.
# 
# Similarly, `closest_distance` will be defined that peforms the same function as `closest_station` but it returns the distance in KM instead.
# 
# The functions will be applied to `prop_df` to generate the desired columns

# In[ ]:


def closest_station(row):
    # property coordinates
    prop_coords = (row["lat"], row["lng"])
    dist_dict = {}    
    
     # calculate haversine distance for each stop
    for stop, coords in stops_dict.items():
        dist_dict[stop] = haversine(prop_coords, coords)
        
    # extract stop with the shortest distance    
    for sub_stop, dist in dist_dict.items():
        if dist == min(dist_dict.values()):
            return sub_stop
        
def closest_distance(row):
    # property coordinates
    prop_coords = (row["lat"], row["lng"])
    dist_dict = {}    
    
     # calculate haversine distance for each stop
    for stop, coords in stops_dict.items():
        dist_dict[stop] = haversine(prop_coords, coords)
        
    # extract stop with the shortest distance    
    for sub_stop, dist in dist_dict.items():
        if dist == min(dist_dict.values()):
            return round(dist, 3)


# In[ ]:


prop_df["closest_train_station_id"] = prop_df.apply(closest_station, axis = 1)
prop_df["distance_to_closest_train_station"] = prop_df.apply(closest_distance, axis = 1)
prop_df.head()


# ### 3.4 travel_min_to_MC
# 
# The next column to be integrated is the `travel_min_to_MC`. For each property, this column holds the rounded average travel time (minutes) of the direct journeys from the closest train station to the Melbourne Central station on weekdays (i.e. Monday-Friday) departing between 7 to 9am. If there are no direct journeys between the closest station and Melbourne Central station, the value will be set to "not available". If the closest station to a property is Melbourne Central station itself, then the value will be set to 0.
# 
# Direct journey means that you can reach the Melbourne Central station without changing your train at any point in the journey. So, when you board the train on the closest station, you can directly go to the Melbourne Central station.
# 
# To generate the travel times, three data files will be used:
# 1. `calendar`
# 2. `trips`
# 3. `stop_times`
# 
# The dataframe `calendar` displays the operating days for each `service_id`. This dataframe will be filtered to only include `service_id`(s) that run on all the weekdays

# In[ ]:


# read in calendar
calendar = pd.read_csv("data/Vic_GTFS_data/metropolitan/calendar.txt")
calendar


# In[ ]:


# filter services_id's to include those that run on all weekdays
calendar = calendar[((calendar["monday"] == 1) & (calendar["tuesday"] == 1) & (calendar["wednesday"] == 1) & (calendar["thursday"] == 1) & (calendar["friday"] == 1))]
calendar


# The `trips` dataframe displays the `trip_id` for each `service_id` in `calendar`. We will filter `trips` to only include `service_id`'s that exist in `calendar`

# In[ ]:


# read in trips
trips = pd.read_csv("data/Vic_GTFS_data/metropolitan/trips.txt")
trips.head()


# In[ ]:


# filter to only include service_id's in calendar
trips = trips[trips["service_id"].isin(calendar["service_id"])]
trips.head()


# The `stop_times` dataframe displays the `arrival_time`, `departure_time` and `stop_id` for each `trip_id`. `stop_times` will be filtered to only contain the `trip_id`'s contained in `trips`. 

# In[ ]:


# read in stop_times
stop_times = pd.read_csv("data/Vic_GTFS_data/metropolitan/stop_times.txt")
stop_times.head()


# In[ ]:


# filter to only include trip_id's present in trips
stop_times = stop_times[stop_times["trip_id"].isin(trips["trip_id"])]
stop_times.head()


# The `departure_time` column will be filtered to only contain times between 7:00:00 and 23:59:59

# In[ ]:


stop_times = stop_times[((stop_times["departure_time"] >= "07:00:00") & (stop_times["departure_time"] < "24:00:00")) & (stop_times["arrival_time"] < "24:00:00")]
stop_times.head()


# The `arrival_time` and `departure_time` columns will be converted to datetime

# In[ ]:


stop_times.loc[:, 'arrival_time'] = pd.to_datetime(stop_times['arrival_time'],format= '%H:%M:%S')
stop_times.loc[:, 'departure_time'] = pd.to_datetime(stop_times['departure_time'],format= '%H:%M:%S')
stop_times.head()


# We can see that `arrival_time` and `departure_time` now contain dates, but for our purposes it is of no concern. Only the time matters.
# 
# We can now calculate the time difference in minutes between two times like so [(source)](https://www.datasciencemadesimple.com/difference-two-timestamps-seconds-minutes-hours-pandas-python-2/):

# In[ ]:


depart_time = stop_times.loc[8911, "departure_time"]
arrive_time = stop_times.loc[8912, "arrival_time"]
(arrive_time - depart_time) / np.timedelta64(1,'m')


# A series containing the `trip_id`'s as index and that trip's stops as values will be created. This will be done by grouping `stop_times` by `trip_id` and retrieving the unique `stop_id`'s for that `trip_id`

# In[ ]:


trip_stops = stop_times.groupby("trip_id")["stop_id"].unique()
trip_stops


# `trips_stops` will be converted into a dictionary `trip_dict`

# In[ ]:


trip_dict = trip_stops.to_dict()
# convert series of stops into a list
trip_dict = {trip: list(stops) for trip, stops in trip_dict.items()}
trip_dict


# A function `melb_cen_time` will be defined that accepts a stop as input, and returns one of:
# 1. **0** if the stop is Melbourne Central
# 2. **average journey time** if Melbourne Central can be reached in a single trip and the departure time is between 7am and 9am
# 3. **not available** if neither of the above are satisfied
# 
# First we need the `stop_id` for Melbourne Central. Using the `stops` file in the previous sections, we can retrieve the `stop_id` corresponding to Melbourne Central:

# In[ ]:


for line in stops:
    if "Melbourne Central" in line:
        print(line.split(",")[0])


# Melbourne Central's stop is **19842**

# In[ ]:


def melb_cen_time(stop):
    times = []
    routes = []
    
    # closest station is Melbourne Central
    if stop == 19842:
        return 0
    
    for route, stops_list in trip_dict.items():
        # check both stops in the route and Melbourne Central's stop comes after the closest station stop
        if (stop in stops_list) and (19842 in stops_list) and (stops_list.index(stop) < stops_list.index(19842)):
            routes.append(route)

    # if no routes contain both stops, or if Melbourne Central comes before our stop, then there are no direct journeys
    if len(routes) == 0:
        return "not available"
    
    for r in routes:
        # check departure time from the stop is between 7am and 9am  
        if stop_times.loc[(stop_times["trip_id"] == r) & (stop_times["stop_id"] == stop), "departure_time"].dt.hour.values[0] <= 9:                       
            depart_time = stop_times.loc[(stop_times["trip_id"] == r) & (stop_times["stop_id"] == stop), "departure_time"].values[0]
            arrive_time = stop_times.loc[(stop_times["trip_id"] == r) & (stop_times["stop_id"] == 19842), "arrival_time"].values[0]
            time = (arrive_time - depart_time) / np.timedelta64(1,'m')
            times.append((time))
    
    # none of the potential routes satisfy the conditions
    if len(times) == 0:
        return "not available"
    
    else:
        return round(np.mean(times))


# Instead of applying the function to each row in `prop_df`, to save time, we will create a dictionary `stop_to_MC_times` that will contain the time according to each stop by applying `melb_cen_time` to the unique values of `closest_train_station_id` in `prop_df`

# In[ ]:


stop_to_MC_times = {stop: melb_cen_time(stop) for stop in prop_df["closest_train_station_id"].unique()}


# Using a lambda a function, the `travel_min_to_MC` will be created by extracting the time that matches the `closest_train_station_id` in `stop_to_MC_times`

# In[ ]:


prop_df["travel_min_to_MC"] = prop_df["closest_train_station_id"].apply(lambda x: stop_to_MC_times[x])
prop_df.head()


# ### 3.5 direct_journey_flag
# 
# This column indicates whether there is a direct journey to the Melbourne Central station from the closest station between 7-9am on the weekdays. The value is 1 if there is a direct trip (i.e. no transfer between trains is required to get from the closest train station to the Melbourne Central station) and 0 otherwise.
# 
# To generate the values in the column, a lambda function will be applied where if `travel_min_to_MC` is equal to **not available** then 0 is returned, otherwise it will return 1

# In[ ]:


prop_df["direct_journey_flag"] = prop_df["travel_min_to_MC"].apply(lambda x: 0 if x == "not available" else 1)
prop_df.head()


# ### 3.6 COVID Cases Columns
# 
# The final columns to be integrated are the COVID cases columns. The columns will hold the following values for each property according to its `lga`:
# - `30_sep_cases`: The number of Covid-19 positive cases on the 30th of September
# - `last_14_days_cases`: The rounded average of Covid-19 cases for the last 14 days starting from 29th of September backward
# -`last_30_days_cases`: The rounded average of Covid-19 cases for the last 30 days starting from 29th of September backward
# - `last_60_days_cases`: The rounded average of Covid-19 cases for the last 60 days starting from 29th of September backward
# 
# The COVID figures will be retrieved from the website [covidlive.com.au](https://covidlive.com.au/).
# 
# The figures for a given `lga` can be accessed using the following URL:
# 
# **https://covidlive.com.au/vic/** followed by the name of the `lga`
# 
# The URL will be opened using `urlopen` from the `urllib` package. The HTML object will then be parsed using `BeautifulSoup` from the `bs4` package. To retrieve the dates, all the **td** tags with the class attribute **COL1 DATE** will be found. Similarly, to retrieve the cases, all the **td** tags with the class attribute **COL4 CASES** will be found. The dates and cases will then be joined together into a dataframe.
# 
# The cases in the URL are presented in a cumulative manner from day to day. Therefore the cases from two dates will be differenced to calculate each figure. The dates needed to calculate the desired values for the columns are:
# - August 01: to calculate 60 day average
# - August 31: to calculate 30 day average
# - September 16: to calculate 14 day average
# - September 29: to calculate all
# - September 30: to calculate September 30 cases
# 
# To illustrate, the process described will be implemented on the `lga` **Maribyrnong**

# In[ ]:


html = urlopen("https://covidlive.com.au/vic/maribyrnong")
bsObj = BeautifulSoup(html, "html.parser")

dates_tags = bsObj.find_all("td", "COL1 DATE")
dates_tags


# To retrieve the text from each tag, `get_text` will be used on each tag

# In[ ]:


dates = [date.get_text() for date in dates_tags]
dates


# In[ ]:


cases_tags = bsObj.find_all("td", "COL4 CASES")
cases = [case.get_text() for case in cases_tags]
cases


# The string case numbers will be converted to integers

# In[ ]:


numbers = [int(num.replace(",", "")) for num in cases]
numbers


# The dates and case numbers will be placed in a dataframe, the case numbers for the required dates will then be retrieved

# In[ ]:


df = pd.DataFrame({"dates": dates, "cases": numbers})
    
sep_30 = df.loc[df["dates"] == "30 Sep", "cases"].values[0]
sep_29 = df.loc[df["dates"] == "29 Sep", "cases"].values[0]
sep_16 = df.loc[df["dates"] == "16 Sep", "cases"].values[0]
aug_31 = df.loc[df["dates"] == "31 Aug", "cases"].values[0]
aug_01 = df.loc[df["dates"] == "01 Aug", "cases"].values[0]

print("Sep 30:", sep_30)
print("Sep 29:", sep_29)
print("Sep 16:", sep_16)
print("Aug 31:", aug_31)
print("Aug 01:", aug_01)


# The average case numbers per period can now be calculated

# In[ ]:


print("Sep 30 cases:        ", sep_30 - sep_29)
print("14 days average cases:", round((sep_29 - sep_16) / 14))
print("30 days average cases:", round((sep_29 - aug_31) / 30))
print("60 days average cases:", round((sep_29 - aug_01) / 60))


# The process described above will be implemented on all the LGA's. The results will be stored in a dictionary `cases_dict`

# In[ ]:


cases_dict = {}

for lga in prop_df["lga"].unique():
    # replace space in name with "-"
    if " " in lga:
        lga_clean = lga.replace(" ", "-")
    
    else:
        lga_clean = lga
        
    html = urlopen("https://covidlive.com.au/vic/" + lga_clean.lower())
    bsObj = BeautifulSoup(html, "html.parser")
    
    dates_tags = bsObj.find_all("td", "COL1 DATE")
    dates = [date.get_text() for date in dates_tags]
    
    cases_tags = bsObj.find_all("td", "COL4 CASES")
    cases = [case.get_text() for case in cases_tags]
    numbers = [int(num.replace(",", "")) for num in cases]
    
    if len(dates) > 0 and len(numbers) > 0:
        df = pd.DataFrame({"dates": dates, "cases": numbers})
    
        sep_30 = df.loc[df["dates"] == "30 Sep", "cases"].values[0]
        sep_29 = df.loc[df["dates"] == "29 Sep", "cases"].values[0]
        sep_16 = df.loc[df["dates"] == "16 Sep", "cases"].values[0]
        aug_31 = df.loc[df["dates"] == "31 Aug", "cases"].values[0]
        aug_01 = df.loc[df["dates"] == "01 Aug", "cases"].values[0]
    
        cases_dict[lga] = {"sep_30_cases": sep_30 - sep_29, "fortnight_avg": round((sep_29 - sep_16) / 14),
                          "month_avg": round((sep_29 - aug_31) / 30), "two_month_avg": round((sep_29 - aug_01) / 60)}
        
cases_dict


# For each COVID column, the value will be arrived at by applying a lambda function which retrieves the cases of the required period corresponding to the LGA in `cases_dict`, if the `lga` is not "not available"

# In[ ]:


prop_df["30_sep_cases"] = prop_df["lga"].apply(lambda x: cases_dict[x]["sep_30_cases"] if x != "not available" else "not available")
prop_df["last_14_days_cases"] = prop_df["lga"].apply(lambda x: cases_dict[x]["fortnight_avg"] if x != "not available" else "not available")
prop_df["last_30_days_cases"] = prop_df["lga"].apply(lambda x: cases_dict[x]["month_avg"] if x != "not available" else "not available")
prop_df["last_60_days_cases"] = prop_df["lga"].apply(lambda x: cases_dict[x]["two_month_avg"] if x != "not available" else "not available")
prop_df.head()


# The property dataframe `prop_df` is now complete

# In[ ]:


# save prop_df as csv
#prop_df.to_csv("solution.csv", index = False)


# ## 4. Data Reshaping
# 
# In this section, the effect of different normalisation and transformation techniques on the COVID columns generated above will be examined, assuming that we want to develop a linear model to predict the `30_sep_cases` using `last_14_days_cases`, `last_30_days_cases`, and `last_60_days_cases` attributes. There are two aims: the first is that we want out features to be on the same scale and second, we want our features to have as much linear relationship as possible with the predicted variable (i.e., `30_sep_cases`). Furthermore, we will create and assess linear models along the way to ascertain if normalisation/transformation is required and if so, what type? This will be done by exploring the model diagnositcs. It is important to note that in this task all the explanatory variables will be used, in other words none of the variables will be removed in an attempt to enhance the linear model.
# 
# ## 4.1 Data Cleaning and Exploration
# 
# The data will be filtered to only contain the columns necessary. Furthermore, the data will undergo several cleaning steps to arrive at our final dataframe; these include removing rows with no COVID case figures availabe, removing duplicate `lga`s since each `lga` will have the same number of cases when the same `lga` is repeated in the dataframe, and removing rows where all the values are zero since they add no value to our analysis and might affect the models to be built

# In[ ]:


# select necessary columns
covid = prop_df.iloc[:, [5, 10, 11, 12, 13]].copy()
# remove rows with "not available"
covid = covid[covid["30_sep_cases"] != "not available"]
# drop duplicate LGA's
covid.drop_duplicates("lga", inplace = True)
covid.drop(columns = "lga", inplace = True)
# adjust type
covid = covid.astype("int64")
# drop rows with all zeros
covid = covid.loc[(covid!=0).any(axis=1)]
# reset index
covid.reset_index(drop = True, inplace = True)

covid.head()


# In[ ]:


covid.describe()


# We have 45 total observations. A histogram and a boxplot for each variable will be plotted to see the distribution of the data

# In[ ]:


covid.hist()


# In[ ]:


covid.boxplot()


# We can see there is an outlier in each each variable. Let's pull out the outlier

# In[ ]:


covid[(covid["30_sep_cases"] > 200) & (covid["last_14_days_cases"] > 100) & (covid["last_30_days_cases"] > 100) & (covid["last_60_days_cases"] > 50)]


# We will plot scatter plots of the `30_sep_cases` against each independent variable

# In[ ]:


covid.plot(x = "last_14_days_cases", y = "30_sep_cases", kind = "scatter")


# In[ ]:


covid.plot(x = "last_30_days_cases", y = "30_sep_cases", kind = "scatter")


# In[ ]:


covid.plot(x = "last_60_days_cases", y = "30_sep_cases", kind = "scatter")


# We can see that for each plot, the outlier identified above creates a large gap in our plots, which nullifies our assumption of linearity since any possible shape can be present in the gap if more data is availabe. Therefore, it is clear that standardisation/normalistion is necessary.

# ## 4.2 Initial Model
# 
# To determine the best standardisation/transformation to apply, an initial linear model will be built using the variables as they are with no alterations, future models will then be compared against this model to determine improvements. The linear model will be built using the `sklearn` library. First we will examine the correlation between the variables

# In[ ]:


covid.corr()


# As expected there is very high correlation between the variables, since the figures for the longer time periods have the smaller time periods built into them. We can further explore the multicollinearity between the explanatory variables by calculating the variance inflation factor (VIF) of each variable [(source)](https://www.datasciencemadesimple.com/difference-two-timestamps-seconds-minutes-hours-pandas-python-2/). As a rule of thumb, a VIF greater than 5 is considered high and indicates multicollinearity. The VIF will be calculated using the `variance_inflation_factor` function from the `statsmodels` package

# In[ ]:


# build function to calculate VIF
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)


# In[ ]:


# calculate VIF of covid dataframe explanatory variables
calc_vif(covid.iloc[:,1:])


# As expected, the variables have very high VIF values since the variables are very related to each other.
# 
# Finally, we will run the linear regression model on our unedited variables

# In[ ]:


# train, test split
X_train, X_test, y_train, y_test = train_test_split(covid.iloc[:,1:],covid.iloc[:,:1], random_state = 111)


# In[ ]:


# initialize model
lm = LinearRegression()
# fit the linear model using the training data
lm.fit(X_train, y_train)
# r-squared
print ('r-squared for this model = ', lm.score(X_test,y_test))


# Various normalisation and transformation techniques will be now be examined and compared to the original plots, figures, and model.
# 
# ### 4.3 Normalisation
# 
# We will now explore whether the data should be normalised to give more accurate results
# 
# #### 4.3.1 Standardization
# 
# The first normalisation method to be trialled is standardisation where the values for each explanatory variable will be rescaled to have mean 0 and standard deviation 1, this will be done using the `StandardScaler` function from the `sklearn` package

# In[ ]:


# fit the standardisation
std_scale = preprocessing.StandardScaler().fit(covid[['last_14_days_cases', 'last_30_days_cases', 'last_60_days_cases']])
# transfrom the explanatory variables into the standardised values
covid_std = std_scale.transform(covid[['last_14_days_cases', 'last_30_days_cases', 'last_60_days_cases']])

covid_std[0:5]


# The standardised values will be added to the `covid` dataframe

# In[ ]:


covid['last_14_days_cases_std'] = covid_std[:,0]
covid['last_30_days_cases_std'] = covid_std[:,1]
covid['last_60_days_cases_std'] = covid_std[:,2]
covid.head()


# In[ ]:


covid.describe()


# We observe that mean and standard deviation for the standardised variables are approximately 0 and 1, respectively.
# 
# We will plot the scatter plots , calculate the correlation coefficients and the VIF using the standardised variables

# In[ ]:


covid.plot(x = "last_14_days_cases_std", y = "30_sep_cases", kind = "scatter")


# In[ ]:


covid.plot(x = "last_30_days_cases_std", y = "30_sep_cases", kind = "scatter")


# In[ ]:


covid.plot(x = "last_60_days_cases_std", y = "30_sep_cases", kind = "scatter")


# In[ ]:


covid.corr()


# In[ ]:


# calculate VIF of covid dataframe explanatory variables
calc_vif(covid.iloc[:,4:])


# We can see that the standardisation did not alter any of the plots, correlation coefficients, or VIF values.
# 
# Finally, we will run a linear regression using the standardised variables as explanatory variables

# In[ ]:


# train, test split
X_train, X_test, y_train, y_test = train_test_split(covid.iloc[:,4:],covid.iloc[:,:1], random_state = 111)


# In[ ]:


# initialize model
lm = LinearRegression()
# fit the linear model using the training data
lm.fit(X_train, y_train)
# r-squared
print ('r-squared for this model = ', lm.score(X_test,y_test))


# We can observe that standardisation does not have any effect and the r-squared is the same as the original model.
# 
# Therefore we can conclude that the variables should not be standardised. The standardised variables will now be dropped

# In[ ]:


covid.drop(columns = ['last_14_days_cases_std', 'last_30_days_cases_std', 'last_60_days_cases_std'], inplace = True)


# #### 4.3.2 Minmax Normalisation
# 
# The second normalisation technique to be trialled is the Minmax technique where the values of the variables will be normalised to be in the range 0 to 1, this will be done using the `MinMaxScaler` function from the `sklearn` package

# In[ ]:


# fit the minmax normalisation
minmax_scale = preprocessing.MinMaxScaler().fit(covid[['last_14_days_cases', 'last_30_days_cases', 'last_60_days_cases']])
# transfrom the explanatory variables into the normalised values
covid_minmax = minmax_scale.transform(covid[['last_14_days_cases', 'last_30_days_cases', 'last_60_days_cases']])
covid_minmax[0:5]


# The normalised values will be added to the `covid` dataframe

# In[ ]:


covid['last_14_days_cases_nrm'] = covid_minmax[:,0]
covid['last_30_days_cases_nrm'] = covid_minmax[:,1]
covid['last_60_days_cases_nrm'] = covid_minmax[:,2]
covid.head()


# In[ ]:


covid.describe()


# We observe that minimum and maximum for the normalised variables are 0 and 1, respectively.
# 
# We will plot the scatter plots , calculate the correlation coefficients and the VIF using the standardised variables

# In[ ]:


covid.plot(x = "last_14_days_cases_nrm", y = "30_sep_cases", kind = "scatter")


# In[ ]:


covid.plot(x = "last_30_days_cases_nrm", y = "30_sep_cases", kind = "scatter")


# In[ ]:


covid.plot(x = "last_60_days_cases_nrm", y = "30_sep_cases", kind = "scatter")


# In[ ]:


covid.corr()


# In[ ]:


# calculate VIF of covid dataframe explanatory variables
calc_vif(covid.iloc[:,4:])


# We can see that the normalisation did not alter any of the plots, correlation coefficients, or VIF values.
# 
# Finally, we will run a linear regression using the normalised variables as explanatory variables

# In[ ]:


# train, test split
X_train, X_test, y_train, y_test = train_test_split(covid.iloc[:,4:],covid.iloc[:,:1], random_state = 111)


# In[ ]:


# initialize model
lm = LinearRegression()
# fit the linear model using the training data
lm.fit(X_train, y_train)
# r-squared
print ('r-squared for this model = ', lm.score(X_test,y_test))


# We can observe that normalisation does not have any effect and the r-squared is the same as the original model.
# 
# Therefore we can conclude that the variables should not be normalised at all. This is due to the fact that the explanatory variables are all on the same scale already, namely they are all measurements of days.

# In[ ]:


# drop the normalised variables
covid.drop(columns = ['last_14_days_cases_nrm', 'last_30_days_cases_nrm', 'last_60_days_cases_nrm'], inplace = True)


# ### 4.4 Transformation
# 
# The effect of transformation on the explanatory variables will now be examined
# 
# #### 4.4.1 Root Transformation
# 
# For this transformation, each of the variables will be recalculated to its square root. We will first alter any negative values to 0

# In[ ]:


covid_root = covid.copy()
covid_root.loc[covid["30_sep_cases"] < 1, "30_sep_cases"] = 0


# In[ ]:


# generate square root columns

covid_root['30_sep_cases_sqrt'] = None
i = 0
for row in covid_root.iterrows():
    covid_root['30_sep_cases_sqrt'].at[i] = math.sqrt(covid_root["30_sep_cases"][i])
    i += 1

covid_root['last_14_days_cases_sqrt'] = None
i = 0
for row in covid_root.iterrows():
    covid_root['last_14_days_cases_sqrt'].at[i] = math.sqrt(covid_root["last_14_days_cases"][i])
    i += 1
    
covid_root['last_30_days_cases_sqrt'] = None
i = 0
for row in covid_root.iterrows():
    covid_root['last_30_days_cases_sqrt'].at[i] = math.sqrt(covid_root["last_30_days_cases"][i])
    i += 1

covid_root['last_60_days_cases_sqrt'] = None
i = 0
for row in covid_root.iterrows():
    covid_root['last_60_days_cases_sqrt'].at[i] = math.sqrt(covid_root["last_60_days_cases"][i])
    i += 1

covid_root = covid_root.astype({"30_sep_cases_sqrt": "float", "last_14_days_cases_sqrt": "float", "last_30_days_cases_sqrt": "float", "last_60_days_cases_sqrt": "float"})
covid_root.head()


# In[ ]:


covid_root.describe()


# We will plot the scatter plots , calculate the correlation coefficients and the VIF using the standardised variables

# In[ ]:


covid_root.plot(x = "last_14_days_cases_sqrt", y = "30_sep_cases_sqrt", kind = "scatter")


# In[ ]:


covid_root.plot(x = "last_30_days_cases_sqrt", y = "30_sep_cases_sqrt", kind = "scatter")


# In[ ]:


covid_root.plot(x = "last_60_days_cases_sqrt", y = "30_sep_cases_sqrt", kind = "scatter")


# In[ ]:


covid_root.corr()


# In[ ]:


# calculate VIF of covid dataframe explanatory variables
calc_vif(covid_root.iloc[:,5:])


# We can see that the root transformation improved the plots by making the points more spread out, however the gap still exists due to the outlier, high correlation still exists between the variables, and the VIF while extremely high, is lower than the original model
# 
# Finally, we will run a linear regression using the transformed variables

# In[ ]:


# train, test split
X_train, X_test, y_train, y_test = train_test_split(covid_root.iloc[:,5:],covid_root.iloc[:,4:5], random_state = 111)


# In[ ]:


# initialize model
lm = LinearRegression()
# fit the linear model using the training data
lm.fit(X_train, y_train)
# r-squared
print ('r-squared for this model = ', lm.score(X_test,y_test))


# The r-squared for this model is better than the original model

# #### 4.4.2 Square Power Transformation
# 
# The next type of transformation to be trialled is the square power transformation on each variable

# In[ ]:


# generate square power columns

covid['30_sep_cases_pow'] = None
i = 0
for row in covid.iterrows():
    covid['30_sep_cases_pow'].at[i] = math.pow(covid["30_sep_cases"][i], 2)
    i += 1

covid['last_14_days_cases_pow'] = None
i = 0
for row in covid.iterrows():
    covid['last_14_days_cases_pow'].at[i] = math.pow(covid["last_14_days_cases"][i], 2)
    i += 1
    
covid['last_30_days_cases_pow'] = None
i = 0
for row in covid.iterrows():
    covid['last_30_days_cases_pow'].at[i] = math.pow(covid["last_30_days_cases"][i], 2)
    i += 1

covid['last_60_days_cases_pow'] = None
i = 0
for row in covid.iterrows():
    covid['last_60_days_cases_pow'].at[i] = math.pow(covid["last_60_days_cases"][i], 2)
    i += 1

covid = covid.astype("int64")
covid.head()


# In[ ]:


covid.describe()


# We will plot the scatter plots , calculate the correlation coefficients and the VIF using the standardised variables

# In[ ]:


covid.plot(x = "last_14_days_cases_pow", y = "30_sep_cases_pow", kind = "scatter")


# In[ ]:


covid.plot(x = "last_30_days_cases_pow", y = "30_sep_cases_pow", kind = "scatter")


# In[ ]:


covid.plot(x = "last_60_days_cases_pow", y = "30_sep_cases_pow", kind = "scatter")


# In[ ]:


covid.corr()


# In[ ]:


# calculate VIF of covid dataframe explanatory variables
calc_vif(covid.iloc[:,5:])


# We can see that the square power transformation deteriorated the plots by making the points more compact, high correlation still exists between the variables, and the VIF is now much higher than the original model. All of which indicate the square power transformation is not suitable.
# 
# Finally, we will run a linear regression using the transformed variables

# In[ ]:


# train, test split
X_train, X_test, y_train, y_test = train_test_split(covid.iloc[:,5:],covid.iloc[:,4:5], random_state = 111)


# In[ ]:


# initialize model
lm = LinearRegression()
# fit the linear model using the training data
lm.fit(X_train, y_train)
# r-squared
print ('r-squared for this model = ', lm.score(X_test,y_test))


# The r-squared for this model is much worse than the original model

# In[ ]:


covid.drop(columns = ['30_sep_cases_pow', 'last_14_days_cases_pow', 'last_30_days_cases_pow', 'last_60_days_cases_pow'], inplace = True)


# #### 4.4.3 Log Transformation
# 
# In this type of transformation, the variables will be calculated to their log. First we will convert any non-positive values to 1

# In[ ]:


covid_log = covid.copy()
covid_log.loc[covid["30_sep_cases"] < 1, "30_sep_cases"] = 1
covid_log.loc[covid["last_14_days_cases"] < 1, "last_14_days_cases"] = 1
covid_log.loc[covid["last_30_days_cases"] < 1, "last_30_days_cases"] = 1
covid_log.loc[covid["last_60_days_cases"] < 1, "last_60_days_cases"] = 1


# In[ ]:


# generate log columns

covid_log['30_sep_cases_log'] = None
i = 0
for row in covid_log.iterrows():
    covid_log['30_sep_cases_log'].at[i] = math.log(covid_log["30_sep_cases"][i])
    i += 1

covid_log['last_14_days_cases_log'] = None
i = 0
for row in covid_log.iterrows():
    covid_log['last_14_days_cases_log'].at[i] = math.log(covid_log["last_14_days_cases"][i])
    i += 1
    
covid_log['last_30_days_cases_log'] = None
i = 0
for row in covid_log.iterrows():
    covid_log['last_30_days_cases_log'].at[i] = math.log(covid_log["last_30_days_cases"][i])
    i += 1

covid_log['last_60_days_cases_log'] = None
i = 0
for row in covid_log.iterrows():
    covid_log['last_60_days_cases_log'].at[i] = math.log(covid_log["last_60_days_cases"][i])
    i += 1

covid_log = covid_log.astype({"30_sep_cases_log": "float", "last_14_days_cases_log": "float", "last_30_days_cases_log": "float", "last_60_days_cases_log": "float"})
covid_log.head()


# In[ ]:


covid_log.describe()


# We will plot the scatter plots , calculate the correlation coefficients and the VIF using the standardised variables

# In[ ]:


covid_log.plot(x = "last_14_days_cases_log", y = "30_sep_cases_log", kind = "scatter")


# In[ ]:


covid_log.plot(x = "last_30_days_cases_log", y = "30_sep_cases_log", kind = "scatter")


# In[ ]:


covid_log.plot(x = "last_60_days_cases_log", y = "30_sep_cases_log", kind = "scatter")


# In[ ]:


covid_log.corr()


# In[ ]:


# calculate VIF of covid dataframe explanatory variables
calc_vif(covid_log.iloc[:,5:])


# We can see that the log transformation vastly improved the plots by making the points much more distributed and greatly reducing the gap, high correlation still exists between the variables, and the VIF while still high, is lower than the original model.
# 
# Finally, we will run a linear regression using the transformed variables

# In[ ]:


# train, test split
X_train, X_test, y_train, y_test = train_test_split(covid_log.iloc[:,5:],covid_log.iloc[:,4:5], random_state = 111)


# In[ ]:


# initialize model
lm = LinearRegression()
# fit the linear model using the training data
lm.fit(X_train, y_train)
# r-squared
print ('r-squared for this model = ', lm.score(X_test,y_test))


# The r-squared for this model is the same as the original model. 

# #### 4.4 Box Cox Transformation
# 
# The final transformation method to be trialled is the box cox transformation. First we will convert any values less than or equal to zero to 1, this will be done using the `boxcox` function from the `scipy` package

# In[ ]:


box_cox = covid.copy()
box_cox.loc[box_cox["30_sep_cases"] <= 0, "30_sep_cases"] = 1
box_cox.loc[box_cox["last_14_days_cases"] <= 0, "last_14_days_cases"] = 1
box_cox.loc[box_cox["last_30_days_cases"] <= 0, "last_30_days_cases"] = 1
box_cox.loc[box_cox["last_60_days_cases"] <= 0, "last_60_days_cases"] = 1


# In[ ]:


# generate box cox columns
box_cox["30_sep_cases_box"],_ = stats.boxcox(box_cox["30_sep_cases"])
box_cox["last_14_days_cases_box"],_ = stats.boxcox(box_cox["last_14_days_cases"])
box_cox["last_30_days_cases_box"],_ = stats.boxcox(box_cox["last_30_days_cases"])
box_cox["last_60_days_cases_box"],_ = stats.boxcox(box_cox["last_60_days_cases"])
box_cox.head()


# In[ ]:


box_cox.describe()


# We will plot the scatter plots , calculate the correlation coefficients and the VIF using the standardised variables

# In[ ]:


box_cox.plot(x = "last_14_days_cases_box", y = "30_sep_cases_box", kind = "scatter")


# In[ ]:


box_cox.plot(x = "last_30_days_cases_box", y = "30_sep_cases_box", kind = "scatter")


# In[ ]:


box_cox.plot(x = "last_60_days_cases_box", y = "30_sep_cases_box", kind = "scatter")


# In[ ]:


box_cox.corr()


# In[ ]:


# calculate VIF of covid dataframe explanatory variables
calc_vif(box_cox.iloc[:,5:])


# We can see that the box cox transformation improved the original plots by making the points more distributed and eliminating the gap due to the outlier, however, we have many points accumulated at the zero point of the x-axis and a new gap is formed between those points and the remaining points, high correlation still exists between the variables, and the VIF while still high, is lower than the original model.
# 
# Finally, we will run a linear regression using the transformed variables

# In[ ]:


# train, test split
X_train, X_test, y_train, y_test = train_test_split(box_cox.iloc[:,5:],box_cox.iloc[:,4:5], random_state = 111)


# In[ ]:


# initialize model
lm = LinearRegression()
# fit the linear model using the training data
lm.fit(X_train, y_train)
# r-squared
print ('r-squared for this model = ', lm.score(X_test,y_test))


# We can observe that the r-squared for this model is the same as the original

# ### 4.5 Final Model
# 
# Based on the above experiments regarding normalisation and transformation, the final model will:
# - **Not be normalised**. The explanatory variables are already all on the same scale. Furthermore, normalising the variables does not enhance the linear regression model
# - **Log transformed**. The log transformation provides the most linear relationship between each of the independent variables and the dependent variable according to the scatter plots. The r-squared is high for the log transformed model at 79%

# ## 5. Conclusion
# 
# This assignment displayed various methods and techniques towards the integration of datasets from different sources. These methods included database merging, the use of shapefiles, connecting several datasets to each other to derive the requierd information, as well as webscraping. Integration issues such as the presence of duplicates were also examined and solved. Furthermore, normalisation and transformation techniques and their effect on linear model creation based on the COVID figures were also analysed, and an attempt was made to recommend the most appropriate transformation.
# 
# ## 6. References
# 
# - Bhandari, A. (2020, March 20). What is Multicollinearity? Here’s Everything You Need to Know. Retrieved from https://www.analyticsvidhya.com/blog/2020/03/what-is-multicollinearity/
# - Datascience Made Simple. (2021). Difference Between Two Timestamps in Seconds, Minutes, Hours in Pandas Python. Retrieved from
# https://www.datasciencemadesimple.com/difference-two-timestamps-seconds-minutes-hours-pandas-python-2/
# - Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. (Publisher link).
# - Pypi. (2021). Haversine 2.5.1. Retrieved from https://pypi.org/project/haversine/
# - J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.
# - McKinney, W., & others. (2010). Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51–56)
# - Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
# - Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and statistical modeling with python. In 9th Python in Science Conference.
# - Van Rossum, G. (2020). The Python Library Reference, release 3.8.2. Python Software Foundation.
# - Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2
