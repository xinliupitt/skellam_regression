import collections
import datetime
import dateutil.parser
import math
import numpy as np
import pandas as pd
import pickle
from pytz import timezone

weather_path = "plosone_weather/"
weather = pd.read_csv(weather_path + "CHI_Weather.csv")
tz = timezone('US/Central')

def to_iso(origin):
    strings = origin.split(' ')
    return strings[0]+'T'+strings[1]+'.000Z'

mydates = []
for w_idx, w in weather.iterrows():
    mydate = dateutil.parser.parse(to_iso(w['dt_iso']))
    mydate = mydate.astimezone(tz)
    mydates.append(mydate)

# find the start and end index of year 2018 in the raw weather dataframe
for time_idx, time in enumerate(mydates):
    if time.year == 2018 and time.month == 1:
        print(time_idx)
        start = time_idx
        idx_new = time_idx
        break

for time_idx, time in enumerate(mydates[idx_new:]):
    if time.year == 2019 and time.month == 1:
        print(time_idx)
        end = start + time_idx
        break

# process temperature data
date_to_temp = {}
for w_idx, w in weather.iterrows():
    if w_idx < start:
        continue
    if w_idx == end:
        break
    date_to_temp[(mydates[w_idx].month, mydates[w_idx].day, mydates[w_idx].hour)] = w['temp']

for month in range(1, 13):
    for day in range(1, 32):
        for hour in range(0, 23):
            inter_hour = hour+0.5
            try:
                inter_temp = (date_to_temp[(month, day, hour)] + date_to_temp[(month, day, hour+1)])/2
            except:
                continue
            date_to_temp[(month, day, inter_hour)] = inter_temp
f = open(weather_path+"date_to_temp_2018.pkl","wb")
pickle.dump(date_to_temp,f)
f.close()

# process cloud percentage data
date_to_cloud = {}
for w_idx, w in weather.iterrows():
    if w_idx < start:
        continue
    if w_idx == end:
        break
    date_to_cloud[(mydates[w_idx].month, mydates[w_idx].day, mydates[w_idx].hour)] = w['clouds_all']

for month in range(1, 13):
    for day in range(1, 32):
        for hour in range(0, 23):
            inter_hour = hour+0.5
            try:
                inter_temp = (date_to_cloud[(month, day, hour)] + date_to_cloud[(month, day, hour+1)])/2
            except:
                continue
            date_to_cloud[(month, day, inter_hour)] = inter_temp
f = open(weather_path + "date_to_cloud_2018.pkl","wb")
pickle.dump(date_to_cloud,f)
f.close()

# process wind speed data
date_to_ws = {}
for w_idx, w in weather.iterrows():
    if w_idx < start:
        continue
    if w_idx == end:
        break
    date_to_ws[(mydates[w_idx].month, mydates[w_idx].day, mydates[w_idx].hour)] = w['wind_speed']

for month in range(1, 13):
    for day in range(1, 32):
        for hour in range(0, 23):
            inter_hour = hour+0.5
            try:
                inter_temp = (date_to_ws[(month, day, hour)] + date_to_ws[(month, day, hour+1)])/2
            except:
                continue
            date_to_ws[(month, day, inter_hour)] = inter_temp
f = open(weather_path + "date_to_ws_2018.pkl","wb")
pickle.dump(date_to_ws,f)
f.close()

mains = list(weather['weather_main'].unique())
m_to_idx = {}
for idx in range(len(mains)):
    m_to_idx[mains[idx]] = idx

# process categorical descriptions
date_to_wm = {}
for w_idx, w in weather.iterrows():
    if w_idx < start:
        continue
    if w_idx == end:
        break
    if (mydates[w_idx].month, mydates[w_idx].day, mydates[w_idx].hour) not in date_to_wm:
        date_to_wm[(mydates[w_idx].month, mydates[w_idx].day, mydates[w_idx].hour)] = [0]*len(m_to_idx)
    date_to_wm[(mydates[w_idx].month, mydates[w_idx].day, mydates[w_idx].hour)][m_to_idx[w['weather_main']]] = 1

for month in range(1, 13):
    for day in range(1, 32):
        for hour in range(0, 23):
            inter_hour = hour+0.5
            try:
                inter_temp = date_to_wm[(month, day, hour+1)]
            except:
                continue
            date_to_wm[(month, day, inter_hour)] = inter_temp
f = open(weather_path+"date_to_wmm_2018.pkl","wb")
pickle.dump(date_to_wm,f)
f.close()
print("Weather data generated.")
