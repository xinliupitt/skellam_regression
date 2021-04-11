import numpy as np
import pandas as pd
import pickle

weather_path = "plosone_weather/"
data_path = "plosone_"

date_to_temp = pd.read_pickle(weather_path+"date_to_temp_2018.pkl")
date_to_cloud = pd.read_pickle(weather_path + "date_to_cloud_2018.pkl")
date_to_ws = pd.read_pickle(weather_path + "date_to_ws_2018.pkl")
date_to_wm = pd.read_pickle(weather_path+"date_to_wmm_2018.pkl")

def normalize_num(D_vals):
    D_vals_new = []
    max_val, min_val = max(D_vals), min(D_vals)
    for val in D_vals:
        D_vals_new.append((val-min_val)/(max_val-min_val))
    return D_vals_new

for id_focused in range(0, 300):
    try:
        df_train = pd.read_pickle(data_path+'depts_2018/depts_whole_'+str(id_focused)+'.pkl')
        print('Starting station ID:', id_focused)
    except:
        continue
    drop_columns = ["index", 'wm_1', 'wm_2', 'Clouds', 'Clear', 'Rain',
           'Mist', 'Drizzle', 'Snow', 'Haze', 'Fog', 'Thunderstorm', 'Smoke',
           'Squall', 'Dust', 'temp', 'cloud', 'pressure', 'humidity',
           'wind_speed']
    for col in drop_columns:
        try:
            df_train.drop(columns=[col], inplace=True)
        except:
            continue
    weather_records = []
    for _, record in df_train.iterrows():
        month = record["month"]
        day_m = record["day_m"]
        hour_idx = record["hour_idx"]
        slot = (month, day_m, hour_idx)
        weather_record = []
        weather_record.append(date_to_temp[slot])
        weather_record.append(date_to_cloud[slot])
        weather_record.append(date_to_ws[slot])
        weather_record += date_to_wm[slot]
        weather_records.append(weather_record)
    weather_records = np.transpose(np.asarray(weather_records))
    weather_names = ['temp', 'cloud', 'wind_speed', 'Clouds', 'Clear', 'Rain', 'Mist', 'Drizzle', 'Snow', 'Haze', 'Fog', 'Thunderstorm', 'Smoke', 'Squall', 'Dust']
    for w_idx in range(len(weather_records)):
        records = weather_records[w_idx]
        if w_idx in [0, 1, 2]:
            records = normalize_num(records)
        col_name = weather_names[w_idx]
        se = pd.Series(records)
        df_train[col_name] = se.values
    df_train.to_pickle(data_path+'depts_2018/depts_whole_'+str(id_focused)+'.pkl')
print("Weather data added to dataframe.")
