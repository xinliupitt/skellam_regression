# Skellam Regression

This repo provides information for the manuscript "Excess demand prediction for bike sharing systems".

## Basic example codes of Skellam regression

To the best of our knowledge, we are the first to implement Skellam regression for multiple input covariates in Python. Particularly:
* As basic example codes, our implementation of Skellam regression is [skellam_regression_code.py](skellam_regression_code.py).
* The format of input covariates and output is numpy array, not pandas dataframe. 
* Please run `python3 skellam_regression.py` to start the Skellam regression training and prediction processes.

## Skellam regression for Chicago Divvy

Our manuscript focuses on building a Skellam model to predict the total demand of Chicago Divvy bike sharing system. We have the observed, excess and total demand data ready in the folder:
* plosone_depts_2018
	* `depts_whole_*` stores the covariates for bike demand
	* `depts_UD_whole_*` stores the bike total demand
* plosone_depts_temp_df
	* `depts_UD_dict_*` stores the bike excess demand
* plosone_arrivs_2018
	* `arrivs_whole_*` stores the covariates for dock demand
	* `arrivs_UD_whole_*` stores the dcok total demand
* plosone_arrivs_temp_df
	* `arrivs_UD_dict_*` stores the dock excess demand

To obtain the weather data as input covariates,
* please subscribe “History Bulk” in Openweathermap: https://home.openweathermap.org/marketplace.
* Download the dataset and rename it to `CHI_Weather.csv`. 
* Put this dataset file in the folder `plosone_weather`.
* run `python3 weather_generator.py` to fetch the data of our year of interest (2018) from the whole weather dataset.
* run `python3 weather_to_stations.py` to convert the fetched weather data to the format of input covariates for the model.

Now we have the all input covariates and outputs ready. We can start the training and prediction process for Divvy by doing this:
* As an example command, run `python3 skellam_total_rush.py --start_s 230 --end_s 232 --control 0 --peak 1 &> skellam_total_rush.txt`.
* The numbers between `--start_s` and `--end_s` are the station IDs the model will be built for. In the example command, the model will be built for Staton 230 and 231.
* If `--control` is 0, it means we use the total demand as the dependent variable in training process; if `--control` is 1, we use the observed demand in training
* If `--peak` is 1, it means we build model for peak hours; if `--peak` is 0, we build model for non-peak hours.
* Summaries will be printed in the log file `skellam_total_rush.txt`. 
