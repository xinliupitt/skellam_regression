# Skellam Regression

This repo provides information for the manuscript "Excess demand prediction for bike sharing systems".

## Basic example codes of Skellam regression

To the best of our knowledge, we are the first to implement Skellam regression for multiple input covariates in Python. Particularly:
* As basic example codes, our implementation of Skellam regression is [skellam_regression_code.py](skellam_regression_code.py).
* The format of input covariates and output is numpy array, not pandas dataframe. 
* Please run `python3 skellam_regression.py` to start the Skellam regression training and prediction processes.

## Skellam regression for Chicago Divvy

Our manuscript focuses on building a Skellam model to predict the total demand of Chicago Divvy bike sharing system. We have the observed, excess and total demand data ready in the folder:
* Folder 1
* Folder 2

To obtain the weather data as input covariates, please subscribe “History Bulk” in Openweathermap: https://home.openweathermap.org/marketplace. Then download the dataset and rename it to "CHI_Weather.csv". Next, use the following scripts to further process the data:
* run `python3 weather_generator.py` to fetch the data of our year of interest (2018) from the whole weather dataset.
* run `python3 weather_to_stations.py` to convert the fetched weather data to the format of input covariates for the model.

Now we have the all input covariates and outputs ready. We can start the training and prediction process for Divvy by doing this:
* run
