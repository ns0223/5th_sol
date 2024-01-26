# <b>IEEE CIS Competition source code [FRESNO Team](https://alipourmousavi.com/research_FRESNO.html)</b>

## Ranked 5th Place in the competition

The source code includes a data folder with data inputs, a Forecasting solution replication notebook, an Optimisation solution replication notebook, and a experiments notebook concludes our solution developments. Please run the Forecasting notenook before optimisation solution becasue the optimisation is built on forecasting results.

## Environment Installation

Required environment is included in reqiorement.txt, which can be installed by:

pip install -r requirements.txt

or:

conda install --file requirements.txt

----

## Data Preprocessing

### About Data

- Forecasting input data is in the data directory 'data/data_processed.pickle', which is pre-processed and cleanned from [IEEE-CIS Technical Challenge on Predict+Optimize for Renewable Energy Scheduling](https://ieee-dataport.org/competitions/ieee-cis-technical-challenge-predictoptimize-renewable-energy-scheduling) [1]
- Optimisation input data has two parts including Forecasting output and the price data in 'data/AEMO_price'
- Two historical submissions in phase1 are exported in 'data/submissions_phase1.pickle'. The buildings' consumption prediction model requires this phase1 prediction data for adjusting AFL day's real data.The Experiments section uses this file to tracking the prediction models' performance changes.

### Forecasting data preprocessed based on rules below

- For any day missing more than half of the valid values, those missing values are replaced by yearly average data accroding to the day type. (weekday average or weekend average)
- For any day missing small amount of data, the missing point is filled with that day's average value
- For buildings have no knowledge on certain time of the day (every data at that time is invalide), the value is set to 0.
- Occupany rate is based on given consumption by days, converted to UTC time accrodingly.
- Special cases are addressed in the [report](#report).

### Forecasting data structure

- The data_processed_phase2.pickle contains a dictionary of 14 cleanned objects in DataFrame as below:
  - Data file = {
    - 'Building0': Building0 consumption data, day type (Monday-Sunday), missing value (if the value was Nan), weather data, occupancy,
    - 'Building1': Building1 consumption data, day type (Monday-Sunday), missing value (if the value was Nan), weather data, occupancy,
    - 'Building3': Building3 consumption data, day type (Monday-Sunday), missing value (if the value was Nan), weather data, occupancy,
    - 'Building4': Building4 consumption data, day type (Monday-Sunday), missing value (if the value was Nan), weather data, occupancy,
    - 'Building5': Building5 consumption data, day type (Monday-Sunday), missing value (if the value was Nan), weather data, occupancy,
    - 'Building6': Building6 consumption data, day type (Monday-Sunday), missing value (if the value was Nan), weather data, occupancy,
    - 'Solar0': Solar0 generation data, day type (Monday-Sunday), missing value (if the value was Nan), weather data, occupancy,
    - 'Solar1': Solar1 generation data, day type (Monday-Sunday), missing value (if the value was Nan), weather data, occupancy,
    - 'Solar2': Solar2 generation data, day type (Monday-Sunday), missing value (if the value was Nan), weather data, occupancy,
    - 'Solar3': Solar3 generation data, day type (Monday-Sunday), missing value (if the value was Nan), weather data, occupancy,
    - 'Solar4': Solar4 generation data, day type (Monday-Sunday), missing value (if the value was Nan), weather data, occupancy,
    - 'Solar5': Solar5 generation data, day type (Monday-Sunday), missing value (if the value was Nan), weather data, occupancy,
    - 'occupancy': occupancy rate from '2020-11-01 00:00:00' to '2020-12-01 00:00:00',
    - 'weather': weather data from '2020-11-01 00:00:00' to '2020-12-01 00:00:00'
        }

----

## Loads Forecasting

The [Forecasting.ipynb](https://gitlab.com/ryuan/ieee-cis-data-challenge-fresno/-/blob/main/Forecasting.ipynb) contains the proposed forecasting solution for building demands and solar generations. Input features are intentionally left as a main function for checking and reading purpose. Other funtions and classes can be found in methods.py.

----

## Schedule Optimisation

The [Optimisation.ipynb](https://gitlab.com/ryuan/ieee-cis-data-challenge-fresno/-/blob/main/Optimisation.ipynb) contains the optimal scheduling algorithm for batteries and recurring activities. The inputs are the AEMO price file, forecasts from the building and solar panels. This forecast path can be specified in the "forecast_path" variable. The output are the schedules for all 10 instances stored in the same folder with forecast results, which can be found in the current working directory after running the [Forecasting.ipynb](https://gitlab.com/ryuan/ieee-cis-data-challenge-fresno/-/blob/main/Forecasting.ipynb).

----

## Experiments (Playground)

The [Experiments.ipynb](https://gitlab.com/ryuan/ieee-cis-data-challenge-fresno/-/blob/main/Experiments.ipynb) contains some insights and trails on how we ended up with the proposed methods. We addressed some possible improvements and validations which we did not manage to finish by the end of competition. Reads are free to try and play around with the mid-products and funtions with this notebook.

----

## Report

we also dsicribe our methodology in the report: [Optimal activity and battery scheduling algorithmusing load and solar generation forecasts](https://gitlab.com/ryuan/ieee-cis-data-challenge-fresno/-/blob/main/Report.pdf)

----

## Reference

[1] Christoph Bergmeir, May 27, 2021, "IEEE-CIS Technical Challenge on Predict+Optimize for Renewable Energy Scheduling", IEEE Dataport, doi: [https://dx.doi.org/10.21227/1x9c-0161.s](https://dx.doi.org/10.21227/1x9c-0161.s)

----

## Cite

R. Yuan, N. T. Dinh, Y. Pipada and S. Ali Pourmouasvi, "IEEE CIS Competition source code FRESNO Team," 2021. [Online]. Available: [https://gitlab.com/ryuan/ieee-cis-data-challenge-fresno](https://gitlab.com/ryuan/ieee-cis-data-challenge-fresno)

----

## Contact

Rui Yuan, ([E-mail](mailto:r.yuan@adelaide.edu.au), [LinkedIn](https://www.linkedin.com/in/rui-yuan-5953aa168/))

Nam Dinh, ([E-mail](mailto:trongnam.dinh@adelaide.edu.au), [LinkedIn](https://www.linkedin.com/in/nam-dinh-7b97b1207/))

Yogesh Pipada, ([E-mail](mailto:yogeshpipada.sunilkumar@adelaide.edu.au). [LinkedIn](https://www.linkedin.com/in/yogesh-pipada-sunil-kumar-29a864126/))
