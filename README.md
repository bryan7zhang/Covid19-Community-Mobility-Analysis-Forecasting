# Covid-19 Canada Community Mobility Analysis and Forecasting
Covid-19 greatly affects how people commute domestically and internationally. 
The goal of this project is to apply machine learning techniques on the community mobility trend data in relation
with Covid-19 statistics. 


## Chosen Datasets
Five datasets are chosen for this project in analysing the correlation between community mobility
and Covid-19 cases in Canada: 
 1. [Canada Covid-19 case dataset](https://health-infobase.canada.ca/covid-19/visual-data-gallery/) from StatCanada
 2. [Covid-19 Community Mobility Report](https://www.google.com/covid19/mobility/) from Google
 3. [Covid-19 Mobility Trends Report](https://covid19.apple.com/mobility) from Apple
 4. [Canada Holiday API](https://canada-holidays.ca/api/v1/provinces) from Open Canada
 5. [Stringency index of OxCGRT](https://ourworldindata.org/grapher/covid-stringency-index) from University of Oxford


## Data Processing
The data ingestion process akin to Data ETLs encompassing data extraction, transformation, and loading.
Check out the [dataset-generator](./DataPreprocessing/dataset_generator.py) script carrying out data clean-up, data extraction
and data transformation before feeding to machine learning models.


## Machine Learning Models
Two main supervised learning models, XGBoost and Vector AutoRegressive (VAR) models were used to predict Covid-19 in Canada
and provinces of Canada. To analyse and forecasting variables over time, VAR is more suitable than AutoRegressive Integrated 
Moving Average model (ARIMA model, uni-variate) to crunch the data in this project for cases forecasting and mobility trend 
forecasting. The predictions from different models are compared and evaluated based on the metrics of mean
absolute error (MAE), root mean squared error (RMSE), percentage error.

Unsupervised learning techniques were also applied to classify and predict the potential exposure risks using the mobility
data coupled with the daily active Covid-19 cases. The main technique used was K-Means Clustering.


## Other Contributors
Robert Vu & Philip Bolinao
