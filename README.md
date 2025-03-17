# Predictive Maintenance of NASA Turbofan Jet Engine

## Overview

This project focuses on developing predictive maintenance models for NASA's Turbofan Jet Engines. By analyzing engine sensor data, the goal is to predict the Remaining Useful Life (RUL) of each engine, enabling proactive maintenance and reducing downtime.

## Dataset Description

The project utilizes the [NASA Turbofan Engine Degradation Simulation Data Set](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6). This dataset comprises multiple multivariate time series from different engine units, each starting with varying degrees of initial wear and manufacturing variation. Each engine operates normally at the start and develops a fault over time.

Key characteristics of the dataset include:
- Multiple operational settings and sensor measurements.
- Data divided into training and test sets.
- The objective is to predict the RUL for each engine in the test set. There are 100 different engine units.

## Folder structure
```
Predictive_Maintenance-NASA_Turbofan_Jet_Engine/
│
├── data/
│   └── processed/
│
├── FD001 Analysis/
│   ├── Main.ipynb
|   ├── Testing.ipynb
|   ├── saved_models/
|   ├── saved_scalers/
|   ├── preprocess.py
│
├── Deep Learning/
│   ├── DL_Main.ipynb
|   ├── neuralnetworks.py
|
├── README.md
├── requirements.txt
```

## Data Preprocessing

Data preprocessing steps include:

- Columns were renamed and empty columns were removed
- New Column (RUL = Max Cycle - Current cycle) was created
- Normalizing sensor measurements, Operating settings and Cycles.

## Exploratory Data Analysis (EDA)

EDA involves:

- Visualizing sensor trends over time
- Identifying correlations between sensors and RUL
- Grouping and observing average sensor values

## Feature Engineering

Features engineered for modeling:

- Clipping was done to contain extreme values
- Undersampling was done randomly to rarer values
- Trend features capturing the slope of sensor signals
- PCA was performed, but yielded poor results

## Modeling Approach

The following predictive models are employed:

- **Random Forest Regressor:** To capture non-linear relationships and interactions between features.
- **XGBoost:** To handle unbalanced data

GridSearchCV was performed to optimize hyperparameters. Validation set was split from the training data file.

## Model Evaluation

Models are evaluated using:

- **Root Mean Squared Error (RMSE):** Measures the average magnitude of prediction errors.
- **Mean Absolute Error (MAE):** Provides the average absolute difference between predicted and actual RUL.

## How to Use This Repository

To replicate the analysis:

1. Clone the repository:
```
    git clone https://github.com/Parithimaal/Predictive_Maintenance-NASA_Turbofan_Jet_Engine.git
```
2. Install required packages
```
    pip install -r requirements.txt
```
3. For training and tweaking run Main.ipynb
4. The Testing.ipynb contains the evaluation of saved models from Main.ipynb

## Results and Discussion
The XGBoost model was the best performer with the an RMSE of 37.71. Interestingly, performing PCA with components responsible for 96% of variance yielded much poorer results. The feature importance of Random Forest Mode indicated that the number of cycles, sensors values 11 were the most important factors based on which splits were made. Though the performance of the models is appreciable, it could be improved by providing the models context of the temporal series and past data for each unit.

## Future Work
- **Deep Learning:** Using Neural Networks such as LSTM and GRU to capture temporal patterns
- **Asymmetric Loss function** Using an Asyemmtric Loss function to penalize over-prediction than under-prediction

## References
[NASA Turbofan Engine Degradation Simulation Data Set](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data)

