# flight-price-prediction

### Project Overview

This project focuses on predicting flight ticket prices using machine learning.
The goal is to estimate the cost of a flight based on travel details such as airline, departure time, class, number of stops, and days left until departure.


### Problem Description

Flight prices depend on multiple variables: class, airline, duration, demand, and time before departure.
The objective is to train a model that can predict the flight price given these parameters.
Such a model can help users plan trips or assist airlines in dynamic pricing analysis.

### Dataset

Dataset source: [Flight Price Prediction – Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

**Main features**
- `airline`: Airline company  
- `flight`: Flight code  
- `source_city`: Departure city  
- `destination_city`: Arrival city  
- `departure_time`: Departure time category (Morning, Afternoon, Evening, etc.)  
- `arrival_time`: Arrival time category  
- `stops`: Number of stops (non-stop, 1 stop, etc.)  
- `class`: Ticket class (Economy / Business)  
- `duration`: Flight duration (minutes)  
- `days_left`: Days remaining until departure  
- `price`: Ticket price (target variable)

## Exploratory Data Analysis
- Checked for missing values and data consistency  
- Explored price distribution and variation across airlines and classes  
- Observed that:
  - Business class tickets are significantly more expensive  
  - Prices increase as the number of days left decreases  
  - Longer flights tend to be more expensive

## Feature Engineering
- Converted categorical time features (`departure_time`, `arrival_time`) to numeric mappings  
- Applied `LabelEncoder` to categorical columns  
- Scaled numerical features using `StandardScaler` (for linear models)  
- Split data into Train (60%), Validation (20%), and Test (20%) sets 

## Model Training
Trained and compared several models:
- Linear Regression 
- Ridge Regression
- Gradient Boosting
- XGBoost
- **Random Forest**

The Random Forest model achieved the best performance with strong generalization.

## Feature Importance
Key factors influencing ticket prices:
1. `class` – the most dominant feature  
2. `duration` – longer flights cost more  
3. `days_left` – fewer days left → higher price  
4. `airline`, `source_city`, `destination_city` – moderate impact  
5. `stops`, `departure_time`, `arrival_time` – minor but consistent influence

## Running the Project Locally

### 1. Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the API
```bash
python predict.py
```

### 3. Test a Prediction
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"airline": "Vistara", "flight": "UK-811", "source_city": "Delhi", "departure_time": "Morning", "stops": "non-stop", "arrival_time": "Evening", "destination_city": "Mumbai", "class": "Economy", "duration": 135, "days_left": 20}' \
  http://127.0.0.1:9696/predict
```

## Docker

### build and run
```bash
docker build -t flight-price-service .
docker run -p 9696:9696 flight-price-service
```

## Cloud Deployment

The service is deployed on Render as a Dockerized web service.

Endpoint (POST):
`https://flight-price-prediction-6kw1.onrender.com/predict`

Example request:
```bash

curl -X POST -H "Content-Type: application/json" \
  -d '{"airline": "Vistara", "flight": "UK-811", "source_city": "Delhi", "departure_time": "Morning", "stops": "non-stop", "arrival_time": "Evening", "destination_city": "Mumbai", "class": "Economy", "duration": 135, "days_left": 20}' \
  https://flight-price-prediction-6kw1.onrender.com/predict
  
```


### Midterm Project - ML Zoomcamp 2025
### Flight Price Prediction
**Aleksandra Beliaeva**




