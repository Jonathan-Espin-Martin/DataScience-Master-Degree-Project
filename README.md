# Supervised Learning for Failure Detection in Engine Generators  
**Leveraging Data Infrastructure, Feature Engineering, and Monitoring via Machine Learning Dashboards**

---

## Overview  
This repository provides the implementation and results of a supervised machine learning approach for failure detection in industrial engine generators. The study is centered around enhancing operational reliability by predicting potential failure events through data-driven insights. It leverages advanced machine learning models such as Random Forest (RF), XGBoost, and Artificial Neural Networks (ANN), integrated seamlessly with the SCADA Ignition platform for real-time monitoring and actionable insights.

---

## Key Features  
- **Data Handling**: Integration with SCADA systems for data acquisition, cleaning, and transformation of a 26-month dataset comprising 8 million samples and 89 features.
- **Models Implemented**:
  - **Random Forest**: Known for robustness and feature interpretability.
  - **XGBoost**: A gradient boosting model excelling in precision and recall.
  - **ANN (MLP)**: Neural networks capturing non-linear relationships for sophisticated failure predictions.
- **Metrics**: Emphasis on F2 Score to prioritize recall, ensuring no failure events are missed.
- **Visualization**: A machine learning dashboard for real-time predictions and feature importance monitoring.

---

## Project Workflow  

### 1. **Data Acquisition and Preprocessing**
- Data is gathered via a private SCADA Ignition system connected to the generator. 
- Historical and real-time data are stored in structured SQL databases. 
- Steps include:
  - Data merging: Combining raw process variables with manually annotated failure events.
  - Cleaning and transformation: Addressing missing values, normalizing features, and reducing dimensionality.

### 2. **Model Training and Evaluation**
- Models are trained using hyperparameter tuning with cross-validation. 
- Evaluation metrics include confusion matrices, F2 scores, precision, and recall.
- Feature importance analysis is conducted to identify the most critical variables influencing failures.

### 3. **Integration with SCADA**
- Machine learning models are deployed within the SCADA Ignition system using Python scripts. 
- Real-time predictions are provided via the dashboard, offering probabilistic insights into failure risks.

---

## Results  
### Random Forest
- **F2 Score**: 0.9450  
- Key Features:
  - Actuator position % (0.355)
  - Generator phase current L2 (0.139)
  - Apparent power (0.119)

![Random Forest Feature Importance](images/random_forest_feature_importance.png)

---

### XGBoost
- **F2 Score**: 0.9908  
- Key Features:
  - Actuator position % (0.544)
  - Exhaust gas temperature B1 (0.050)
  - Apparent power (0.112)

![XGBoost Feature Importance](images/xgboost_feature_importance.png)

---

### ANN (MLP)
- **F2 Score**: 0.95  
- Key Features:
  - Integrated Gradients analysis highlights key variables impacting failures.
- Training performance:
  ![ANN Training Performance](images/ann_training_performance.png)

---

## Machine Learning Dashboard  
An interactive dashboard is hosted on the SCADA Ignition platform:
- **Live Predictions**: Real-time failure probabilities.  
- **Feature Importance**: Visual insights into the most critical operational variables.  
- **Historical Analysis**: Explore feature trends over time.  

![Machine Learning Dashboard](images/ml_dashboard.png)

---

## Installation  
1. Clone this repository:  
   ```bash
   git clone https://github.com/Jonathan-Espin-Martin/DataScience-Master-Degree-Project.git
   cd DataScience-Master-Degree-Project
