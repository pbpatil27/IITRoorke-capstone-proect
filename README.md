# Delhivery Last-Mile Delay Intelligence Pipeline

## 1. Project Overview

The **Delhivery Last-Mile Delay Intelligence Pipeline** is a machine learning-based analytics project designed to predict shipment delivery delays and help logistics teams take proactive operational decisions.

Last-mile delivery is one of the most important stages of the logistics process. Delays can affect customer satisfaction, delivery costs, hub performance, and overall operational efficiency.

This project develops an end-to-end intelligence pipeline that:

- Cleans and preprocesses shipment data.
- Identifies patterns associated with delivery delays.
- Predicts expected delivery time using regression models.
- Classifies shipments based on their probability of delay.
- Segments delivery hubs based on operational performance.
- Recommends appropriate actions for potentially delayed shipments.

**Project duration:** 14 days  
**Total marks:** 100  
**Domain:** Logistics and Supply Chain Analytics  
**Tools:** Python, Google Colab, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib

---

## 2. Business Problem

Logistics companies process thousands of shipments across multiple delivery hubs. Several factors can influence delivery performance, including:

- Delivery distance.
- Number of stops on a delivery route.
- Shipment weight.
- Service type.
- Weather conditions.
- Promotional sale events.
- Hub capacity and operational efficiency.
- Dispatch timing.

The primary business problem is:

**How can a logistics company identify shipments that are likely to be delayed before the delivery deadline is missed, and use that information to improve last-mile operations?**

This project addresses the problem through predictive analytics and operational recommendations.

---

## 3. Project Objectives

1. Perform data cleaning and exploratory data analysis.
2. Identify important factors influencing delivery duration.
3. Develop a regression model to estimate delivery time.
4. Develop a classification model to identify potentially delayed shipments.
5. Evaluate models using appropriate performance metrics.
6. Segment hubs into operational performance categories.
7. Generate actionable recommendations for new shipments.
8. Design a production monitoring and retraining strategy.

---

## 4. Dataset Description

The project uses synthetically generated shipment data designed to represent last-mile logistics operations.

### Dataset summary

| Dataset | Records | Description |
|---|---:|---|
| Raw shipments | 1,224 | Historical shipment records, including duplicates and data-quality issues |
| Clean shipments | 1,200 | Records after duplicate removal and preprocessing |
| Hub information | 24 | Hub locations, regions, tiers, and capacities |
| New shipments | 48 | New records used for prediction |

### Important features

| Feature | Description |
|---|---|
| shipment_id | Unique shipment identifier |
| hub_id | Delivery hub identifier |
| service_type | Economy, Standard, or Express |
| payment_mode | COD or Prepaid |
| parcel_weight_kg | Shipment weight |
| distance_km | Delivery distance |
| n_stops | Number of delivery stops |
| rain_flag | Indicates rainy conditions |
| sale_event_flag | Indicates promotional sale periods |
| dispatch_time | Shipment dispatch timestamp |
| hub_tier | Operational hub tier |
| hub_daily_capacity | Daily hub capacity |
| delivery_hours | Actual delivery duration |

### Target variables

**Regression target:** `delivery_hours`

**Classification target:** `is_delayed`

A shipment is classified as delayed when:

`delivery_hours > promised_hours`

Promised delivery hours:

| Service type | Promised hours |
|---|---:|
| Economy | 96 |
| Standard | 60 |
| Express | 36 |

---

## 5. Technology Stack

- **Python:** Main programming language.
- **Pandas:** Data manipulation and preprocessing.
- **NumPy:** Numerical operations and synthetic data generation.
- **Matplotlib and Seaborn:** Exploratory data visualization.
- **Scikit-learn:** Machine learning, feature selection, model evaluation, and clustering.
- **Joblib:** Saving and loading trained models.
- **Google Colab:** Development environment.

---

## 6. Project Workflow

The project is divided into four major stages.

### Part A: Data Cleaning and Exploratory Data Analysis

The raw dataset contains duplicate records, missing values, inconsistent service-type formatting, and invalid weight values.

The following preprocessing operations were performed:

- Removed 24 duplicate records.
- Standardized service-type values.
- Handled missing payment modes.
- Converted declared values into numeric values.
- Identified invalid shipment weights.
- Replaced missing numeric values using training-data preparation medians.
- Integrated hub information.
- Created additional features.

### Feature engineering

Additional features include:

- `promised_hours`
- `is_cod`
- `is_weekend`
- `dispatch_hour`
- `log_declared_value`
- `is_delayed`

The cleaned dataset contains 1,200 records and 27 columns.

### Key exploratory findings

The overall delay rate was approximately **23.17%**.

Delay rates varied considerably across service types:

| Service type | Delay rate |
|---|---:|
| Economy | 8.57% |
| Express | 48.97% |
| Standard | 20.56% |

Delay rates also varied by hub tier:

| Hub tier | Delay rate |
|---|---:|
| Tier 1 | 4.00% |
| Tier 2 | 21.50% |
| Tier 3 | 44.00% |

Distance was an important factor:

| Distance range | Delay rate |
|---|---:|
| 0–200 km | 2.19% |
| 201–400 km | 13.68% |
| 401–600 km | 43.64% |
| Above 600 km | 85.23% |

These findings suggest that longer delivery distances and operationally stretched hubs are associated with higher delay rates.

---

## 7. Part B: Delivery Time Prediction (Regression)

The regression task predicts the expected delivery duration in hours.

### Models evaluated

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

A baseline model using the training-set mean was also evaluated.

### Model selection

Five-fold cross-validation was used to compare Linear Regression and Gradient Boosting.

| Model | Mean CV R² | Standard deviation |
|---|---:|---:|
| Linear Regression | 0.9261 | 0.0048 |
| Gradient Boosting | 0.9353 | 0.0045 |

Gradient Boosting was selected because it achieved the higher mean cross-validation R².

### Final regression test results

| Metric | Result |
|---|---:|
| MAE | 3.4261 hours |
| MSE | 19.9546 |
| RMSE | 4.4671 hours |
| R² | 0.9505 |

**Interpretation:** The final regression model achieved an R² of approximately 0.95, indicating strong predictive performance on the held-out test dataset. Its test-set predictions had an average absolute error of approximately 3.43 hours.

The final model was saved as:

`models/delay_model.pkl`

---

## 8. Part C: Shipment Delay Classification

The classification task predicts whether a shipment is likely to exceed its promised delivery time.

### Models evaluated

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

An all-on-time baseline was also evaluated.

### Feature selection

SelectKBest with ANOVA F-test was used to evaluate different feature counts.

The best validation result was achieved with seven selected features:

1. promised_hours
2. distance_km
3. n_stops
4. rain_flag
5. sale_event_flag
6. hub_tier
7. hub_daily_capacity

### Final classifier configuration

- Algorithm: Random Forest Classifier
- Number of estimators: 100
- Random state: 42
- Selected features: 7
- Classification threshold: 0.50

### Final test results

| Metric | Result |
|---|---:|
| Accuracy | 86.67% |
| Precision | 68.89% |
| Recall | 75.61% |
| F1-score | 72.09% |

**Interpretation:**

- Accuracy measures the overall proportion of correct predictions.
- Precision measures how many shipments flagged as delayed were actually delayed.
- Recall measures how many actual delayed shipments were successfully identified.
- F1-score balances precision and recall.

The classifier achieved an F1-score above the project requirement of 0.60 and recall above the requirement of 0.55.

The final classifier was saved as:

`models/delay_classifier.pkl`

The classification configuration, including the selected features, threshold, and preprocessing medians, was saved as:

`models/model_config.pkl`

---

## 9. Part D: Hub Segmentation and Operational Intelligence

K-Means clustering was used to segment delivery hubs based on operational characteristics.

### Features used for clustering

- Mean delivery hours.
- Delay rate.
- Mean delivery distance.
- Mean number of delivery stops.

The features were standardized before clustering.

### Cluster selection

Values of K from 2 to 5 were compared.

Although K=4 and K=5 achieved higher separation ratios, K=3 was selected for operational simplicity and interpretability. Each of the three clusters contains at least seven hubs.

### Final hub segments

| Segment | Number of hubs | Average delay rate |
|---|---:|---:|
| Standard | 7 | 2.57% |
| Watch | 8 | 15.50% |
| Escalate | 9 | 46.00% |

### Operational interpretation

**Standard**

These hubs demonstrate relatively low delay rates and shorter average delivery durations.

Recommended action: Continue regular monitoring.

**Watch**

These hubs show moderate delivery challenges and require closer monitoring.

Recommended action: Review route planning, workload, and dispatch efficiency.

**Escalate**

These hubs demonstrate higher delay rates and longer average delivery distances.

Recommended action: Prioritize operational reviews and management intervention.

### Tier versus actual performance

Four hubs demonstrated a mismatch between their assigned hub tier and their observed operational segment:

| Hub | Assigned tier | Actual segment | Delay rate |
|---|---|---|---:|
| HUB07 | Tier 1 | Watch | 14% |
| HUB13 | Tier 2 | Escalate | 46% |
| HUB15 | Tier 2 | Escalate | 36% |
| HUB22 | Tier 3 | Watch | 20% |

This demonstrates why operational decisions should consider actual performance rather than relying exclusively on predefined hub categories.

---

## 10. Part E: New Shipment Prediction and Recommended Actions

The project includes a separate dataset containing 48 new shipments.

The intended prediction workflow is:

1. Load the new shipment dataset.
2. Apply the same preprocessing logic used for historical data.
3. Load the saved classification model.
4. Calculate delay probabilities.
5. Compare probabilities against the selected threshold of 0.50.
6. Join the shipment's hub with its operational segment.
7. Generate a recommended action.

### Recommended action logic

| Prediction | Hub segment | Recommended action |
|---|---|---|
| Delayed | Escalate | Escalate to hub manager |
| Delayed | Standard or Watch | Proactive customer call |
| Not delayed | Any segment | No action |

The intended output file is:

`outputs/new_shipment_actions.csv`

It should contain the shipment identifier, hub identifier, predicted delay probability, delay flag, hub segment, and recommended action.

*The 48-shipment prediction output should be generated and verified before reporting its flagged-shipment count or action totals.*

---

## 11. Model Persistence and Reproducibility

The trained models and configuration are saved using Joblib.

Model files:

- `models/delay_model.pkl`
- `models/delay_classifier.pkl`
- `models/model_config.pkl`

A model reload validation was performed for the classifier.

The reloaded classifier produced identical probabilities to the original fitted pipeline on the test dataset, with a maximum probability difference of zero.

This confirms that the saved classifier can be reloaded consistently within the tested environment.

---

## 12. Production Deployment Plan

A production implementation would integrate the prediction pipeline with the logistics company's shipment management system.

### Proposed workflow

1. Receive new shipment records.
2. Validate incoming data.
3. Apply the shared preprocessing pipeline.
4. Generate delivery-time estimates and delay probabilities.
5. Assign operational risk categories.
6. Send alerts to the appropriate hub teams.
7. Record actual delivery outcomes.
8. Monitor model performance and operational trends.

### Proposed monitoring metrics

The following are suggested production monitoring thresholds, not measured results from this synthetic dataset.

| Monitoring metric | Proposed trigger |
|---|---|
| Delay rate | Investigate when above 30% over a rolling 7-day period |
| Missing feature values | Investigate when above 2% |
| Classification recall | Review the model if below 65% |
| Prediction drift | Investigate when a major input feature's distribution changes substantially |

### Retraining strategy

- Monitor model performance weekly.
- Review performance monthly.
- Retrain quarterly or earlier if significant drift occurs.
- Compare predictions with actual delivery outcomes.
- Validate the new model before replacing the production model.

### Sale-event monitoring

Sale-event periods should be monitored separately because shipment volumes and delivery conditions may change during promotional periods.

Compare sale-event performance with similar historical sale periods rather than relying only on ordinary operating periods.

---

## 13. Limitations

1. **Synthetic dataset:** The models were trained using simulated shipment data rather than actual Delhivery operational records.
2. **Limited historical period:** The dataset represents a limited simulated time window and may not capture seasonal patterns.
3. **Operational variables:** Important real-world factors such as rider availability, traffic conditions, failed delivery attempts, and live route changes are not fully represented.
4. **Model generalization:** Performance on synthetic test data does not guarantee similar performance on real-world shipments.
5. **Hub segmentation:** K-Means clustering is sensitive to the selected features and number of clusters.
6. **Threshold selection:** The classification threshold may need adjustment according to operational costs and business priorities.

The system should be validated using real shipment records before being used for actual operational decisions.

---

## 14. Project Structure

```text
Delhivery_Last_Mile_Delay_Intelligence/
│
├── data/
│   ├── raw_shipments.csv
│   ├── hubs.csv
│   ├── new_shipments.csv
│   └── modelling_table.csv
│
├── src/
│   ├── shipment_prep.py
│   └── predict_delay.py
│
├── models/
│   ├── delay_model.pkl
│   ├── delay_classifier.pkl
│   └── model_config.pkl
│
├── outputs/
│   ├── part_a_checks.csv
│   ├── part_a_delay_by_group.csv
│   ├── part_a_correlations.csv
│   ├── part_b_metrics.csv
│   ├── part_b_final_test.csv
│   ├── part_c_models.csv
│   ├── part_c_final_test.csv
│   ├── part_d_hub_profile.csv
│   ├── hub_segments.csv
│   ├── segment_profile.csv
│   ├── tier_vs_segment_mismatch.csv
│   └── new_shipment_actions.csv
│
├── Delhivery_Last_Mile_Delay_Intelligence.ipynb
│
└── README.md
```

*The structure represents the intended final repository. Some files are generated during notebook execution and should be checked before submission.*

---

## 15. How to Run the Project

The project was developed in Google Colab.

1. Open the project notebook in Google Colab.
2. Install the required Python packages.
3. Run the synthetic data-generation cell.
4. Run the data-cleaning and feature-engineering cells.
5. Perform exploratory data analysis.
6. Train and evaluate the regression models.
7. Train and evaluate the classification models.
8. Perform hub segmentation using K-Means.
9. Save the trained models and configuration.
10. Reload and validate the saved models.
11. Run the new-shipment prediction workflow.
12. Review the generated CSV reports.

### Required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

**Important:** In Google Colab, the runtime filesystem is temporary. Download or save important project files to persistent storage before disconnecting or restarting the runtime.

---

## 16. Conclusion

The Delhivery Last-Mile Delay Intelligence Pipeline demonstrates how machine learning can support logistics operations through predictive analytics.

The project successfully developed:

- A regression model for estimating delivery duration.
- A classification model for identifying potentially delayed shipments.
- A hub segmentation framework for operational monitoring.
- A model persistence and reload-validation workflow.
- An action framework for responding to predicted shipment delays.

The final regression model achieved an R² of approximately 0.95, while the classification model achieved an F1-score of approximately 0.72 on the held-out test dataset.

The hub segmentation analysis also identified operational differences that were not fully explained by the predefined hub tiers.

Although the results are based on synthetic data, the project provides a foundation for developing a more advanced logistics intelligence system using real-world shipment information.

**Final goal:** Move from reactive delivery-delay management toward proactive, data-driven last-mile operations.
