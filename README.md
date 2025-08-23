# Smart Network Optimization for Enhanced Connectivity

**Goal:** Monitor enterprise network traffic, predict imminent congestion/high utilization, and recommend bandwidth reallocation to sustain optimal performance.

## Repository Structure
```
smart-network-optimization/
├── configs/
│   └── config.yaml
├── data/
│   ├── README.md
│   ├── processed/              # generated
│   └── raw/                    # raw dumps (optional)
├── models/                     # saved models
├── notebooks/
│   └── 01_eda.ipynb
├── outputs/                    # predictions/metrics
├── scripts/
│   ├── train.py
│   └── predict.py
├── src/
│   ├── data/
│   ├── models/
│   ├── reallocator/
│   └── utils/
├── tests/
│   └── test_data_loading.py
├── .github/workflows/ci.yml
├── .gitignore
├── LICENSE
└── requirements.txt
```



## Approach (baseline)

- Supervised **classification** for *congestion in next interval*: features include traffic volume, latency, bandwidth used/allocated, utilization ratio, time-of-day/week, lags, and rolling means.
- Simple **reallocation policy**: shift bandwidth from underutilized to overutilized devices/links based on predicted risk and a utilization threshold.
- Extensions: time-series forecasting (ARIMA/LSTM), anomaly detection for outages, and reinforcement learning for closed-loop control.
# Proactive Network Congestion Prediction & Bandwidth Management

This project implements a machine learning system to proactively predict network congestion on a set of routers and provide automated, actionable recommendations for bandwidth allocation. It leverages XGBoost models trained on time-series network data to forecast congestion probabilities and drive a rule-based recommendation engine.

---

###  Project Goal

The primary objective is to move from a reactive to a **proactive** network management strategy. By anticipating congestion before it occurs, the system aims to:
-   Improve network performance and reliability.
-   Optimize resource allocation by adjusting bandwidth dynamically.
-   Reduce manual intervention and operational overhead.

---

###  System Architecture & Data Pipelines

The project is built around two distinct data pipelines to ensure robust development, testing, and validation:

1.  **Pipeline P (Processed Data)** 
    * **Source:** Raw, historical log files from multiple sources.
    * **Process:** The `cleaning_given_data.ipynb` notebook ingests, cleans, merges, and features-engineers this data.
    * **Output:** `model_ready_dataset_p.csv`. This dataset represents a real-world scenario.

2.  **Pipeline G (Generated Data)** 
    * **Source:** Synthetically generated from scratch.
    * **Process:** The `dataset_generator.ipynb` notebook creates a clean, well-structured dataset with realistic, simulated network patterns (e.g., peak business hours, evening usage spikes).
    * **Output:** `model_ready_dataset_g.csv`. This dataset provides a controlled environment for model testing and serves as a performance baseline.

---

###  Notebooks

This repository contains the following notebooks, which form the core of the project.

#### ### 1.  `cleaning_given_data.ipynb`

This notebook handles the entire ETL (Extract, Transform, Load) process for the real-world data pipeline.

* **Purpose**: To process and merge multiple raw data sources into a single, model-ready dataset.
* **Inputs**:
    * `Router_A_router_log_15_days.csv`
    * `Router_B_router_log_15_days.csv`
    * `Router_C_router_log_15_days.csv`
    * `application_usage.csv`
    * `user_activity.csv`
    * `external_factors.csv`
    * `configuration_history.csv`
* **Key Steps**:
    1.  **Load & Consolidate**: Loads all raw CSVs and combines the individual router logs.
    2.  **Feature Engineering**: Creates the target variable `New_Flag` (congestion event) based on a logical rule: a congestion event is flagged if `utilization > 85%` OR (`utilization > 70%` AND `latency > 45ms`).
    3.  **Aggregate**: Computes daily summaries from detailed logs (e.g., `total_peak_app_traffic`, `total_logins`, `Num_Config_Changes`).
    4.  **Merge & Clean**: Merges all data frames on `Date` and/or `Device Name` and fills any resulting `NaN` values.
    5.  **Export**: Saves the final, sorted dataset as `final_model_ready_dataset.csv`.

#### ### 2.  `dataset_generator.ipynb`

This notebook generates a high-quality, synthetic dataset for controlled model training and evaluation.

* **Purpose**: To synthetically generate a complete and realistic dataset that mimics real-world network behavior.
* **Key Steps**:
    1.  **Simulate Router Logs**: Generates hourly log data, programmatically increasing traffic and latency to simulate **peak business hours** and **evening usage spikes**.
    2.  **Simulate Daily Summaries**: Creates synthetic daily data for application usage, user activity, and external events (e.g., outages, maintenance).
    3.  **Integrate & Model Events**: Merges all synthetic data and simulates the impact of events (e.g., doubling latency during a network outage).
    4.  **Create Target Variable**: Applies a rule-based function (`create_new_flag`) to label congestion events.
    5.  **Export**: Saves the final dataset as `new_realistic_dataset_v2.csv`.

#### ### 3.  `model+recommendation_p.ipynb` & `model+recommendation_g.ipynb`

These notebooks are the core of the project, containing the logic for model training, prediction, and recommendation. They are identical except for the dataset they use (`_p` for processed, `_g` for generated).

* **Purpose**: To train congestion prediction models and use them to generate automated bandwidth adjustment recommendations.
* **Core Components**:
    * **Feature Engineering (`create_training_samples`)**:
        * This function implements a **sliding window** approach. To predict congestion at hour `T`, it uses data from the previous 12 hours (`T-12` to `T-1`).
        * It creates a single feature vector of $12 \text{ hours} \times 3 \text{ routers} \times 9 \text{ metrics} = 324$ features, capturing the complete network state over the window.
    * **Model Training**:
        * An **XGBoost Classifier** is used for its high performance.
        * A **multi-model strategy** is employed: three independent models are trained, one specializing in the patterns of each router (`Router_A`, `Router_B`, `Router_C`).
    * **Prediction (`predict_congestion_proba`)**:
        * This function takes a timestamp, prepares the 12-hour historical feature vector, and returns a **congestion probability** (0.0 to 1.0) for each router.
    * **Recommendation Engine (`bandwidth_recommendation`)**:
        * This is a rule-based system that translates the model's probability scores into actionable advice. It uses both the **predicted probability** and the current **bandwidth utilization**.
        * **Sample Rules**:
            * `IF congestion_prob >= 0.8`: **HIGH RISK** -> Recommend `increase_bandwidth`.
            * `IF congestion_prob <= 0.2` and `utilization <= 0.4%`: **OPTIMIZE** -> Recommend `decrease_bandwidth`.
            * Other conditions result in `maintain` or `monitor` actions.
    * **Evaluation & Serialization**:
        * Model performance is evaluated using **Accuracy**, **Brier Score**, **RMSE**, and **Confusion Matrices**.
        * The final trained models and metadata are saved to a `.pkl` file for easy deployment.

