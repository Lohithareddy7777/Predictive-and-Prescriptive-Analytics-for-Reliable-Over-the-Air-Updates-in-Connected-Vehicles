🚗 Risk-Aware OTA Update Scheduling Using Predictive Analytics
📌 About the Project

Modern connected vehicles receive software updates through Over-The-Air (OTA) mechanisms. While OTA updates are convenient, they often fail due to reasons such as low battery charge, poor network connectivity, large update sizes, or the vehicle being in motion. These failures increase service costs, frustrate users, and may even introduce safety or security risks.

This project focuses on predicting the likelihood of an OTA update failure before the update starts and using that prediction to decide whether the update should be executed immediately, delayed, or deferred.
Instead of stopping at prediction, the project emphasizes practical decision-making, which is essential for real automotive systems.

🎯 What This Project Tries to Achieve

Simulate realistic OTA update scenarios using vehicle telemetry data

Train a machine learning model to estimate OTA failure risk

Understand which factors most strongly influence update failures

Convert model outputs into clear scheduling recommendations

Estimate how much OTA failure rates can be reduced using this approach

📊 Dataset Overview

Since real OTA logs are confidential and not publicly available, a synthetic dataset was created using realistic assumptions inspired by connected vehicle behavior.

Input Features
Feature	Description
battery_soc	Battery State of Charge (%)
signal_strength_rssi	Cellular signal strength in dBm
vehicle_is_moving	: Indicates whether the vehicle is in motion
prior_fail_count	Number of previous OTA failures
update_size_mb	Size of the OTA update
time_of_day	Time when the update is attempted
Target Variable

update_failure

0 → Update successful

1 → Update failed

Failure probabilities were intentionally increased under conditions such as low battery, weak signal, large updates, and vehicle movement to reflect real-world OTA behavior.

🧠 Approach and Methodology
1. Data Simulation

OTA update scenarios are generated with controlled randomness

Failure probability is calculated using domain-inspired rules

The dataset contains a realistic imbalance between successful and failed updates

2. Machine Learning Model

An XGBoost classifier is used to predict OTA failure

Categorical features are one-hot encoded

Data is split into training and testing sets using stratified sampling

3. Model Evaluation

The model is evaluated using:

Accuracy

Precision, recall, and F1-score

Confusion matrix

Feature importance analysis to understand decision drivers

4. Risk-Based OTA Scheduling

Predicted failure probabilities are grouped based on:

Battery SoC (Low / Medium / High)

Signal strength (Weak / Moderate / Strong)

From this, a decision table is generated that recommends:

Allow OTA

Delay / Retry Later

Defer OTA

5. Business Impact Estimation

The project compares:

Baseline OTA failure rate

Failure rate after applying risk-aware scheduling

This helps quantify the practical value of the approach.

📈 Key Outcomes
Important Factors Influencing OTA Failure

Update size

Battery State of Charge

Network signal strength

Previous OTA failures

Vehicle movement

Example OTA Scheduling Decisions
Battery SoC	Signal Strength	Avg Failure Risk	Recommendation
Low	Weak	Very High	Defer OTA
Medium	Weak	Moderate	Delay / Retry Later
High	Strong	Low	Allow OTA
Business Impact

Baseline failure rate: ~43%

Optimized failure rate: ~23%

Failure reduction: ~48%

This shows that even simple scheduling rules, when guided by ML predictions, can significantly improve OTA reliability.

🛠️ Tools & Technologies

Python

Pandas & NumPy

Scikit-learn

XGBoost

Matplotlib
