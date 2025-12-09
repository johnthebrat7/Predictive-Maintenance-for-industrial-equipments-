# Predictive-Maintenance-for-industrial-equipments-


This project builds a predictive maintenance classifier using the AI4I 2020 Predictive Maintenance dataset from the UCI Machine Learning Repository. The goal is to predict whether a machine will fail based on its operating conditions and derived features, enabling proactive maintenance and reduced downtime.

The workflow starts with data loading and exploratory data analysis. Using pandas, seaborn, Plotly, and a correlation heatmap, the code inspects data types, distributions, correlations, and class imbalance for the target variable “Machine failure” and individual failure modes (TWF, HDF, PWF, OSF, RNF). A ydata-profiling (pandas‑profiling) report provides an automated, in‑depth overview of data quality and structure.

Next, the project performs feature engineering to better capture machine behavior: it creates power (Rotational speed * Torque), power wear (Power * Tool wear), temperature difference, and temperature–power interaction features. Non‑informative identifiers are dropped, and categorical variables (e.g., Type) are one‑hot encoded. A pairplot and PCA projection are used to visualize the feature space colored by failure status, giving intuition about class separability. The Hopkins statistic is computed to assess the cluster tendency of the data.

The dataset is then split into training and test sets with stratification, and features are normalized using MinMaxScaler. Several classification models are trained and compared: K‑Nearest Neighbors, Decision Tree, Random Forest, and Gradient Boosting. For each model, the code reports accuracy, precision, recall, F1‑score, Matthews correlation coefficient, and timing (train and predict), along with confusion matrices for visual error analysis. For tree‑based models, feature importance plots highlight which engineered and original features contribute most to predicting machine failures.

Overall, this project demonstrates a complete, practical workflow for supervised learning in a predictive maintenance setting: from raw data to engineered features, visualization, model training, evaluation, and interpretability.
