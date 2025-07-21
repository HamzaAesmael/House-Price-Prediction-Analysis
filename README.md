## Project Overview

This script performs the following steps:
1.  **Data Ingestion**: Downloads the housing dataset from a remote source.
2.  **Data Wrangling**: Cleans the data by handling missing values and dropping irrelevant columns.
3.  **Exploratory Data Analysis (EDA)**: Investigates relationships between features (like `sqft_living` and `waterfront`) and the target variable (`price`) using visualizations like box plots and regression plots.
4.  **Model Development**: Builds several regression models to predict house prices:
    *   Simple Linear Regression
    *   Multiple Linear Regression
    *   A `Pipeline` with Polynomial Features for a more complex model.
5.  **Model Evaluation and Refinement**: Splits the data into training and testing sets and uses Ridge Regression to prevent overfitting, ultimately comparing model performance with R-squared scores.

## Libraries Used

This project relies on the following Python libraries:
*   pandas
*   numpy
*   scikit-learn
*   matplotlib
*   seaborn
  the link for the data : https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv
