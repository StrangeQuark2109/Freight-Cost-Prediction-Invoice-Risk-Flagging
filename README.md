# ML Freight Cost Prediction & Invoice Risk Flagging

A machine learning project that combines freight cost regression, invoice risk classification, and a Streamlit application to support logistics cost analysis and vendor invoice review.

## Project Overview

Freight charges and vendor invoices are important components of supply chain expenditure. This project uses historical vendor invoice and purchase data to estimate freight costs and flag invoices that may require manual review.

The project combines SQL-based data extraction, exploratory data analysis, statistical feature selection, machine learning model evaluation, and an interactive Streamlit portal. Link to the dataset - https://topmate.io/ayushi_mishra/1981139

## Vendor Invoice Intelligence Portal

| Freight Cost Prediction | Invoice Risk Flagging | Prediction Result |
|:---:|:---:|:---:|
| ![Freight Cost Prediction](./Vendor%20Invoice%20Intelligence%20Portal_1-1.png) | ![Invoice Risk Flagging](./Vendor%20Invoice%20Intelligence%20Portal_2-1.png) | ![Prediction Result](./Vendor%20Invoice%20Intelligence%20Portal_2-2.png) |

## Key Features

### 1. Freight Cost Prediction

Predict freight costs using invoice dollar value as the model input.

- Extract vendor invoice data from a SQLite database.
- Explore relationships between invoice value, quantity, and freight charges.
- Compare three regression algorithms:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Evaluate models using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².
- Select the model with the lowest MAE and save it for application use.

### 2. Invoice Risk Flagging

Classify vendor invoices into flagged and non-flagged categories using purchase and invoice information.

The initial risk-labeling logic flags an invoice when either of the following conditions is met:

- The absolute difference between invoice dollars and the associated purchase-item dollars exceeds $5.
- The average purchase receiving delay exceeds 10 days.

These rule-generated labels are used as the target for supervised classification.

The classification workflow includes:

- Aggregating purchase information by purchase order number.
- Calculating invoice timing features, including days from purchase order to invoice and days from invoice to payment.
- Comparing flagged and non-flagged invoices using statistical tests.
- Scaling selected features with `MinMaxScaler`.
- Training Logistic Regression, Decision Tree, and Random Forest classifiers.
- Tuning a Random Forest classifier using `RandomizedSearchCV` with five-fold cross-validation and macro F1 scoring.
- Evaluating classification performance using accuracy, classification reports, and confusion matrices.

**Important:** A flagged invoice is a candidate for manual review, not proof of fraud or an actual accounting error. The target labels are generated from predefined rules rather than independently verified fraud outcomes.

### 3. Interactive Streamlit Portal

The Streamlit application, titled **Vendor Invoice Intelligence Portal**, combines both models in one interface.

**Freight Cost Prediction**

- Enter an invoice dollar amount.
- Generate a predicted freight cost.
- View model MAE and RMSE when those metrics are available in the saved model artifact.

**Invoice Manual Approval Flag**

- Enter invoice quantity, invoice dollars, freight, purchase-order-to-invoice days, total item quantity, total item dollars, and average receiving delay.
- Apply the saved feature scaler.
- Predict whether the invoice should be flagged for manual review.
- View prediction probabilities when supported by the classifier.

## Technology Stack

| Technology           | Purpose                                                         |
| -------------------- | --------------------------------------------------------------- |
| Python               | Data processing and machine learning                            |
| Pandas & NumPy       | Data manipulation and numerical operations                      |
| SQL & SQLite         | Invoice data extraction and purchase-order aggregation          |
| Matplotlib & Seaborn | Exploratory data visualization                                  |
| SciPy                | Statistical significance testing                                |
| Scikit-learn         | Regression, classification, preprocessing, and model evaluation |
| Joblib               | Saving and loading trained models and the scaler                |
| Streamlit            | Interactive prediction interface                                |

## Project Workflow

1. **Data extraction:** Connect to `inventory.db` and query the `vendor_invoice` and `purchases` tables.
2. **Exploratory analysis:** Examine freight cost relationships, invoice characteristics, and feature correlations.
3. **Freight modeling:** Train and compare regression models using invoice dollars to predict freight.
4. **Risk-label construction:** Generate invoice flags using purchase-value discrepancies and receiving delays.
5. **Feature selection:** Compare flagged and non-flagged groups using independent-samples t-tests and select statistically significant features.
6. **Classification:** Scale selected features, train classification models, and tune the Random Forest model using cross-validation.
7. **Evaluation:** Review regression metrics, classification reports, and confusion matrices.
8. **Deployment interface:** Integrate the saved models into a Streamlit application for interactive predictions.

## Getting Started

### Prerequisites

Python 3.10 or a compatible version, along with the required libraries.

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels joblib streamlit
```

### Run the analysis

1. Place `inventory.db` in the database path expected by the notebooks.
2. Open `Freightcostregression.ipynb` and run the cells to analyze freight costs, train the regressors, and save the selected model.
3. Open `InvoiceClassification.ipynb` and run the cells to construct invoice risk labels, train and evaluate classifiers, and save the classifier and scaler.
4. Ensure the saved model files are available at the paths expected by the Streamlit application.

### Launch the application

After creating `app.py` and placing the required model artifacts in the expected locations, run:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the prediction portal.

## Model Evaluation

The project uses the following metrics:

**Regression**

- **MAE:** Average absolute freight prediction error.
- **RMSE:** Gives greater weight to larger prediction errors.
- **R²:** Measures the model's explanatory performance on the evaluation data.

**Classification**

- **Accuracy:** Percentage of predictions classified correctly.
- **Precision and recall:** Help assess the performance of the invoice-flagging classes.
- **F1-score:** Balances precision and recall.
- **Confusion matrix:** Shows the counts of correct and incorrect predictions by class.

Actual metric values should be reported from the notebook outputs rather than assumed in advance.

## Limitations and Considerations

- Freight cost prediction currently uses invoice dollar value as its sole input feature.
- Invoice risk labels are based on predefined rules, so the classifier learns those labels rather than independently verified fraud cases.
- The Streamlit application relies on saved model and scaler files that must be compatible with the training pipeline.
- Predictions should support human review and should not be treated as definitive financial or fraud determinations.

## Future Improvements

- Add more shipment, vendor, quantity, and delivery-related features to freight prediction.
- Validate risk rules against independently reviewed invoice outcomes.
- Improve feature consistency between model training and the Streamlit application.
- Deploy the portal with a managed hosting platform and reliable model-artifact storage.
- Add dashboards for freight cost trends, model performance, and flagged invoice summaries.

---

**Domain:** Supply Chain Analytics · Freight Cost Modeling · Invoice Risk Classification · Machine Learning
