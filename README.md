# Fraud Detection Machine Learning Pipeline

An end-to-end machine learning project for detecting fraudulent financial transactions using Python and Scikit-learn.

The project covers the complete workflow from **exploratory data analysis (EDA)** and **feature engineering** to **data preprocessing, model training, evaluation, and model serialization**.

The final model uses **Logistic Regression with class balancing** and is implemented inside a Scikit-learn pipeline that combines numerical scaling and categorical encoding.

---

## 📌 Project Overview

Financial transaction fraud is a highly imbalanced classification problem where fraudulent transactions represent only a small portion of all transactions.

The objective of this project is to analyze transaction data, identify patterns associated with fraudulent behavior, engineer meaningful features, and train a machine learning model capable of classifying transactions as either legitimate or fraudulent.

The complete workflow is:

```text
Raw Transaction Dataset
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Feature Selection
        │
        ▼
Train / Test Split
        │
        ▼
Preprocessing Pipeline
        │
        ├── StandardScaler
        │
        └── OneHotEncoder
        │
        ▼
Logistic Regression
        │
        ▼
Model Evaluation
        │
        ▼
Save Complete Pipeline
        │
        ▼
Fraud Detection Model
```

---

# 📊 Dataset

The dataset contains financial transaction records with information about transaction types, amounts, account balances, and fraud labels.

Important features include:

| Feature          | Description                                                      |
| ---------------- | ---------------------------------------------------------------- |
| `type`           | Type of financial transaction                                    |
| `amount`         | Transaction amount                                               |
| `oldbalanceOrg`  | Origin account balance before the transaction                    |
| `newbalanceOrig` | Origin account balance after the transaction                     |
| `oldbalanceDest` | Destination account balance before the transaction               |
| `newbalanceDest` | Destination account balance after the transaction                |
| `isFraud`        | Target variable indicating whether the transaction is fraudulent |

The target variable is:

```text
isFraud

0 → Legitimate transaction
1 → Fraudulent transaction
```

---

# 🔎 Exploratory Data Analysis

The first stage of the project focuses on understanding the dataset and identifying patterns that could be useful for fraud detection.

## Dataset Inspection

The dataset is initially inspected using:

```python
df.head()
df.info()
df.shape
df.columns
```

These operations are used to understand:

* Dataset dimensions
* Column names
* Data types
* Sample records
* Overall structure

---

## Missing Value Analysis

Missing values are checked using:

```python
df.isnull().sum().sum()
```

This determines the total number of missing values across the dataset.

---

## Fraud Distribution

The distribution of legitimate and fraudulent transactions is examined using:

```python
df["isFraud"].value_counts()
```

The percentage of fraudulent transactions is also calculated.

This is particularly important because fraud detection datasets are usually **highly imbalanced**.

---

# 📈 Transaction Type Analysis

The frequency of each transaction type is visualized using a bar chart.

```python
df["type"].value_counts().plot(kind="bar")
```

This provides an overview of the most common transaction types in the dataset.

---

## Fraud Rate by Transaction Type

The fraud rate for each transaction type is calculated using:

```python
fraudByType = (
    df.groupby("type")["isFraud"]
      .mean()
      .sort_values(ascending=False)
)
```

### Why `groupby()`?

`groupby()` divides the dataset into groups based on a column.

In this case:

```python
df.groupby("type")
```

creates separate groups for transaction types such as:

```text
PAYMENT
TRANSFER
CASH_OUT
CASH_IN
DEBIT
```

Then:

```python
["isFraud"].mean()
```

calculates the average fraud label for each group.

Because:

```text
0 = legitimate
1 = fraud
```

the mean represents the **fraud rate**.

This allows the project to identify which transaction types have a higher concentration of fraudulent activity.

---

# 💰 Transaction Amount Analysis

Statistical information about transaction amounts is explored using:

```python
df["amount"].describe()
```

A logarithmic transformation is also used for visualization:

```python
np.log1p(df["amount"])
```

This helps visualize the distribution when transaction amounts are highly skewed.

A histogram with KDE is generated using Seaborn:

```python
sns.histplot(
    np.log1p(df["amount"]),
    bins=100,
    kde=True
)
```

The project also uses a boxen plot to compare transaction amounts between legitimate and fraudulent transactions:

```python
sns.boxenplot(
    data=df[df["amount"] < 50000],
    x="isFraud",
    y="amount"
)
```

The amount filter makes the visualization easier to interpret by reducing the effect of extremely large transactions.

---

# 🧮 Feature Engineering

Feature engineering is performed to create additional information from the existing transaction attributes.

Two balance-difference features are created.

## Origin Balance Difference

```python
df["balanceDiffOrig"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)
```

This represents the change in the origin account's balance.

---

## Destination Balance Difference

```python
df["balanceDiffDest"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)
```

This represents the change in the destination account's balance.

These engineered features help provide additional information about how money moves between accounts.

The project also checks for negative balance differences to identify potentially unusual balance behavior.

---

# ⏱️ Fraud Over Time

Fraudulent transactions are analyzed over the transaction `step` variable.

```python
FraudsPerStep = (
    df[df["isFraud"] == 1]["step"]
    .value_counts()
    .sort_index()
)
```

A line plot is then used to visualize fraud activity over time.

```python
plt.plot(
    FraudsPerStep.index,
    FraudsPerStep.values
)
```

After the temporal analysis is completed, the `step` column is removed from the modeling dataset:

```python
df.drop(columns="step", inplace=True)
```

---

# 👤 Account Analysis

The project investigates transaction activity at the account level.

### Top senders

```python
df["nameOrig"].value_counts().head(10)
```

Identifies the most frequently appearing origin accounts.

### Top receivers

```python
df["nameDest"].value_counts().head(10)
```

Identifies the most frequently appearing destination accounts.

### Accounts involved in fraud

```python
df[df["isFraud"] == 1]["nameOrig"].value_counts().head(10)
```

Identifies origin accounts that appear most frequently in fraudulent transactions.

---

# 🔍 Transaction Pattern Analysis

The project focuses on `TRANSFER` and `CASH_OUT` transactions:

```python
FraudTypes = df[
    df["type"].isin(["TRANSFER", "CASH_OUT"])
]
```

### Why `isin()`?

`isin()` checks whether a value belongs to a specified collection.

For example:

```python
df["type"].isin(["TRANSFER", "CASH_OUT"])
```

means:

> Select rows where the transaction type is either `TRANSFER` or `CASH_OUT`.

This is more concise than writing multiple conditions with `|`.

The filtered transactions are then visualized using:

```python
sns.countplot(
    data=FraudTypes,
    x="type",
    hue="isFraud"
)
```

This allows legitimate and fraudulent transactions to be compared within these transaction categories.

---

# 🚨 Suspicious Balance Pattern

The project also investigates transactions where the origin account had a positive balance before the transaction but ended with a zero balance.

```python
ZeroAfterTransfer = df[
    (df["oldbalanceOrg"] > 0) &
    (df["newbalanceOrig"] == 0) &
    (df["type"].isin(["TRANSFER", "CASH_OUT"]))
]
```

This combines multiple conditions to identify a potentially suspicious transaction pattern.

---

# 🔗 Correlation Analysis

Correlation between important numerical variables and the fraud target is calculated using:

```python
corr = df[
    [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud"
    ]
].corr()
```

The correlation matrix is visualized with a Seaborn heatmap:

```python
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
```

This provides a visual overview of linear relationships between the numerical variables.

---

# 🤖 Machine Learning Model

After completing the EDA and feature engineering stages, the project moves to machine learning.

The final model is a:

**Logistic Regression classifier**

Logistic Regression was selected as a classification model for predicting whether a transaction is fraudulent.

---

# 🧹 Feature Selection

The following columns are removed before model training:

```python
dfModel = df.drop(
    ["nameOrig", "nameDest", "isFlaggedFraud"],
    axis=1
)
```

The account identifiers `nameOrig` and `nameDest` are removed because they are high-cardinality identifiers rather than useful generalized numerical features for this model.

`isFlaggedFraud` is also excluded from the model features.

The target variable remains:

```python
y = df["isFraud"]
```

and the input features are:

```python
x = dfModel.drop(["isFraud"], axis=1)
```

---

# ⚙️ Data Preprocessing

The project contains both numerical and categorical features.

### Categorical feature

```python
String = ["type"]
```

### Numerical features

```python
numerical = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]
```

A `ColumnTransformer` is used to apply the appropriate preprocessing to each feature type.

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical
        ),
        (
            "Str",
            OneHotEncoder(drop="first"),
            String
        )
    ],
    remainder="drop"
)
```

### Numerical preprocessing

`StandardScaler` standardizes numerical features so they are placed on a comparable scale.

### Categorical preprocessing

`OneHotEncoder` converts the categorical `type` feature into numerical columns that the machine learning model can use.

`drop="first"` removes one category to avoid redundant dummy variables.

---

# 🔄 Train/Test Split

The dataset is divided into training and testing sets:

```python
xTrain, xTest, yTrain, yTest = train_test_split(
    x,
    y,
    test_size=0.3,
    stratify=y
)
```

The dataset is split into:

* **70% training data**
* **30% testing data**

The `stratify=y` parameter is particularly important for this project because the target classes are imbalanced.

It helps maintain a similar fraud/non-fraud class distribution in both the training and testing sets.

---

# 🧠 Machine Learning Pipeline

The preprocessing and model are combined into a single Scikit-learn pipeline:

```python
pipline = Pipeline([
    ("prep", preprocessor),
    (
        "clf",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        )
    )
])
```

This pipeline ensures that preprocessing and prediction are performed consistently.

The model uses:

```python
class_weight="balanced"
```

to give additional importance to the minority class.

This is useful because fraudulent transactions are significantly less common than legitimate transactions.

The model is then trained using:

```python
pipline.fit(xTrain, yTrain)
```

---

# 📊 Model Evaluation

Predictions are generated using:

```python
y_pred = pipline.predict(xTest)
```

The model is evaluated using a classification report:

```python
classification_report(yTest, y_pred)
```

The classification report provides metrics including:

* Precision
* Recall
* F1-score
* Support

A confusion matrix is also generated:

```python
confusion_matrix(yTest, y_pred)
```

This provides a detailed view of:

```text
                 Predicted
              Normal   Fraud

Actual Normal    TN       FP

Actual Fraud     FN       TP
```

The model score is also calculated using:

```python
pipline.score(xTest, yTest)
```

---

# 💾 Model Serialization

After training and evaluation, the complete machine learning pipeline is saved using Joblib:

```python
import joblib

joblib.dump(
    pipline,
    "Fraud_detection_pipline.pkl"
)
```

The saved `.pkl` file contains the complete pipeline, including:

```text
Preprocessing
     │
     ├── StandardScaler
     │
     └── OneHotEncoder
     
     ↓

Logistic Regression
```

This allows the trained model to be loaded later without retraining it.

---

# 🛠️ Technologies Used

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| Python           | Programming language                   |
| Pandas           | Data manipulation and analysis         |
| NumPy            | Numerical operations                   |
| Matplotlib       | Data visualization                     |
| Seaborn          | Statistical visualization              |
| Scikit-learn     | Machine learning and preprocessing     |
| Joblib           | Model serialization                    |
| Jupyter Notebook | Development and experimentation        |
| Git / GitHub     | Version control and project management |

---

# 📁 Project Structure

```text
fraud-detection-ml-pipeline/
│
├── AIML Dataset.csv
├── analysis_model.ipynb
├── Fraud_detection_pipline.pkl
├── README.md
└── requirements.txt
```

As the project is extended with a local application, additional files and directories can be added for deployment.

---

# 🚀 Running the Project

## 1. Clone the repository

```bash
git clone https://github.com/your-username/fraud-detection-ml-pipeline.git
```

## 2. Navigate to the project

```bash
cd fraud-detection-ml-pipeline
```

## 3. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the notebook

Open:

```text
analysis_model.ipynb
```

and execute the cells sequentially.

The notebook performs the complete analysis, trains the model, evaluates it, and generates:

```text
Fraud_detection_pipline.pkl
```

---

# 📌 Key Concepts Demonstrated

This project demonstrates practical knowledge of:

* Exploratory Data Analysis
* DataFrame manipulation with Pandas
* `groupby()`
* `value_counts()`
* Boolean filtering
* `isin()`
* Feature engineering
* Data visualization
* Distribution analysis
* Correlation analysis
* Categorical encoding
* Numerical scaling
* Train/test splitting
* Stratified sampling
* Machine learning pipelines
* Logistic Regression
* Imbalanced classification
* Precision, Recall and F1-score
* Confusion matrices
* Model serialization

---

# 🔮 Future Improvements

Possible extensions to the project include:

* Comparing Logistic Regression with tree-based models
* Hyperparameter tuning
* Cross-validation
* More advanced fraud-specific feature engineering
* Threshold optimization
* Precision-Recall curve analysis
* ROC-AUC evaluation
* Model explainability
* Building a local web interface
* Deploying the model as an API
* Containerizing the application with Docker
* Cloud deployment

---

# 🎯 Conclusion

This project demonstrates an end-to-end approach to financial fraud detection using machine learning.

The workflow begins with understanding the transaction data through EDA, continues with feature engineering and preprocessing, and ends with a trained and evaluated Logistic Regression model.

The final Scikit-learn pipeline combines preprocessing and classification into a single reusable object, which is then serialized using Joblib for future prediction and deployment.

The project provides a practical foundation for developing a complete fraud detection application.

---

## 👨‍💻 Author

**Khaled Mansour**

Built as a practical machine learning project to develop experience with data analysis, feature engineering, classification, model evaluation, and deployment.
