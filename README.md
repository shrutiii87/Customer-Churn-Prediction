# 📉 Customer Churn Prediction — Multi-Algorithm Classification Showdown

<img width="1600" height="560" alt="download" src="https://github.com/user-attachments/assets/0414ce86-6afd-409d-9f00-c9c897d94272" />

---

This project is built on supervised classification in machine learning. It covers KNN, Naive Bayes, SVM and Decision Tree, compares how each algorithm separates churners from loyal customers, handles class imbalance with SMOTE and class_weight, and applies all of them to the IBM Telco Customer Churn dataset to help a telecom retention team choose the model to deploy.

---

## 📄 Problem Statement

<table>
<tr>
<td width="50%" valign="top">


### 📉 Customer Churn Prediction — Multi-Algorithm Classification Showdown

telecom company similar to Jio or Airtel. The company holds a dataset of 7,043 customers and is losing paying subscribers every month without being able to predict who is about to leave.

**Classification — predict whether a customer will churn (Churn: Yes = 1 churned, No = 0 retained)**

The task is to explore and clean the data, engineer features, balance the classes, build KNN, Naive Bayes, SVM and Decision Tree models, tune them .

measure everything with accuracy, precision, recall, F1, ROC-AUC and training time, analyse the customers the best model misses, and save the final model as a reusable pipeline.

 
</td>
<td width="50%" valign="top">

<img width="1376" height="1543" alt="churn_final_no_ml_workflow_padded" src="https://github.com/user-attachments/assets/65b75662-017b-4bcc-bb0d-4f374d2ebdb7" />


</td>
</tr>
</table>


---

## 📂 Project Files

| 📄 File / Folder | 📌 Description |
|------------------|----------------|
| 📓 `CustomerChurn_SupervisedLearning.ipynb` | Main notebook — the complete, annotated theory, EDA, preprocessing, modelling, evaluation, error analysis and pipeline code |
| 📊 `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Raw IBM Telco customer dataset (7,043 records) |
| 🤖 `churn_model.pkl` | Saved final pipeline (StandardScaler + Decision Tree) |
| 📝 `summary_report.md` | ~450-word summary report for the business team |
| 📘 `README.md` | Project documentation and workflow guide |
| 📂 `Visuals` | Folder of the output visuals/graphs |

---

# 🛠 Tools Used

<div>
<img src="https://img.shields.io/badge/Python-172B4D?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter%20Notebook-F28C28?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-E8A04A?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-D97726?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-C66A13?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-B87535?style=for-the-badge&logo=matplotlib&logoColor=white"/>
<img src="https://img.shields.io/badge/Seaborn-8C603D?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/imbalanced--learn-A85D24?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/KNN-F28C28?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/Naive%20Bayes-D98232?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/SVM-172B4D?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/Decision%20Tree-29466C?style=for-the-badge&logoColor=white"/>
</div>
---

## 🎬 Project Demo

<a href="https://drive.google.com/file/d/1wjJbPCsCKkx1DvVgVPra_X10lBQl_AaA/view?usp=sharing">
<img src="https://img.shields.io/badge/WATCH%20VIDEO-F28C28?style=for-the-badge&logo=googledrive&logoColor=white"/>
</a>

---


### 🧬 Dataset Structure — IBM Telco Customer Churn Dataset

| Field Name | Data Type | Description | Notes |
|------------|-----------|-------------|-------|
| `customerID` | Object | Unique identifier for each customer | Identifier only — dropped |
| `gender` | Object | Male / Female | Model feature (mapped to 1 / 0) |
| `SeniorCitizen` | Integer | 1 = senior citizen, 0 = not | Model feature |
| `Partner` | Object | Has a partner (Yes / No) | Model feature (mapped to 1 / 0) |
| `Dependents` | Object | Has dependents (Yes / No) | Model feature (mapped to 1 / 0) |
| `tenure` | Integer | Months the customer has stayed (0–72) | Model feature |
| `PhoneService` | Object | Has phone service (Yes / No) | Model feature (mapped to 1 / 0) |
| `MultipleLines` | Object | Has multiple phone lines | Model feature — "No phone service" merged into "No" |
| `InternetService` | Object | DSL / Fiber optic / No | Model feature (one-hot encoded) |
| `OnlineSecurity` | Object | Online security add-on | Replaced by `num_services` |
| `OnlineBackup` | Object | Online backup add-on | Replaced by `num_services` |
| `DeviceProtection` | Object | Device protection add-on | Replaced by `num_services` |
| `TechSupport` | Object | Tech support add-on | Replaced by `num_services` |
| `StreamingTV` | Object | Streaming TV add-on | Replaced by `num_services` |
| `StreamingMovies` | Object | Streaming movies add-on | Replaced by `num_services` |
| `Contract` | Object | Month-to-month / One year / Two year | Model feature (one-hot encoded) |
| `PaperlessBilling` | Object | Paperless billing (Yes / No) | Model feature (mapped to 1 / 0) |
| `PaymentMethod` | Object | Payment method (4 types) | Model feature (one-hot encoded) |
| `MonthlyCharges` | Float | Monthly bill (18.25–118.75) | Model feature |
| `TotalCharges` | Object → Float | Total amount billed | Model feature — 11 blank rows fixed with median |
| `Churn` | Object | Yes = churned, No = retained | 🎯 Classification target (73.5% retained / 26.5% churned) |

---

## 🧠 Part 1: Problem Framing & Theory Notes

### 1. What is Customer Churn? Why is predicting churn valuable (CAC vs CLV)?

🎯 Insight: Churn means a paying customer leaves the service. Getting a new customer (CAC — Customer Acquisition Cost) is much more expensive than keeping an old one, and every churned customer also takes away future revenue (CLV — Customer Lifetime Value). If Jio/Airtel can predict who will leave, the retention team can give offers in time and protect the revenue.

---

### 2. Confusion Matrix — TP, TN, FP, FN and their business cost.

🎯 Insight: TP = churned and predicted churn, TN = stayed and predicted stay, FP = stayed but predicted churn, FN = churned but predicted stay. A False Negative is costly because we lose the customer and his CLV without ever trying to save him. A False Positive is cheap, we only waste a small offer or call on a customer who would stay anyway.

---

### 3. Class Imbalance and SMOTE.

🎯 Insight: Class imbalance means one class has far less data than the other, so the model becomes biased to the majority class. The Telco data is imbalanced (26.5% churned vs 73.5% stayed). SMOTE creates synthetic minority samples using nearest neighbours so both classes become equal in the training data.

---

### 4. How KNN, Naive Bayes, SVM and Decision Tree work.

🎯 Insight: KNN takes the majority class among the k nearest customers. Naive Bayes uses Bayes theorem with an independence assumption. SVM finds the maximum-margin boundary (non-linear with the RBF kernel). Decision Tree keeps splitting data with yes/no questions on features until each leaf gives a class.

---

### 5. Precision vs Recall for churn.

🎯 Insight: Precision matters more when the retention action is costly or calling capacity is limited, for example if the team can call only ~500 customers a day or gives a big discount, wrongly flagged loyal customers waste money and call slots.

---

## 🧪 Part 2: Dataset Loading & Exploratory Data Analysis

<img width="1600" height="560" alt="download" src="https://github.com/user-attachments/assets/e9765d19-31a5-40f4-b2b0-bab6e244db04" />


### 6. Load the data and inspect shape, info, describe and head.

```python
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.shape)
df.info()
```

🎯 Insight: The dataset has 7,043 rows and 21 columns with no null values, but `TotalCharges` loads as object instead of a number.

---

### 7. Convert TotalCharges to float64 and investigate why it was object.

```python
blank = df[df['TotalCharges'].str.strip() == '']
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
```

🎯 Insight: 11 rows contain only a blank space, which made pandas read the column as object. All 11 have tenure = 0, meaning they are new customers not billed yet. They are converted to NaN and filled with the median later.

---

### 8. Check class balance.

```python
print(df['Churn'].value_counts(normalize=True) * 100)
```

🎯 Insight: 1,869 customers churned (26.5%) and 5,174 stayed (73.5%), roughly a 3 : 1 imbalance, so accuracy alone would be misleading.

---

### 9. Univariate analysis — target, numeric histograms and key categorical features.


🎯 Insight: Countplot, histograms for tenure / MonthlyCharges / TotalCharges, and countplots for Contract, InternetService and PaymentMethod were plotted to understand distributions before modelling.

---

### 10. Churn rate by Contract type.

```python
rate = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
```

🎯 Insight: Month-to-month contracts churn the most (42.7%), then One year (11.3%) and Two year (2.8%) — about 15x higher for month-to-month than for two-year contracts.

---

### 11. Churn rate by tenure bucket.

```python
bucket = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 72], labels=['0-12', '13-24', '25-48', '49-72'])
```

🎯 Insight: Churn falls steadily with tenure: 47.4% (0–12 months), 28.7% (13–24), 20.4% (25–48) and 9.5% (49–72). New customers are the most at risk.

---

### 12. MonthlyCharges vs Churn.

```python
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
```

🎯 Insight: Yes, churners pay more per month — median 79.65 against 64.43 for non-churners.

---

### 13. Correlation heatmap.

🎯 Insight: tenure and TotalCharges have a strong correlation (0.83) and MonthlyCharges and TotalCharges a good one (0.65). Other numeric features have weak correlation.

---

### 14. Highest-risk segment check.

```python
seg = df[(df['Contract'] == 'Month-to-month') & (df['MonthlyCharges'] > 70) & (df['tenure'] <= 12)]
```

🎯 Insight: The EDA confirms it. 856 customers fall in this segment and 69.0% of them churn, against 26.5% overall.

---

## 🔧 Part 3: Data Preprocessing & Feature Engineering

<img width="1600" height="560" alt="download" src="https://github.com/user-attachments/assets/8d890a51-0d6c-4e88-8dda-0ac43f0f5f8a" />


### 15. Drop and clean.

```python
df = df.drop('customerID', axis=1)
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
```

🎯 Insight: customerID has no predictive value so it is dropped, and the 11 blank TotalCharges values are filled with the median, leaving 0 missing values.

---

### 16. Feature engineering.

```python
df['tenure_group'] = pd.cut(df['tenure'], bins=[-1, 12, 36, 60, 100], labels=['New', 'Mid', 'Senior', 'Loyal'])
df['num_services'] = (df[service_cols] == 'Yes').sum(axis=1)
df['AutoPay'] = df['PaymentMethod'].str.contains('automatic').astype(int)
```

🎯 Insight: Three new features were created — `tenure_group` (New / Mid / Senior / Loyal), `num_services` (count of the 6 add-on services) and `AutoPay` (1 if the payment method is automatic).

---

### 17. Encoding.

```python
df[col] = df[col].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], dtype=int)
df['tenure_group'] = df['tenure_group'].map({'New': 0, 'Mid': 1, 'Senior': 2, 'Loyal': 3})
```

🎯 Insight: Binary columns are mapped to 1/0, three nominal columns are one-hot encoded, and tenure_group is label encoded in its natural order. After dropping the 6 replaced add-on columns the data has 24 columns (23 features + Churn).

---

### 18. Scaling and train-test split.

```python
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)
```

🎯 Insight: All features are scaled uniformly for a fair comparison, since KNN and SVM are distance-based. The split is 80% train (5,634) and 20% test (1,409) with stratify=y so the churn ratio is the same in both.

---

### 19. Handle class imbalance with SMOTE (training set only).

```python
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```

🎯 Insight: The training set has 4,139 stayed and 1,495 churned. After SMOTE both classes have 4,139. SMOTE is applied only after the split and only on training data, so the test set stays original.

---

## 🤖 Part 4: Model Building — KNN & Naive Bayes

<img width="1600" height="600" alt="download" src="https://github.com/user-attachments/assets/660989cb-de35-4a02-a82d-84d3a2f379ce" />



### 20. KNN baseline (k = 5).

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sm, y_train_sm)
```

🎯 Insight: Baseline KNN gives Accuracy 0.719, Precision 0.480, Recall 0.695, F1 0.568 and AUC 0.776.

---

### 21. Tune k and re-train with the best k.

```python
for k in [1, 3, 5, 7, 9, 11, 15]:
    ...
best_k = k_list[f1_list.index(max(f1_list))]
```

🎯 Insight: Best k is 15 (F1 kept improving as k increased on the validation set). With k = 15 recall improves from 0.70 to 0.80 and AUC to 0.802, but precision stays low at 0.47.

---

### 22. Gaussian Naive Bayes.

```python
nb = GaussianNB()
nb.fit(X_train_sm, y_train_sm)
```

🎯 Insight: Naive Bayes gives Accuracy 0.723, Precision 0.487, Recall 0.805, F1 0.607 and AUC 0.818 — the highest recall and AUC of all four models, and it trains in 0.015 seconds.

---

### 23. Class priors and the independence assumption.

```python
print('Class priors:', nb.class_prior_)
print('Actual churn rate:', y.mean())
```

🎯 Insight: The learned priors are [0.5, 0.5] but the actual churn rate is 26.5%, because the model was trained on SMOTE-balanced data. The independence assumption does not fully hold — tenure and TotalCharges are 0.83 correlated and MonthlyCharges depends on internet service and add-ons — so its probabilities are not very reliable even though recall is good.

---

## 🧮 Part 5: Model Building — SVM & Decision Tree

<img width="1600" height="600" alt="download" src="https://github.com/user-attachments/assets/b6b3f219-72f5-46a2-8761-3f02debbbdca" />



### 24. SVM (RBF kernel) baseline.

```python
svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
svm.fit(X_train_sm, y_train_sm)
```

🎯 Insight: Baseline SVM gives Accuracy 0.737, Precision 0.503, Recall 0.765, F1 0.607 and AUC 0.814.

---

### 25. Tune C with 3-fold cross-validation.

```python
for c in [0.1, 1, 10, 100]:
    cross_val_score(SVC(C=c, ...), X_train_sm, y_train_sm, cv=3, scoring='f1')
```

🎯 Insight: Cross-validated F1 rises with C: 0.770 (C = 0.1), 0.792 (1), 0.820 (10) and 0.828 (100), so C = 100 is chosen. However the final test metrics get worse — Recall drops to 0.674, F1 to 0.573 and AUC to 0.783 — a sign that high C overfits the SMOTE-augmented training data.

---

### 26. Interpretability vs performance of SVM.

🎯 Insight: SVM is a black box, so we cannot easily explain to the business why a customer is flagged. A simple model managers can understand and trust is often preferred over a slightly better black-box model.

---

### 27. Decision Tree (max_depth = 5, class_weight = 'balanced').

```python
dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
dt.fit(X_train_sm, y_train_sm)
```

🎯 Insight: Accuracy 0.744, Precision 0.512, Recall 0.773, F1 0.616 and AUC 0.827. The root split feature is `Contract_Month-to-month`.

---

### 28. Tune max_depth and plot feature importances.

```python
for d in [3, 4, 5, 6, 7, 8, None]:
    ...
```

🎯 Insight: Cross-validated F1 is best at depth 7 (0.798). The tuned tree gives Accuracy 0.757, Precision 0.530, Recall 0.722, F1 0.612 and AUC 0.816. Top importances: Contract_Month-to-month (0.498), tenure_group (0.149), InternetService_Fiber optic (0.081), MonthlyCharges (0.076) and tenure (0.075).

---

### 29. class_weight='balanced' vs SMOTE.

```python
dt_smote = DecisionTreeClassifier(max_depth=best_depth, random_state=42).fit(X_train_sm, y_train_sm)
dt_cw    = DecisionTreeClassifier(max_depth=best_depth, class_weight='balanced', random_state=42).fit(X_train, y_train)
```

🎯 Insight: Using the Decision Tree as the best model so far, class_weight='balanced' gives better churn recall (0.765) than SMOTE (0.722).

---

## 📊 Part 6: Model Evaluation & Comparison

<img width="1600" height="600" alt="download" src="https://github.com/user-attachments/assets/768836dd-a060-4932-879a-e0cd91395660" />


### 30. Consolidated comparison table (sorted by Recall).

```python
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall',
                                            'F1-Score', 'AUC-ROC', 'Training Time (s)'])
results_df = results_df.sort_values('Recall', ascending=False)
```

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Training Time (s) |
|-------|----------|-----------|--------|----------|---------|-------------------|
| Naive Bayes | 0.723 | 0.487 | 0.805 | 0.607 | 0.818 | 0.015 |
| KNN | 0.706 | 0.469 | 0.802 | 0.592 | 0.802 | 0.007 |
| Decision Tree | 0.757 | 0.530 | 0.722 | 0.612 | 0.816 | 0.098 |
| SVM | 0.734 | 0.499 | 0.674 | 0.573 | 0.783 | 28.567 |

🎯 Insight: Naive Bayes and KNN catch the most churners (recall ~0.80) but with precision below 0.49. The Decision Tree has the best accuracy, precision and F1. SVM is the slowest by far and the weakest on recall and AUC.

---

### 31. Recommendation for the retention team.

🎯 Insight: Deploy the **Decision Tree**. More than half of the customers flagged by Naive Bayes and KNN are wrong, which wastes the ~500 calls a day. The tree has the best precision (0.53), accuracy (0.757) and F1 (0.61), AUC almost equal to the best, still catches 72% of churners, is easy to explain to the business and predicts instantly. If more recall is needed later, class_weight='balanced' (recall 0.76) or a lower probability threshold can be used.

---

## 🔍 Part 7: Error Analysis & Interpretation

<img width="1600" height="600" alt="download" src="https://github.com/user-attachments/assets/6e5687d9-b2d5-4286-bfe1-3abcd337b670" />



### 32. False Negative profile of the best model.

```python
fn = X_test_raw[(y_test.values == 1) & (pred == 0)]
churners = X_test_raw[y_test.values == 1]
```

| Group | Count | Avg tenure (months) | Avg MonthlyCharges | Month-to-month |
|-------|-------|---------------------|--------------------|----------------|
| Missed churners (FN) | 104 | 29.8 | 67.2 | 61.5% |
| All churners | 374 | 16.5 | 72.8 | 88.0% |

🎯 Insight: The model missed 104 of 374 churners. The missed customers look like loyal customers (longer tenure, longer contracts, lower bill). The model learned that churners are new, month-to-month and high-bill customers, so it fails on churners who do not fit this typical profile.

---

### 33. Top 5 split features and business insights.

🎯 Insight:
- `Contract_Month-to-month` (first split) — no long-term contract, so the customer can leave any time. Month-to-month churns 42.7% vs 2.8% for two-year (about 15x).
- `tenure_group` and `tenure` — new customers (first 12 months) are the riskiest group (47% churn).
- `InternetService_Fiber optic` — fiber customers churn 41.9% vs 19.0% for DSL.
- `MonthlyCharges` — customers who pay more churn more (median 79.65 vs 64.43).

---

## 🚀 Part 8: Pipeline & Deployment

<img width="1600" height="600" alt="download" src="https://github.com/user-attachments/assets/9ad53677-cab7-4438-8153-b9d57e8a9b19" />


### 34. Save the final pipeline.

```python
pipeline = Pipeline([('scaler', scaler), ('model', dt)])
joblib.dump(pipeline, 'churn_model.pkl')
```

🎯 Insight: The StandardScaler and the tuned Decision Tree are wrapped in one sklearn Pipeline and saved as `churn_model.pkl`, so raw customer features can be passed straight in for prediction.

---

### 35. Load the pickle and predict for 5 sample customers.

```python
loaded_model = joblib.load('churn_model.pkl')
prob  = loaded_model.predict_proba(sample)[:, 1]
label = loaded_model.predict(sample)
```

| Customer | Churn probability | Predicted label | Actual |
|----------|-------------------|-----------------|--------|
| 1 | 0.099 | 0 | 1 |
| 2 | 0.684 | 1 | 1 |
| 3 | 0.892 | 1 | 1 |
| 4 | 0.000 | 0 | 0 |
| 5 | 0.892 | 1 | 0 |

🎯 Insight: The loaded pipeline predicts correctly for 3 of 5 sample customers, with one missed churner (customer 1) and one false alarm (customer 5), which matches the model's measured precision and recall.

---

## 📂 Project Workflow

1. Problem Framing → Churn, CAC vs CLV, confusion matrix costs, class imbalance, the four algorithms, precision vs recall
2. Dataset Understanding → Load, fix TotalCharges, class balance, univariate and bivariate EDA
3. Preprocessing → Drop ID, median fill, feature engineering, encoding, uniform scaling
4. Train/Test Split → 80/20 stratified split (5,634 / 1,409 customers)
5. Class Imbalance → SMOTE on the training set only
6. KNN & Naive Bayes → Baseline, tuning of k, priors, independence check
7. SVM & Decision Tree → Tuning of C and max_depth, feature importance, class_weight vs SMOTE
8. Model Evaluation → Accuracy, Precision, Recall, F1, ROC-AUC and training time comparison
9. Error Analysis → Profile of missed churners and business insights
10. Pipeline → Save `churn_model.pkl` and test predictions on 5 customers

---

## ⚙️ Installation & Setup

```bash
# clone the repository
git clone https://github.com/yourusername/telco-churn-supervised-learning.git
cd telco-churn-supervised-learning

# create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib jupyter

# launch the notebook
jupyter notebook CustomerChurn_SupervisedLearning.ipynb
```

📥 Dataset: [IBM Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 🙏 Thank You


Thank you for taking the time to explore this project!
Your feedback, suggestions, and contributions are always welcome.

⭐ If you found this project helpful, don't forget to star the repository and share it with others.
