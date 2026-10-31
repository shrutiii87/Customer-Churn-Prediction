# Customer Churn Prediction — Summary Report

<img width="1600" height="560" alt="download" src="https://github.com/user-attachments/assets/0414ce86-6afd-409d-9f00-c9c897d94272" />

---

## Business Problem and Dataset
A telecom company similar to Jio or Airtel is losing paying subscribers every month and cannot predict who will leave. Acquiring a customer (CAC) costs far more than keeping one, and each churner also removes future revenue (CLV). The goal is to build a model that flags likely churners so the retention team can act early. The data is the IBM Telco Customer Churn dataset: 7,043 customers and 21 columns covering demographics, services, contract, billing and the Churn label. Only 26.5% of customers churned (1,869 vs 5,174), so the data is imbalanced at roughly 3 : 1.

## Preprocessing and Class Imbalance
`customerID` was dropped. `TotalCharges` loaded as text because 11 new customers (tenure = 0) had a blank value; these were converted to NaN and filled with the median. I engineered `tenure_group`, `num_services` and `AutoPay`, mapped binary columns to 1/0, one-hot encoded InternetService, Contract and PaymentMethod, and scaled all features with StandardScaler. Data was split 80/20 with stratification (5,634 train / 1,409 test). SMOTE was applied only to the training set (4,139 vs 1,495 became 4,139 vs 4,139), so the test set stayed untouched. 

---

## Recommended Model

<img width="1840" height="800" alt="image" src="https://github.com/user-attachments/assets/a7222cb7-f1d5-438f-a63e-1d62e4a41593" />

I recommend the **Decision Tree** (max_depth = 7). On the test set it has the best accuracy (0.757), precision (0.530) and F1 (0.612), with recall 0.722 and AUC 0.816. Naive Bayes (recall 0.805) and KNN (0.802) catch more churners, but their precision is below 0.49, so over half of flagged customers would be wrong and waste the retention team's roughly 500 calls a day. SVM was the weakest (recall 0.674, AUC 0.783) and took 28.6 seconds to train. The tree is also easy to explain and predicts instantly. If more recall is needed, class_weight='balanced' (recall 0.765) or a lower probability threshold can be used.

---

## Top 3 Churn Signals and Actions
1. **Month-to-month contract** — 42.7% churn against 2.8% for two-year contracts. *Action:* offer discounts or perks to move these customers to one- or two-year plans.
2. **Short tenure** — 47.4% churn in the first 12 months, falling to 9.5% after 49 months. *Action:* run onboarding calls and a welcome offer during the first year.
3. **Fiber optic and high monthly bill** — fiber customers churn 41.9% against 19.0% for DSL, and churners pay a median of 79.65 against 64.43. *Action:* review fiber price and service quality and give loyalty offers to high-bill customers.

The combined segment (month-to-month, charges above 70, tenure up to 12 months) churns at 69%, so it should be contacted first. Error analysis shows the model misses 104 of 374 churners, who tend to be longer-tenure, longer-contract customers, so these atypical churners need separate monitoring.

## Next Steps
- **More data:** add usage, complaint, call-centre and network-quality data, plus several months of history, to catch churners that do not fit the typical profile.
- **Better models:** tune and test gradient boosting or random forest before considering deep learning, since the data is small and tabular. Set the threshold using the cost of a missed churner versus a wasted call.
- **Real-time scoring API:** serve `churn_model.pkl` behind a simple API and refresh scores regularly. Monitor drift and retrain periodically.
