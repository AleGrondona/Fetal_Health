# Fetal Health

### A Machine Learning model to classify the outcome of Cardiotocogram test to detect potential health issues.

* Full code: [fetal_health.ipynb](https://github.com/AleGrondona/Fetal_Health/blob/main/fetal_health.ipynb)
* Presentation: [Fetal_Health_presentation.pdf](https://github.com/AleGrondona/Fetal_Health/blob/main/Fetal_Health_presentation.pdf)
* Dataset: [fetal_health.csv](https://github.com/AleGrondona/Fetal_Health/blob/main/fetal_health.csv)
* Dataset source: [link](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/data)





## Overview  

The dataset consists **2126 records** of features extracted from cardiotocographic (**CTG**) examinations, providing important data on the condition of fetuses and mothers. Each of these records has been classified by expert obstetricians based on the health of the fetuses.

**21 numerical features**: provide information about heart rate, its short-term and long-term variability, fetal movements, and uterine contractions. The last value is the **health status** of the fetus, classified into three ordered categories: *Normal*, *Suspect* and *Pathological*.

The goal is building a machine learning model able to predict the health status of a fetus by assigning it to one of three categories based on data provided by the CTG test. This is a **classification problem** that would allow preventive action to reduce infant mortality based on a quick and non- invasive test.

## EDA

- There are **no missing values**.
- **Target distribution**
  
  <img width="350" height="286" alt="image" src="https://github.com/user-attachments/assets/872ada08-d198-43f2-a272-0fa5a8f3ac25" />

  The classes are quite **unbalanced**.  
  Approximately 78% of the dataset is populated by the *'Normal'* class.  
  The remaining part is populated for nearly two- thirds by the *'Suspect'* class, which makes up 14% of the dataset.  
  The *'Pathological'* class is a minority, its population is just over 1/10 of the *'Normal'* class, around 8% of the total.  

- **Feature correlations**

  <img width="227" height="131" alt="image" src="https://github.com/user-attachments/assets/d5852c0f-2569-479b-bcd3-9000b87fbc80" />


  There are four features with a correlation higher than the threshold (0.3): *'accel_rate'*, number of heartbeat accelerations per second; *'prol_dec_rate'*, number of prolongued heartbeat decelerations per second; *'short_var_perc'*, percentage of time with abnormal short term variability; *'long_var_perc'*, percentage of time with abnormal long term variability.

- **Feature distributions** and **outliers**

  <img width="312" height="494" alt="image" src="https://github.com/user-attachments/assets/beb5354e-85ca-4f17-af09-8bd4fb3f5889" />

  Boxplots show **outliers** presence; however, I want to keep all the data available for several reasons. The most important reason is that the goal of this alaysis is to identify potential pathological cases for preventive purposes. Since it is reasonable to assume, as a precaution, that values in the distribution tails could be related to fetal health anomalies, I want to keep the information provided by these data.

  The charts reveal the presence of some **outliers**. These data **were retained** due to the nature of this analysis.  
Our main purpose is to preemptively identify pregnancy cases that could compromise the health of the fetuses and mothers.  
It is reasonable to assume, as a precaution, that anomalies in one or more of the features detected by the CTG exam may be indicative of such problems.  
Consequently, the information provided by the records presenting these anomalies is potentially **valuable** for building a model aimed at correctly classifying as many at-risk cases as possible.

- **Evaluation metrics**

  Among the evaluation metrics used to compare different models, **recall** was chosen as the primary reference.  
This metric allows for minimizing the number of cases belonging to a specific class but not classified as such: false negatives. As the number of false negatives decreases, recall increases, making it a suitable metric for our purposes.  
Due to class imbalance and underrepresentation of the two key classes, recall and other metrics were computed using an unweighted average.

## Preprocessing

- **Scaling**: used *StandardScaler* for normalization.

- **Encoding**: used *LabelEncoder* to rename target classes (1,2,3) to (0,1,2). I need this operation in order to use XGBoost model.

- **Train/Test Split**: used an 80/20 partition for model evaluation.

## Model selection

Four classification models were tested and compared:

- **KNN**
  
  <img width="284" height="84" alt="Screenshot 2025-09-10 alle 19 18 05" src="https://github.com/user-attachments/assets/f1e6dc2c-3353-496e-84a3-00e2738834aa" />

- **Decision Tree**
  
  <img width="284" height="84" alt="Screenshot 2025-09-10 alle 19 18 05" src="https://github.com/user-attachments/assets/c8a42ab5-0047-4a75-bae0-bccebd073ac6" />

- **Random Forest**
  
  <img width="284" height="90" alt="Screenshot 2025-09-10 alle 19 18 26" src="https://github.com/user-attachments/assets/683bb06f-985f-4bc0-b738-c16480d58286" />

- **XGBoost**
  
  <img width="287" height="88" alt="Screenshot 2025-09-10 alle 19 18 37" src="https://github.com/user-attachments/assets/68cf82ad-9d4b-41a7-ad0c-915435838dc8" />


The best-performing model across all metrics is **XGBoost**.  

**Hyperparameters** were optimized using a **grid search**, selecting the best configuration based on evaluation metrics.  
The updated recall value is: **0.9130471352339388**

## Results

Using the XGBoost classifier, the following results were obtained on the test set.  

<img width="407" height="175" alt="Screenshot 2025-09-15 alle 13 14 12" src="https://github.com/user-attachments/assets/7791b8eb-ca0f-475f-a539-325eeb61e919" />  <br>

Recall is very good for the *'Normal'* and *'Pathological'* classes, but it is significantly lower for the intermediate *'Suspect'* class, which lowers the overall average.  

<br>

<img width="371" height="265" alt="image" src="https://github.com/user-attachments/assets/cb9ec3e5-b08c-49a2-9569-eebf841ab672" />  <br>

From the **confusion matrix**, it can be seen that a significant percentage of *'Suspect'* elements were classified as *'Normal’*, while only two of them were classified as *'Pathological'*.

## Feature Importance  

<img width="500" height="532" alt="image" src="https://github.com/user-attachments/assets/d1ac0315-b5f2-47a4-a0d4-e6e4f3c8f3c2" /> <br>

The four features initially selected based on correlation are, as expected, among the most important in the decision made with XGBoost. However, the two that are clearly the most decisive are 'short_var_mean', which represents the mean value of short term variability, and 'hist_mean', which represents the average value of the heart rate distribution histogram.

## Conclusions

- The XGBoost classifier achieved the best performance across all tested models, with an updated recall of 0.913.

- Recall is high for the *'Normal'* and *'Pathological'* classes, enabling effective identification of the most critical cases.

- The *'Suspect'* class is less accurately classified, often mispredicted as *'Normal'*, which highlights a limitation due to class imbalance.

- Feature importance confirms that short-term variability ('short_var_mean') and heart rate histogram average ('hist_mean') are the most decisive predictors.

- Overall, the model demonstrates strong potential for supporting early detection of fetal health issues using non-invasive CTG data.

