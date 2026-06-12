# 🚨 Fail Safe

# Final Project Report

## 1. Core problem

In educational institutions (like IIT Roorkee), student failure often goes undetected until end-of-semester results, leaving no room for meaningful intervention. Faculty lack a proactive, data-driven tool to identify at-risk students early and understand the root causes behind their struggles.

---

## 2. About dataset

Student Performance Data Set by UCI is used for this analysis.The dataset contains information collected from Portuguese secondary schools focusing on student achievement in secondary education. It is designed to help identify factors that influence academic success and to build early warning systems for students at risk of failing.

### Key Attributes

The dataset contains over 30 distinct attributes describing each student's background and environment.

These features generally fall into three analytical categories:

* **Demographic & Family:** Parents' education levels, family size, parents' cohabitation status, and mother's/father's occupations.
* **Academic History & Habits:** Weekly study time, number of past class failures, school absences, and travel time to school.
* **Social & Lifestyle:** Extracurricular activities, weekend alcohol consumption, free time, and family relationship quality.

### Target Variables

The primary goal of using this dataset is to predict the final academic grade (denoted as G3). The dataset provides grades across three evaluation periods, measured on a 0 to 20 grading scale:

* **G1:** First-period grade.
* **G2:** Second-period grade.
* **G3:** Final grade.

---

## 3. My Strategy

<div align="center">

### 👉 **Video Explaining what i have done = video link**👈

</div>

In this project, my main goal is to predict the final grades (G3) of high school students in two subjects: Maths and Portuguese. Because this is a real world problem, just predicting the final grade one day before exam is not very useful. Predicting early is very useful so teachers can give extra attention to weak students. So I decided to make it like an "Early Warning System" to find out which student need help early on in the academic year.

Because the size of data is very small (total around 1044 rows after combining both subjects), I think DL (Deep Learning) based methods will highly overfit. Deep learning models easily memorize small data and fail on unseen test data. So I completely avoided them. Instead of that, I choose gradient boosting (XGBoost). XGBoost works very good on small tabular data and it is very robust.

I planned a 3-model strategy to predict G3 at different times in the year:

* **Model 1 (Pre First Period exam):** Predicting G3 before the first exam. Meaning I will not be using G1 and G2 scores at all. This is the hardest model because it only relies on background data, but it give the earliest warning.
* **Model 2 (Pre Second Period exam):** Predicting G3 before the second exam. Here I will use G1 score but not G2.
* **Model 3 (Pre Final exam):** Predicting G3 using both G1 and G2 scores. This will give best and most accurate prediction.

---

## 4. Exploratory Data Analysis (EDA)

First I loaded both datasets `student-mat.csv` and `student-por.csv` using pandas. Before merging them together i created a new column called `subject` to tell if that row is for maths or portuguese.

I checked how many students are present in both datasets. I matched their 13 unique demographic features (like school, sex, age, address, Medu, Fedu, etc.) and find out that 382 students are common in both datasets.

To see which categorical feature is affecting G3 more, I used Mutual Information (MI) regression and ANOVA statistical tests. Normal correlation matrix does not work well for categories, so MI was a better choice. I label-encoded the categories just for the MI test. I found that features like `higher` (student wants to take higher education), `Mjob` (mother's job), and `school` are very important to decide the grades. Also, the `subject` column was very important because Maths and Portuguese have different scoring distributions.

### Correlation plot

<img width="1351" height="450" alt="image" src="https://github.com/user-attachments/assets/fa15ee9f-25b9-43bb-a7d6-575f7b1cdbdd" />


---

## 5. Model Training

A critical challenge emerged during the data preparation phase: **Data Leakage**. Because 382 students existed in both the Math and Portuguese datasets, a standard random `train_test_split` would likely place "Student A's" Math records in the training set and "Student A's" Portuguese records in the testing set. The model would effectively "cheat" by memorizing Student A's unique demographic profile.

To ensure a robust evaluation, I implemented **GroupKFold Cross-Validation (with 5 splits)**. By concatenating the 13 demographic features into a unique `student_id` string, I instructed GroupKFold to keep all records belonging to a specific student entirely within the training fold or entirely within the testing fold.

Furthermore, I utilized a modern feature of the XGBoost library: **Native Categorical Support** (`enable_categorical=True` and `tree_method="hist"`). Instead of manually applying One-Hot Encoding (OHE)—which would have exploded the dimensionality of this small dataset and exacerbated overfitting—I passed Pandas category dtypes directly into the model. XGBoost optimally partitioned these categories internally, saving memory and improving accuracy.

---

## 6. Hyperparameter Tuning

I did hyperparameter tuning for the models using the Optuna library. For the tuning parameters:

* I used `n_estimators=1000` with `early_stopping_rounds=50` so the trees stop building automatically when the validation error stops improving.
* I tuned `max_depth` (between 3 and 10), `learning_rate` (log scale), and regularizations like `reg_lambda` to prevent overfitting.

During tuning, I kept XGBoost training objective as `reg:squarederror` (MSE) because it calculates mathematical gradients much better for the trees. But for evaluating the Optuna trials and printing results, I returned Mean Absolute Error (MAE) because I wanted to minimize the actual grade point difference. MAE is much easier to understand in real life (e.g. "the model is off by 1.2 marks on average").

---

## 7. The Three Models and their Performance

After training and tuning using GroupKFold, I compared the models. As expected, as we add more recent grade data, the accuracy improves significantly.

Here is the comparison table of all three models on different metrics:

| Model Version           | Features Used                     | MAE (Mean Absolute Error) | RMSE (Root Mean Sq. Error) | R-Squared (R2) | Practical Use                                                                                   |
| ----------------------- | --------------------------------- | ------------------------- | -------------------------- | -------------- | ----------------------------------------------------------------------------------------------- |
| **Model 1 (Pre-G1)**    | Demographics, Social, Family only | 2.85 points               | 3.50 points                | 0.18           | Early warning at start of year. Hard to predict but useful for finding at-risk students.        |
| **Model 2 (Pre-G2)**    | Demographics + G1 Score           | 1.25 points               | 1.85 points                | 0.76           | Mid-term check. Model learns the student's actual current capacity.                             |
| **Model 3 (Pre-Final)** | Demographics + G1 + G2 Score      | 0.85 points               | 1.30 points                | 0.91           | Final forecast. Since G2 is highly correlated with G3, predictions are very tight and accurate. |

> **Note:** The grades are out of 20. An MAE of 0.85 means the model is on average less than 1 mark away from the actual final score.

---

## 8. Web UI and FastAPI Implementation

To translate these ML models into a tangible tool for educators and concerned authority, I developed a web application architecture using FastAPI for the backend and Streamlit for the frontend.

### Home Page

<img width="1913" height="963" alt="image" src="https://github.com/user-attachments/assets/65f17dc0-e5ed-4f13-b4bd-a97de4c21581" />


### Single Prediction Page

<img width="1901" height="957" alt="image" src="https://github.com/user-attachments/assets/93816a1b-0efd-4662-b7a1-da16c1355292" />


### Batch Prediction Page

<img width="1912" height="583" alt="image" src="https://github.com/user-attachments/assets/177e5f8f-f141-4c1f-afff-7eb005bb4f58" />


---

### Serialization

Before building the API logic, all three tuned models were exported using XGBoost's native `.save_model("model.json")` method. The JSON format was explicitly chosen over Python's pickle library, as JSON is language-agnostic and guarantees forward compatibility, ensuring the models won't break upon future library updates.

### Backend (FastAPI Engine)

The FastAPI application was designed for strict separation of concerns. It loads all three XGBoost models into memory upon startup via asynchronous lifespan events. When the API receives a JSON payload containing student data, it dynamically selects the appropriate model (Pre-G1, Pre-G2, or Pre-Final) based on the user's selected stage.

Crucially, the backend hardcodes the expected variable order, ensuring the Pandas DataFrame aligns perfectly with the model's training structure before inference.

Additionally, the backend integrates the `shap` (SHapley Additive exPlanations) library to calculate feature importance for individual predictions. Executing SHAP in the backend prevents the UI from freezing during intensive calculations.

### Frontend (Streamlit UI)

The Streamlit interface was designed for simplicity. It have 3 pages:-

#### 1. Home page

contains description of project

#### 2. Single Prediction page

The user's first interaction is selecting the "Prediction Stage" via a radio button. This selection dynamically alters the form layout, showing or hiding the G1 and G2 input fields accordingly.

Upon clicking "Predict," the frontend transmits the form dictionary to the FastAPI backend. It then parses the response to display:

* The predicted G3 score (color-coded as High, Average, or At Risk).
* An interpretable Matplotlib bar chart generated from the returned SHAP values, explaining exactly why the model assigned that specific grade (e.g., "High absences decreased the score, but high study time increased it").

#### 3. Batch Prediction

similar to single prediction page but we can we the analysis for many students(whole branch) by uploading csv file containing the required data.

---

🎯 This end-to-end implementation successfully bridges the gap between raw statistical modeling and an actionable, educational tool.
