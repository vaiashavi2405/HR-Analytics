# HR Analytics & Employee Attrition Prediction 📊

## 📌 Project Overview
HR Analytics & Employee Attrition Prediction is a Machine Learning project that analyzes employee data and predicts whether an employee is likely to leave an organization.

The project helps HR departments understand employee attrition patterns, explore workforce insights, and support data-driven decision-making.

## 🎯 Objectives
- Analyze employee demographics and job-related information.
- Identify factors associated with employee attrition.
- Build and compare Machine Learning classification models.
- Predict employee attrition using a trained model.
- Display interactive HR analytics through a Streamlit dashboard.

## 📂 Dataset
**Dataset:** IBM HR Analytics Employee Attrition & Performance

**Source:** Kaggle  
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

The dataset contains employee information such as age, department, job role, monthly income, job satisfaction, overtime, and attrition status.

**Target Variable:** `Attrition`

- `Yes` — Employee left the organization.
- `No` — Employee stayed with the organization.

## 🛠️ Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

## 🤖 Machine Learning Models
1. Logistic Regression
2. Random Forest Classifier

The models are evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

## ✨ Key Features
- Interactive HR analytics dashboard
- Employee attrition prediction
- Exploratory Data Analysis (EDA)
- Model performance comparison
- Employee and department-level insights
- Data visualizations
- Feature importance analysis
- Trained model saving and reuse

## 📁 Project Structure

```text
HR_Analytics_ML/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── models/
│   └── hr_attrition_pipeline.pkl
├── reports/
├── train_model.py
├── app.py
├── HR_Analytics.ipynb
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add the Dataset

Download the dataset from Kaggle, extract the ZIP file, and place the CSV file inside the `data` folder.

### 3. Train the Models

```bash
python train_model.py
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## 🔮 Future Improvements
- Add explainable AI using SHAP.
- Introduce employee attrition risk categories.
- Add department-wise and role-wise analytics.
- Deploy the dashboard online.
- Improve performance through hyperparameter tuning.

## 👩‍💻 Author

**Vaishnavi Dawkhar**

Machine Learning & Data Analytics Enthusiast

## 📜 Disclaimer
This project is intended for educational and analytical purposes. Predictions should support, not replace, fair and human-reviewed HR decisions.
