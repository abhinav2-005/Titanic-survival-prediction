# 🚢 Titanic Survival Prediction

A machine learning project that predicts whether a passenger would survive the Titanic disaster based on passenger information such as class, gender, age, family details, fare, and port of embarkation.

## 📌 Features

* Predicts Titanic passenger survival using a trained **SVC model**
* Data preprocessing and feature scaling using **StandardScaler**
* Streamlit frontend for interactive predictions
* Uses Joblib to load the trained model and preprocessing files

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

## 📊 Input Features

The model uses the following features:

* Passenger Class (`pclass`)
* Sex (`sex`)
* Age (`age`)
* Siblings/Spouses (`sibsp`)
* Parents/Children (`parch`)
* Fare (`fare`)
* Travelling Alone (`alone`)
* Embarked: Cherbourg (`embarked_C`)
* Embarked: Queenstown (`embarked_Q`)
* Embarked: Southampton (`embarked_S`)

## 📁 Project Structure

```text
Titanic-survival-prediction/
│
├── app.py
├── model.pkl
├── standerd_scaler.pkl
├── columns.pkl
├── TItanic_survival_prediction.ipynb
├── tree.txt
└── README.md
```

## ⚙️ Installation

Clone the repository and install the required libraries:

```bash
pip install pandas scikit-learn streamlit joblib
```

## ▶️ Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your browser, where you can enter passenger details and get a survival prediction.

## 🧠 Model

The prediction model is an **SVC (Support Vector Classifier)** trained on the preprocessed Titanic dataset.

The trained model and scaler are stored as:

```text
model.pkl
standerd_scaler.pkl
```

## 🎯 Prediction

The application returns one of two predictions:

* 🟢 Passenger is predicted to survive
* 🔴 Passenger is predicted not to survive

If probability prediction is available from the trained SVC model, the application also displays the estimated survival probability.

## 📚 Dataset

The project is based on the **Titanic passenger dataset**, containing information about passengers and whether they survived the disaster.
