import pandas as pd

df = pd.read_csv("data/college_event_ml_dataset.csv")

print(df.head())
print(df.shape)
print(df.info())

#data cleaning
print(df.isnull().sum())
#for numerical data
df["Distance_KM"] = df["Distance_KM"].fillna(
    df["Distance_KM"].median()
)
#for categorical data
df["Technical_Interest"] = df["Technical_Interest"].fillna(
    df["Technical_Interest"].mode()[0]
)
#verify
print(df.isnull().sum())

# -----------------------------
# Exploratory Data Analysis
# -----------------------------

import matplotlib.pyplot as plt
import seaborn as sns

print("\nStatistical Summary:")
print(df.describe())

print("\nAttendance Count:")
print(df["Attended"].value_counts())

attendance_percentage = df["Attended"].mean() * 100

print("\nOverall Attendance Percentage:")
print(round(attendance_percentage, 2), "%")

#First graph: Event Type
plt.figure(figsize=(10, 5))

sns.countplot(data=df, x="Event_Type")

plt.title("Distribution of Event Types")
plt.xlabel("Event Type")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#Department-wise attendance
plt.figure(figsize=(10, 5))

sns.barplot(
    data=df,
    x="Department",
    y="Attended"
)

plt.title("Attendance by Department")
plt.xlabel("Department")
plt.ylabel("Average Attendance")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Previous attendance vs current attendance
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Attended",
    y="Previous_Attendance_Rate"
)

plt.title("Previous Attendance Rate vs Event Attendance")
plt.xlabel("Current Event Attendance")
plt.ylabel("Previous Attendance Rate")

plt.show()

#Distance vs attendance
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Attended",
    y="Distance_KM"
)

plt.title("Distance vs Event Attendance")
plt.xlabel("Current Event Attendance")
plt.ylabel("Distance from Venue (KM)")

plt.show()

#Feedback analysis
plt.figure(figsize=(10, 5))

sns.barplot(
    data=df,
    x="Event_Type",
    y="Average_Feedback"
)

plt.title("Average Feedback by Event Type")
plt.xlabel("Event Type")
plt.ylabel("Average Feedback")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -----------------------------
# Machine Learning
# -----------------------------

# Target variable
y = df["Attended"]

# Features
X = df[
    [
        "Previous_Events_Attended",
        "Previous_Attendance_Rate",
        "Technical_Interest",
        "Registered",
        "Notification_Received",
        "Distance_KM",
        "Registration_Channel",
        "Event_Type",
        "Department",
        "Year"
    ]
]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

categorical_features = [
    "Technical_Interest",
    "Registration_Channel",
    "Event_Type",
    "Department"
]

numeric_features = [
    "Previous_Events_Attended",
    "Previous_Attendance_Rate",
    "Registered",
    "Notification_Received",
    "Distance_KM",
    "Year"
]

#Convert categorical data into numbers
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

#Create the Logistic Regression model and training it
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

model.fit(X_train, y_train)

print("\nModel training completed successfully!")

#Make predictions
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:20])

#Calculate model accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

#Get a complete classification report
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

#Save the trained model
import joblib

joblib.dump(model, "models/attendance_model.pkl")

print("\nModel saved successfully!")