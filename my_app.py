import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Streamlit app
st.title("Pass/Fail Prediction App")

# Sidebar for user input
st.sidebar.header("User Input")

# Collect user input for marks
subject_names = ['Mathematics', 'Programming', 'Database', 'Networks']
user_input_marks = []
for subject in subject_names:
    user_input_marks.append(st.sidebar.slider(f"{subject} Marks", 0, 100, 50))

# Display user input
st.write("## User Input:")
user_input_df = pd.DataFrame([user_input_marks], columns=subject_names)
st.table(user_input_df)

# Load a larger sample dataset or use your own
# In this example, we'll use a simple generated dataset
data = {
    'Mathematics': [75, 40, 60, 80, 95, 65, 75, 50, 90, 85],
    'Programming': [85, 30, 65, 75, 78, 60, 40, 55, 75, 80],
    'Database': [80, 20, 50, 70, 85, 45, 30, 60, 70, 75],
    'Networks': [90, 38, 55, 60, 95, 80, 65, 45, 70, 88],
    'Pass/Fail': ['Pass', 'Fail', 'Fail', 'Pass', 'Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Pass']
}

df = pd.DataFrame(data)

# Convert Pass/Fail to binary labels (1 for Pass, 0 for Fail)
df['Pass/Fail'] = df['Pass/Fail'].map({'Pass': 1, 'Fail': 0})

# Check if the dataset size is sufficient for training
if len(df) >= 10:
    # Train a simple Random Forest classification model
    X = df[subject_names]
    y = df['Pass/Fail']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make prediction using the user input
    prediction = model.predict([user_input_marks])[0]
    prediction_label = 'Pass' if prediction == 1 else 'Fail'

    # Display predicted result
    st.write("## Prediction:")
    st.write("Predicted Pass/Fail:", prediction_label)

    # Display model evaluation (accuracy on the test set)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("## Model Evaluation:")
    st.write("Accuracy on Test Set:", accuracy)

else:
    st.write("The dataset size is not sufficient for training. Please provide more data.")

# Project credits
st.sidebar.markdown("---")
st.sidebar.markdown("### Project Credits")
st.sidebar.text("This project was done by Utsav Kumar.")
