#pip install streamlit
#pip install pandas
#pip install scikit-learn

# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
from PIL import Image

df = pd.read_csv('diabetes.csv')

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# X AND Y DATA
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# FUNCTION
def user_report():
    pregnancies = st.sidebar.select_slider('Pregnant', options=list(pregnancy_labels.keys()), key='pregnancies')
    glucose = st.sidebar.slider('Glucose', 0, 200, 120, key='glucose')
    bloodpressure = st.sidebar.select_slider('Blood Pressure', options=list(bp_labels.keys()), key='bloodpressure')
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20, key='skinthickness')
    insulin = st.sidebar.slider('Insulin', 0, 846, 79, key='insulin')
    bmi = st.sidebar.select_slider('BMI', options=list(bmi_labels.keys()), key='bmi')
    diabetespedigreefunction = st.sidebar.select_slider('Diabetes Pedigree Function', options=list(diabetespedigree_labels.keys()), key='diabetespedigreefunction')
    age = st.sidebar.slider('Age', 21, 88, 33, key='age')

    user_report_data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bloodpressure': bloodpressure,
        'skinthickness': skinthickness,
        'insulin': insulin,
        'bmi': bmi,
        'diabetespedigreefunction': diabetespedigreefunction,
        'age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Dictionary to map labels to numerical values for select_slider
pregnancy_labels = {'0': 0, '1': 1}
bp_labels = {'0': 0, '100': 100, '122': 122}
bmi_labels = {'0': 0, '47': 47, 'High': 67}
diabetespedigree_labels = {'0.1': 0.1, '1.3': 1.3, '2.4': 2.4}

# Create a sidebar for parameter adjustment
st.sidebar.title('Parameter Adjustment')

#bmi_chart = Image.open("C:/Users/rachel/OneDrive/Desktop/diabities/diabeties/bmi-chart.jpg")  # Replace 'bmi_chart.png with the path to your BMI chart image
#st.subheader("BMI Chart")
#st.image(bmi_chart,width=1000)

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)


# MODEL
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

# OUTPUT
st.subheader('Your Report: ')
output = ''
if user_result[0] == 0:
    output = 'You are not Diabetic'
else:
    output = 'You are Diabetic'
st.title(output)

# Display the model accuracy
accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
st.subheader('Accuracy:')
st.write(f'{accuracy:.2f}%')

