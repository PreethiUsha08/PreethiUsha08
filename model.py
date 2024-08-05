"""import streamlit as st
# Save the model
import pickle
with open("logistic_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Create a Streamlit app
st.title("Titanic Survival Predictor")
st.write("Enter the passenger's details to predict their survival chances:")

# Create input fields for the user
Pclass = st.selectbox("Pclass", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.number_input("Age", min_value=0, max_value=100)
SibSp = st.number_input("SibSp", min_value=0, max_value=10)
Parch = st.number_input("Parch", min_value=0, max_value=10)
Fare = st.number_input("Fare", min_value=0, max_value=1000)
Embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Create a button to submit the input
submit_button = st.button("Predict Survival Chances")

# Create a function to preprocess the input data
def preprocess_input(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    input_data = pd.DataFrame({
        "Pclass": [Pclass],
        "Sex": [Sex],
        "Age": [Age],
        "SibSp": [SibSp],
        "Parch": [Parch],
        "Fare": [Fare],
        "Embarked": [Embarked]
    })
    min_max_scaler = MinMaxScaler()
    input_data[['Age', 'Fare']] = min_max_scaler.fit_transform(input_data[['Age', 'Fare']])
    return input_data

# Create a function to make predictions
def make_prediction(input_data):
    prediction = model.predict(input_data)
    return prediction

# Create a function to display the results
def display_results(prediction):
    if prediction == 1:
        st.write("The passenger is likely to survive!")
    else:
        st.write("The passenger is unlikely to survive.")

# Run the app
if submit_button:
    input_data = preprocess_input(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
    prediction = make_prediction(input_data)
    display_results(prediction)"""