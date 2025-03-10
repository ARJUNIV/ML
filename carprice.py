import numpy as np
import joblib
import pandas as pd
import streamlit as st

df=pd.read_csv(r"C:\Users\User\Downloads\archive (57)\Clean Data_pakwheels.csv")




# Function to load the saved model and label encoders
def load_model():
    model = joblib.load(open(r"C:\Users\User\Desktop\machine learning\supervised\regression\car_price_prediction", "rb"))
    label_encoders = {
        'Company Name': joblib.load(open(r"C:\Users\User\Desktop\machine learning\supervised\regression\L1", "rb")),
        'Model Name': joblib.load(open(r"C:\Users\User\Desktop\machine learning\supervised\regression\L2", "rb")),
        'Engine Type': joblib.load(open(r"C:\Users\User\Desktop\machine learning\supervised\regression\L3", "rb")),
        'Color': joblib.load(open(r"C:\Users\User\Desktop\machine learning\supervised\regression\L4", "rb")),
        'Assembly': joblib.load(open(r"C:\Users\User\Desktop\machine learning\supervised\regression\L5", "rb")),
        'Body Type': joblib.load(open(r"C:\Users\User\Desktop\machine learning\supervised\regression\L6", "rb")),
        'Transmission Type': joblib.load(open(r"C:\Users\User\Desktop\machine learning\supervised\regression\L7", "rb")),
    }
    return model, label_encoders

# Function to preprocess the input data
def preprocess_input(data, label_encoders):
    processed_data = {}
    for key in data.keys():
        if data[key] is None:
            raise ValueError(f"The input for {key} is missing.")
        
        # Print the type and value of each input before conversion
        print(f"Key: {key}, Type: {type(data[key])}, Value: {data[key]}")
        
        # Apply label encoding to categorical fields
        if key in label_encoders:
            processed_data[key] = label_encoders[key].transform([str(data[key])])[0]
        else:
            # Keep numerical fields as is
            processed_data[key] = data[key]
    
    # Debugging: Print processed data
    print("Processed Data:", processed_data)
    
    return np.array(list(processed_data.values())).reshape(1, -1)

# Load the model and label encoders
model, label_encoders = load_model()

# Streamlit interface
st.title("Car Price Prediction") 
st.write("Enter the details of the car to get the estimated price.")

input_data = {
    "Company Name": st.selectbox("Company Name", df['Company Name'].unique()),
    "Model Name": st.selectbox("Model Name", df['Model Name'].unique()),
    "Model Year": st.number_input("Model Year", min_value=int(df['Model Year'].min()), max_value=int(df['Model Year'].max())),
    "Mileage": st.number_input("Mileage", min_value=int(df['Mileage'].min()), max_value=int(df['Mileage'].max())),
    "Engine Type": st.selectbox("Engine Type", df['Engine Type'].unique()),
    "Engine Capacity": st.number_input("Engine Capacity", min_value=int(df['Engine Capacity'].min()), max_value=int(df['Engine Capacity'].max())),
    "Color": st.selectbox("Color", df['Color'].unique()),
    "Assembly": st.selectbox("Assembly", df['Assembly'].unique()),
    "Body Type": st.selectbox("Body Type", df['Body Type'].unique()),
    "Transmission Type": st.selectbox("Transmission Type", df['Transmission Type'].unique())
}

if st.button("Predict Price"):
    try:
        # Debugging: Print input data
        print("Input Data:", input_data)

        # Preprocess input data
        processed_data = preprocess_input(input_data, label_encoders) 

        # Debugging: Print reshaped data
        print("Reshaped Data:", processed_data)  

        # Predict the car price
        prediction = model.predict(processed_data)
        st.write(f"The estimated price of the car is: {prediction[0]:.2f}")
    except ValueError as e:
        st.write(f"ValueError: {e}")
    except TypeError as e:
        st.write(f"TypeError: {e}")
