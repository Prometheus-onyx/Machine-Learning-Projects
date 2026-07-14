import pandas as pd
import joblib as jb
import streamlit as slt


with open("Serach_model.pkl", "rb") as model:
    serach_model = jb.load(model)


def serach_prediction(var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8):

    input_data = pd.DataFrame({
        "longitude": [var_1],
        "latitude": [var_2],
        "housing_median_age": [var_3],
        "total_rooms": [var_4],
        "total_bedrooms": [var_5],
        "population": [var_6],
        "households": [var_7],
        "median_income": [var_8]
    })

    prediction = serach_model.predict(input_data)

    return prediction

def run():

    slt.title("Decision Tree Model")

    var_1 = slt.number_input("Longitude")
    var_2 = slt.number_input("Latitude")
    var_3 = slt.number_input("Housing Median Age")
    var_4 = slt.number_input("Total Rooms")
    var_5 = slt.number_input("Total Bedrooms")
    var_6 = slt.number_input("Population")
    var_7 = slt.number_input("Households")
    var_8 = slt.number_input("Median Income")

    if slt.button("Predict"):

        prediction = serach_prediction(
            var_1,var_2,var_3,var_4,var_5,
            var_6,var_7,var_8
        )

        slt.success(f"Hello! Steve....\n\nYour Estimated House Value is: ${prediction[0]:,.2f}")


if __name__ == '__main__':
    run()

#Runing streamlit server:: streamlit run PyApp.py