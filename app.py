import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
with open("./code/model/car_selling_price.model", "rb") as f:
    model = pickle.load(f)

# Initialize Dash app
app = dash.Dash(__name__)


# App layout
app.layout = html.Div([
    html.H1("Car Selling Price Prediction", style={"textAlign": "center"}),

    html.Div([
        html.Div([
            html.Label("Year of Manufacture:"),
            dcc.Input(id="input-year", type="number", placeholder="Enter year", required=True, style={"width": "100%", "marginTop":"10px"}),
        ], style={"marginBottom": "30px"}),

        html.Div([
            html.Label("Mileage (in KM/L):"),
            dcc.Input(id="input-mileage", type="number", placeholder="Enter mileage", required=True, style={"width": "100%",  "marginTop":"10px"}),
        ], style={"marginBottom": "30px"}),

        html.Div([
            html.Label("Max Power (in BHP):"),
            dcc.Input(id="input-maxpower", type="number", placeholder="Enter max power", required=True, style={"width": "100%",  "marginTop":"10px"}),
        ], style={"marginBottom": "30px"}),

        html.Button("Calculate", id="predict-button", n_clicks=0, style={"marginTop": "20px"}),
    ], style={
        "width": "50%",
        "margin": "auto",
        "padding": "20px",
        "border": "1px solid black",
        "borderRadius": "10px",
        "textAlign": "left"  # Aligns all content to the left within the div
    }),

    html.Div(id="output-prediction", style={"textAlign": "center", "marginTop": "20px", "fontSize": "20px"})
])


# Callback to handle prediction
@app.callback(
    Output("output-prediction", "children"),
    Input("predict-button", "n_clicks"),
    State("input-year", "value"),
    State("input-mileage", "value"),
    State("input-maxpower", "value")
)
def predict_price(n_clicks, year, mileage, max_power):
    if n_clicks > 0:
        # Ensure all inputs are filled
        if year is None or mileage is None or max_power is None:
            return "Please fill in all fields to calculate the price."

        # Prepare the input features as a DataFrame
        input_features = pd.DataFrame({
            "year": [year],
            "mileage": [mileage],
            "max_power": [max_power]
        })

        # Predict the car's price
        predicted_price = model.predict(input_features)[0]
        predicted_price = np.exp(predicted_price)

        # Format and return the result
        return f"The predicted selling price of the car is: ${predicted_price:,.0f}"
    return ""

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
