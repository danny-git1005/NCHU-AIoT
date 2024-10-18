from flask import Flask, render_template, request
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import BytesIO
import base64

app = Flask(__name__)

# 1. 業務理解 (Business Understanding)
# Flask app to generate and visualize linear regression based on user input

# 2. 數據理解 (Data Understanding)
# Function to generate synthetic data and perform linear regression
# The function takes slope (a), noise, and number of points as input.
def generate_data_and_model(a, noise, num_points):
    np.random.seed(42)  # Ensures data generation is reproducible
    X = np.random.rand(num_points, 1) * 10  # Generate random X values between 0 and 10
    y = a * X + 5 + noise * np.random.randn(num_points, 1)  # Generate y = ax + b with added noise

    # 3. 數據準備 (Data Preparation)
    # Fit a simple linear regression model
    model = LinearRegression()  # Initialize linear regression model
    model.fit(X, y)  # Fit the model to the generated data
    y_pred = model.predict(X)  # Predict the output based on the model

    # Return the predicted values and model parameters (slope and intercept)
    return X, y, y_pred, model.coef_[0][0], model.intercept_[0]

# 5. 評估 (Evaluation)
# Function to plot the results (data points and regression line)
def plot_results(X, y, y_pred):
    plt.scatter(X, y, color="blue", label="Data points")  # Plot data points
    plt.plot(X, y_pred, color="red", label="Fitted line")  # Plot the fitted regression line
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Simple Linear Regression")
    plt.legend()

    # Convert the plot to a PNG image and encode it to base64 for rendering in the webpage
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()  # Close the plot to avoid memory issues

    return img_data  # Return the image data for display in the web page

# 6. 部署 (Deployment)
# Flask route to display the input form and results on the same page
@app.route('/', methods=['GET', 'POST'])
def index():
    img_data = None  # Placeholder for the regression plot
    slope = None  # Placeholder for the slope value
    intercept = None  # Placeholder for the intercept value

    if request.method == 'POST':
        try:
            # Collect user input from the form
            a = float(request.form['a'])
            noise = float(request.form['noise'])
            num_points = int(request.form['num_points'])

            # Generate data, perform linear regression, and generate the plot
            X, y, y_pred, slope, intercept = generate_data_and_model(a, noise, num_points)
            img_data = plot_results(X, y, y_pred)  # Generate the plot image

        except Exception as e:
            return f"An error occurred: {str(e)}"  # Handle any potential errors

    # Render the HTML template and pass the regression results and plot image (if available)
    return render_template('index.html', img_data=img_data, slope=slope, intercept=intercept)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)  # Start the Flask server
