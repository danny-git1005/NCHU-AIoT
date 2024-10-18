from flask import Flask, render_template, request
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import BytesIO
import base64

app = Flask(__name__)

# Function to generate synthetic data and perform linear regression
def generate_data_and_model(a, noise, num_points):
    np.random.seed(42)  # For reproducibility
    X = np.random.rand(num_points, 1) * 10  # X values between 0 and 10
    y = a * X + 5 + noise * np.random.randn(num_points, 1)  # y = ax + b with noise

    # Fit a simple linear regression model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    return X, y, y_pred, model.coef_[0][0], model.intercept_[0]

# Function to plot the results
def plot_results(X, y, y_pred):
    plt.scatter(X, y, color="blue", label="Data points")
    plt.plot(X, y_pred, color="red", label="Fitted line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Simple Linear Regression")
    plt.legend()

    # Convert the plot to a PNG image and encode it to base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()  # Close the plot to avoid memory issues

    return img_data

# Flask route to display the input form and results on the same page
@app.route('/', methods=['GET', 'POST'])
def index():
    img_data = None
    slope = None
    intercept = None

    if request.method == 'POST':
        try:
            a = float(request.form['a'])
            noise = float(request.form['noise'])
            num_points = int(request.form['num_points'])

            X, y, y_pred, slope, intercept = generate_data_and_model(a, noise, num_points)
            img_data = plot_results(X, y, y_pred)

        except Exception as e:
            return f"An error occurred: {str(e)}"

    return render_template('index.html', img_data=img_data, slope=slope, intercept=intercept)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
