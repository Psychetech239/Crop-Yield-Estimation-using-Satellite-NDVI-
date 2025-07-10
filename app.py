from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

try:
    model = joblib.load('random_forest_model.pkl')
except FileNotFoundError:
    print("Error: 'random_forest_model.pkl' not found. Make sure the model file is in the correct directory.")
    
    model = None # Set model to None if not found

# Route for the home page (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Route for the form page (form.html)
@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/result', methods=['POST'])
def result():
    if model is None:
        return "Error: Model not loaded. Please check 'random_forest_model.pkl'."

    try:
        # Get data from the form
        temperature = float(request.form['temperature'])
        Precipitation = float(request.form['Precipitation']) # Note: Flask request.form keys are case-sensitive
        ndvi = float(request.form['ndvi'])

        input_data = np.array([[Precipitation, temperature, ndvi]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Render result template with the prediction, rounded to 2 decimal places
        return render_template('result.html', prediction=round(prediction, 2))
    except ValueError:
        return "Invalid input. Please enter numeric values for all fields."
    except KeyError as e:
        return f"Missing form field: {e}. Please ensure all fields are filled."
    except Exception as e:
        return f"An unexpected error occurred during prediction: {e}"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True) # debug=True enables auto-reloading and debugger