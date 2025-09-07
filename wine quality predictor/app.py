from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load pipeline once
pipeline = joblib.load("wine_pipeline.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect features
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # Predict with pipeline
        prediction = pipeline.predict(final_features)[0]

        return render_template("index.html",
                               prediction_text=f"Predicted Wine Quality: {prediction}")
    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {str(e)}")

if __name__ =="__main__":
    app.run(debug=False, use_reloader=False)
