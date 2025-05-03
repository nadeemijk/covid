from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model (make sure the model.pkl file is in the same folder)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')
def get_checkbox_value(field_name):
    return int(request.form.get(field_name, 0))  # Returns 0 if not checked


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            sex = int(request.form['sex'])
            age = int(request.form['age'])
            # Get input data from the form
            pneumonia = get_checkbox_value('pneumonia')
            diabetes = get_checkbox_value('diabetes')
            copd = get_checkbox_value('copd')
            asthma = get_checkbox_value('asthma')
            immunosuprresed = get_checkbox_value('immunosuprresed')
            hypertension = get_checkbox_value('hypertension')
            cardiovascular = get_checkbox_value('cardiovascular')
            renal_chronic = get_checkbox_value('renal_chronic')
            tobacco = get_checkbox_value('tobacco')
            obesity = get_checkbox_value('obesity')
            otherdis = get_checkbox_value('otherdis')

            # Create a feature array
            features = np.array([[sex, pneumonia, age, diabetes, hypertension, asthma, copd, immunosuprresed, cardiovascular, renal_chronic, tobacco, obesity,otherdis]])

            # Predict using the model
            prediction = model.predict(features)

            # Return result to template
            result = "Patient may not have survived" if prediction[0] == 1 else "Patient likely SURVIVED "
            return render_template('index.html', prediction_text=result)
        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run()

