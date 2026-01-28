from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your model - Ensure 'model.pkl' is in the same folder as this script
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: model.pkl not found. Please export it from your notebook first.")

@app.route('/')
def landing():
    """Renders the initial landing page."""
    return render_template('landing.html')

@app.route('/home')
def home():
    """Renders the dashboard page."""
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Mapping for the LabelEncoder used in your notebook: H=0, L=1, M=2
        type_map = {'H': 0, 'L': 1, 'M': 2}
        
        try:
            # 1. Collect the 6 inputs from the UI
            m_type = type_map[request.form['type']]
            air_temp = float(request.form['air_temp'])
            proc_temp = float(request.form['proc_temp'])
            speed = float(request.form['speed'])
            torque = float(request.form['torque'])
            wear = float(request.form['wear'])

            # 2. Append 5 zeros for the failure flags (TWF, HDF, PWF, OSF, RNF)
            # This creates the 11-feature vector the model expects
            input_features = [m_type, air_temp, proc_temp, speed, torque, wear, 0, 0, 0, 0, 0]
            final_features = np.array([input_features])
            
            # 3. Perform Prediction
            prediction = model.predict(final_features)
            
            if prediction[0] == 1:
                result = "🚨 FAILURE PREDICTED: Maintenance Required"
                color = "#ff4b2b" # Bright Red
            else:
                result = "✅ MACHINE HEALTHY: No Issues Detected"
                color = "#2ecc71" # Safe Green
            
            return render_template('home.html', 
                                 prediction_text=result, 
                                 res_color=color)

        except Exception as e:
            # Fallback color if something goes wrong with inputs
            return render_template('home.html', 
                                 prediction_text="⚠️ Input Error: Please enter valid numbers.", 
                                 res_color="#f1c40f") # Warning Yellow

if __name__ == "__main__":
    app.run(debug=True)
