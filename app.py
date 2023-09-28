from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__, template_folder="templates")
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    columns = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'grade', 'sqft_above',
        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
        'sqft_living15', 'sqft_lot15'
    ]
    
    values = [request.form[col] for col in columns]
    arr = np.array(values, dtype=np.float64)
    
    pred = -1 * model.predict([arr])
    
    return render_template('index.html', data=int(pred))

if __name__ == '__main__':
    app.run(debug=True)
