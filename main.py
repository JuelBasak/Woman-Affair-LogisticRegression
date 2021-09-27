from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    rate_marriage = float(request.form['rate_marriage'])
    age = float(request.form['age'])
    yrs_married = float(request.form['yrs_married'])
    children = float(request.form['children'])
    religious = float(request.form['religious'])
    educ = float(request.form['educ'])
    occupation = float(request.form['occupation'])
    occupation_husb = float(request.form['occupation_husb'])

    data = pd.DataFrame([[rate_marriage, age, yrs_married, children, religious, educ, occupation, occupation_husb]],
    columns=['rate_marriage', 'age', 'yrs_married', 'children', 'religious', 'educ', 'occupation', 'occupation_husb'])

    data['married_to_age'] = data['yrs_married'] / data['age']
    data.drop(['yrs_married', 'age'], axis=1, inplace=True)

    full_pipeline = joblib.load('model/full_pipeline.pkl')
    prepared_data = full_pipeline.transform(data)

    log_reg = joblib.load('model/logistic_regression.pkl')
    output = log_reg.predict(prepared_data)

    return render_template('index.html', output=output)


if __name__ == '__main__':
    app.run(debug=True)
