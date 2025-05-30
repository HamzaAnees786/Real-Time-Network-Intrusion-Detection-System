from flask import Flask, render_template, request, redirect, flash
import os
import pandas as pd
import joblib
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Load models and preprocessors
stacking_model = joblib.load('stacking_model.pkl')
catboost = joblib.load('catboost_model.pkl')
lightgbm = joblib.load('lgbm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result_summary, preview_table = analyze_csv(filepath)
            return render_template('result.html', result=result_summary, table=preview_table)
    return render_template('index.html')

def preprocess(df):
    df['Info'] = df['Info'].fillna('')
    known_protocols = set(label_encoder.classes_)
    df['Protocol'] = df['Protocol'].apply(lambda x: x if x in known_protocols else 'UNKNOWN')
    if 'UNKNOWN' not in label_encoder.classes_:
        label_encoder.classes_ = np.append(label_encoder.classes_, 'UNKNOWN')
    df['Protocol'] = label_encoder.transform(df['Protocol'])
    df[['Time', 'Length']] = scaler.transform(df[['Time', 'Length']])
    return df[['Time', 'Length', 'Protocol']]

def analyze_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        original_df = df.copy()
        features = preprocess(df)
        predictions = stacking_model.predict(features)
        predictions = np.where(df['Protocol'] == label_encoder.transform(['UNKNOWN'])[0], 0, predictions)

        original_df['Prediction'] = ['Malicious' if pred == 1 else 'Normal' for pred in predictions]

        malicious_count = (original_df['Prediction'] == 'Malicious').sum()
        normal_count = (original_df['Prediction'] == 'Normal').sum()
        total = len(original_df)

        result_summary = f"Total Packets: {total}<br>Malicious: {malicious_count}<br>Normal: {normal_count}"
        preview_table = original_df[['Time', 'Length', 'Protocol', 'Prediction']].head(20).to_html(classes='table', index=False)
    except Exception as e:
        result_summary = f"Error processing file: {str(e)}"
        preview_table = ""
    return result_summary, preview_table

if __name__ == '__main__':
    app.run(debug=True)