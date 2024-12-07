from flask import Flask, request, send_file, send_from_directory
import os
import io
from werkzeug.utils import secure_filename

from script import predict

from flask_cors import CORS

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        df_result = predict(filepath)
        
        df_result = df_result.round(4)
        
        result_csv = io.StringIO()
        df_result.to_csv(result_csv, index=False)
        result_csv.seek(0)
        
        return send_file(io.BytesIO(result_csv.getvalue().encode('utf-8')),
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name='result.csv')

    return "Invalid file", 400

@app.route('/')
def serve_index():
    return send_from_directory('build', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
