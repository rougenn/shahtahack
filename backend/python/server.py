from flask import Flask, request, send_file, send_from_directory
import os
import io
from werkzeug.utils import secure_filename

from script import predict

app = Flask(__name__, static_folder='build', static_url_path='')

# Папка для сохранения загруженных файлов
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Разрешенные типы файлов
ALLOWED_EXTENSIONS = {'csv'}

# Функция для проверки типа файла
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
        
        # Вызываем вашу функцию predict
        result_csv = predict(filepath)

        # Создаем файл для отправки
        result_file = io.StringIO(result_csv)
        result_file.seek(0)
        
        return send_file(io.BytesIO(result_file.getvalue().encode('utf-8')),
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name='result.csv')

    return "Invalid file", 400

# Роут для главной страницы, которая будет отдавать index.html
@app.route('/')
def serve_index():
    return send_from_directory('build', 'index.html')

# Включаем сервер
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
