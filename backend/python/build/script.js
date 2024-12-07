// Найдём элементы кнопки и input
const uploadButton = document.getElementById('upload-button');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const getResultsButton = document.getElementById('get-results-button');

// Переменная для хранения выбранного файла
let selectedFile = null;

// Когда пользователь нажимает на кнопку "загрузить датасет"
uploadButton.addEventListener('click', () => {
    fileInput.click(); // Открыть диалог выбора файла
});

// Когда пользователь выбирает файл
fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0]; // Получаем первый выбранный файл

    if (file) {
        // Проверяем, что файл в формате CSV
        if (file.type === "text/csv") {
            selectedFile = file;
            fileInfo.innerHTML = `<p>Загружен файл: <strong>${file.name}</strong></p>`;
            console.log(`Выбран файл: ${file.name}`);
        } else {
            fileInfo.innerHTML = `<p style="color: red;">Ошибка: Файл должен быть в формате CSV!</p>`;
            selectedFile = null;
        }
    } else {
        fileInfo.innerHTML = `<p>Файл не выбран.</p>`;
        selectedFile = null;
    }
});

// Когда пользователь нажимает кнопку "получить диапазоны"
getResultsButton.addEventListener('click', () => {
    // Проверяем, был ли выбран файл
    if (!selectedFile) {
        fileInfo.innerHTML = `<p style="color: red;">Ошибка: Выберите файл для отправки.</p>`;
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile); // Добавляем файл в форму

    // Отправляем файл на сервер
    fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.ok) {
            // Если ответ успешный, получаем CSV файл
            return response.blob();
        }
        throw new Error('Ошибка при получении файла');
    })
    .then(blob => {
        // Создаем ссылку для скачивания полученного CSV
        const link = document.createElement('a');
        const url = window.URL.createObjectURL(blob);
        link.href = url;
        link.download = 'result.csv'; // Имя скачиваемого файла
        link.click(); // Имитируем клик для скачивания файла
        window.URL.revokeObjectURL(url); // Освобождаем URL
    })
    .catch(error => {
        fileInfo.innerHTML = `<p style="color: red;">Ошибка: ${error.message}</p>`;
        console.error(error);
    });
});
