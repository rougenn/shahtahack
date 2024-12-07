// Найдём элементы кнопки и input
const uploadButton = document.getElementById('upload-button');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');

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
            fileInfo.innerHTML = `<p>Загружен файл: <strong>${file.name}</strong></p>`;
            console.log(`Выбран файл: ${file.name}`);
        } else {
            fileInfo.innerHTML = `<p style="color: red;">Ошибка: Файл должен быть в формате CSV!</p>`;
        }
    } else {
        fileInfo.innerHTML = `<p>Файл не выбран.</p>`;
    }
});
