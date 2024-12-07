# Используем официальный Python образ
FROM python:3.12-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app/backend/python

# Копируем зависимости и устанавливаем их
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Копируем весь проект в контейнер
COPY . /app/

# Открываем порт 5000, на котором будет работать Flask
EXPOSE 5000

# Команда для запуска сервера
CMD ["python", "server.py"]
