FROM python:3.11-slim

WORKDIR /app

COPY backend/ ./backend
COPY frontend/ ./frontend

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

EXPOSE 5000

WORKDIR /app/backend

CMD ["python", "app.py"]