FROM python:3.12-slim

# Prevent bytecode noise and buffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 600 --retries 10 -r requirements.txt

# Copy application source and static frontend.
COPY main.py .
COPY app/ ./app/
COPY frontend/ ./frontend/

# Copy model/vectorizer assets (required to run predictions).
COPY my_model.h5 countvectorizer.pkl ./

# Data directory persisted by container.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "main.py"]