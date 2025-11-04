FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy only source code (exclude trained model)
COPY models/ /models/
COPY src/ /src/
COPY config/ /config/
COPY .env .

EXPOSE 5000

ENV FLASK_APP=src.serving.app
ENV FLASK_ENV=development
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/v1/health || exit 1

CMD ["flask", "run", "--host=0.0.0.0"]