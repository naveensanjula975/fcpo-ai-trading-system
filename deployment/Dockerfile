FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps (TA-Lib, Postgres, etc. – adjust as needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY ./alembic ./alembic
COPY alembic.ini .
COPY ./docs ./docs

ENV APP_MODULE=fcpo_trading.main:app
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000

CMD ["uvicorn", "fcpo_trading.main:app", "--host", "0.0.0.0", "--port", "8000"]
