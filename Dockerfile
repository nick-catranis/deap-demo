FROM python:3.12-slim

WORKDIR /app

# Install dependencies at build time — source is mounted at runtime
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5050

# --reload watches the mounted source for file changes
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "1", "--reload", "app:app"]
