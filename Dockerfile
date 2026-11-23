FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
# --with-deps installs Chromium plus the system libraries it needs.
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
