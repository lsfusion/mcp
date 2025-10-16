
# Builder stage
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Final image
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=base /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=base /usr/local/bin /usr/local/bin
COPY --from=base /app /app

# Non-root user for security
RUN useradd -u 10001 -m appuser
USER appuser

EXPOSE 8000
CMD ["python", "server.py", "http", "--host", "0.0.0.0", "--port", "8000"]
