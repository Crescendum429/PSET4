FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY build_obt.py .

RUN mkdir -p /app/logs

ENTRYPOINT ["python", "build_obt.py"]
CMD ["--mode", "full", "--year-start", "2015", "--year-end", "2025"]
