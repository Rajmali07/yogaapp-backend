FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["sh", "-c", "gunicorn -w 1 -k uvicorn.workers.UvicornWorker --timeout 300 --bind 0.0.0.0:${PORT:-10000} main:app"]
