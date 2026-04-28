FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY rotator.py .

EXPOSE 3128

CMD ["python3", "-m", "proxy", "--hostname", "0.0.0.0", "--port", "3128", "--plugins", "proxy_rotator.rotator.RotatingProxyPlugin", "--timeout", "30"]
