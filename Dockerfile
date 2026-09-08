# Python 3.10 slim image - kicsi és gyors
FROM python:3.10-slim

# Munkakönyvtár beállítása
WORKDIR /app

# Rendszerfüggőségek (ha esetleg kellenek build toolok a C++ libekhez)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Függőségek másolása és telepítése (cache-elhető réteg)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Forráskód másolása
COPY . .

# Környezeti változók alapértékei
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default parancs: Streamlit anomália dashboard
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]