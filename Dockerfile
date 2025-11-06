FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY streamlit_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY streamlit_app/ .

# Make /app importable so "utils" is importable
ENV PYTHONPATH=/app

EXPOSE 8501
CMD ["streamlit", "run", "utils/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
