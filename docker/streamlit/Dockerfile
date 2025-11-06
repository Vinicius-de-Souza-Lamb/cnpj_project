FROM python:3.11-slim

WORKDIR /app

# Install Python deps first (cache-friendly)
COPY streamlit_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY streamlit_app/ .

# Make sure /app is on Python’s import path
ENV PYTHONPATH=/app

# Expose Streamlit’s default port inside the container
EXPOSE 8501

# Run the app (main script lives in utils/app.py; pages are in utils/pages/)
CMD ["streamlit","run","utils/pages/geografia.py","--server.address","0.0.0.0","--server.port","8501"]






