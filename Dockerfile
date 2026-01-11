# -----------------------------------------------------------------------------
# Streamlit Dockerfile
# - Builds the Streamlit UI container for the CNPJ analytics dashboard
# - Runs on port 8501
# - Entry point: utils/app.py
# -----------------------------------------------------------------------------

    FROM python:3.11-slim

    # App directory inside the container
    WORKDIR /app
    
    # -----------------------------------------------------------------------------
    # Dependencies
    # -----------------------------------------------------------------------------
    COPY streamlit_app/requirements.txt ./requirements.txt
    RUN pip install --no-cache-dir -r requirements.txt
    
    # -----------------------------------------------------------------------------
    # Application code
    # -----------------------------------------------------------------------------
    COPY streamlit_app/ .
    
    # Make /app importable so `utils` can be imported as a package
    ENV PYTHONPATH=/app
    
    # Streamlit default port
    EXPOSE 8501
    
    # Start Streamlit
    CMD ["streamlit", "run", "utils/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]