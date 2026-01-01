FROM python:3.11.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY models/ models/

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8000"]
