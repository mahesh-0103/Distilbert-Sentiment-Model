FROM python:3.9

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Set the Python Path so it can find the 'api' folder
ENV PYTHONPATH=/app

# Start the FastAPI app directly
# Replace 'api.app:app' if your entry file is different
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
