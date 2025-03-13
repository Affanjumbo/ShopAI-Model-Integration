# Use Python 3.11 as the base image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Install system dependencies and ensure pip is available
RUN apt-get update && apt-get install -y python3-pip

# Copy all files to the container
COPY . /app

# Install dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Expose the API port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "models.recommendation_api:app", "--host", "0.0.0.0", "--port", "8000"]
