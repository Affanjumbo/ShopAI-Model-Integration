# 1️⃣ Use Python 3.11 as the base image
FROM python:3.11

# 2️⃣ Set the working directory inside the container
WORKDIR /app

# 3️⃣ Copy all files to the container
COPY . /app

# 4️⃣ Install pip manually
RUN apt-get update && apt-get install -y python3-pip

# 5️⃣ Install required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Expose the API port
EXPOSE 8000

# 7️⃣ Run the FastAPI application
CMD ["uvicorn", "models.recommendation_api:app", "--host", "0.0.0.0", "--port", "8000"]