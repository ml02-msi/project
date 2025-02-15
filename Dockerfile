# # Use an official Python image as a base
# FROM python:3.10-slim

# # Set the working directory inside the container
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# # Install required Python packages
# RUN pip install --no-cache-dir fastapi uvicorn numpy pandas sentence-transformers scikit-learn

# # Copy only app.py into the container
# COPY app.py /app

# # Download the latest installer
# ADD https://astral.sh/uv/install.sh /uv-installer.sh

# # Run the installer then remove it
# RUN sh /uv-installer.sh && rm /uv-installer.sh

# # Ensure the installed binary is on the `PATH`
# ENV PATH="/root/.local/bin/:$PATH"

# # Expose the port FastAPI runs on
# EXPOSE 8000

# # Command to run the application
# CMD ["uv", "run", "app.py"]



## Multi Stage Build

# Base image with dependencies only
FROM python:3.10-slim AS base
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates nodejs npm && rm -rf /var/lib/apt/lists/*

# Install Prettier globally
RUN npm install -g prettier@3.4.2

# Install Python dependencies
RUN pip install --no-cache-dir fastapi numpy pandas \
    scikit-learn requests python-dateutil python-dotenv uvicorn \
    db-sqlite3 duckdb Faker pillow httpx

# Download and install UV
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Final image with app.py (uses cached dependencies)
FROM base AS final
WORKDIR /app
RUN mkdir -p /data
COPY app.py datagen.py evaluate.py /app/

# Verify installation
RUN node -v && npm -v && npx prettier --version

CMD ["uv", "run", "app.py"]
