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

# # Base image with dependencies only
# FROM python:3.9-slim AS base
# WORKDIR /app
# # Install required system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     curl ca-certificates nodejs npm && rm -rf /var/lib/apt/lists/*

# # Install Prettier globally
# # Install Prettier with proper path handling
# # Install nvm and Node.js version 20.17.0 using nvm
# # RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash && \
# #     export NVM_DIR="/root/.nvm" && \
# #     [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" && \
# #     nvm install 20.17.0 && \
# #     nvm alias default 20.17.0 && \
# #     nvm use default && \
# #     node -v && npm -v

# # Set environment variables so that subsequent commands use the installed Node.js
# # ENV NVM_DIR=/root/.nvm
# # ENV NODE_VERSION=20.17.0
# # ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# # Install Prettier globally
# RUN npm install -g prettier@3.4.2

# # Download and install UV
# ADD https://astral.sh/uv/install.sh /uv-installer.sh
# RUN sh /uv-installer.sh && rm /uv-installer.sh
# ENV PATH="/root/.local/bin/:$PATH"

# # Python virtual environment
# # ENV VIRTUAL_ENV=/opt/venv
# # RUN python3 -m venv $VIRTUAL_ENV
# # ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# # Install Python dependencies
# RUN pip install --no-cache-dir fastapi numpy pandas \
#     scikit-learn requests python-dateutil python-dotenv uvicorn \
#     db-sqlite3 duckdb Faker pillow httpx

# # Git
# RUN apt-get update && apt-get install -y git
# RUN git config --global user.name "ProjGit" \
#     && git config --global user.email "proj1@tds.iitm.ac.in"

# # Final image with app.py (uses cached dependencies)
# FROM base AS final
# WORKDIR /app
# RUN mkdir -p /data
# COPY app.py datagen.py evaluate.py /app/

# # Verify installation
# RUN node -v && npm -v && npx prettier --version

# CMD ["uv", "run", "app.py"]



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
# Install Prettier with proper path handling
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