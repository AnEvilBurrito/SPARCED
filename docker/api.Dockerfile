# Standalone API image for SPARCED FastAPI server
# Builds the AMICI-generated SPARCED extension and serves the API via uvicorn.

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps needed for amici + SPARCED extension build
RUN apt-get update -qq && apt-get install -qq -y \
    build-essential \
    curl \
    git \
    python3-dev \
    python3-pip \
    libhdf5-serial-dev \
    libatlas-base-dev \
    swig \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps first for better layer caching
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir 'pip<21' \
    && pip3 install --no-cache-dir numpy==1.22.0 \
    && pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && pip3 install --no-cache-dir fastapi uvicorn

# Copy the repo
COPY . /workspace

# Copy input files and model files to workspace root (expected by simulator)
RUN cp -r /workspace/input_files/* /workspace/ 2>/dev/null || true \
    && cp -r /workspace/Demo/* /workspace/ 2>/dev/null || true

# Build/install the compiled SPARCED python extension
RUN pip3 install --no-cache-dir -e /workspace/Demo/SPARCED

EXPOSE 8000

# Default runtime settings (can be overridden)
ENV SPARCED_MODEL_DIR=Demo/SPARCED \
    SPARCED_SBML_FILE=SPARCED.xml \
    SPARCED_BASE_DIR=/workspace \
    SPARCED_DETERMINISTIC=1

CMD ["python3", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
