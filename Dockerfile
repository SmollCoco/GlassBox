# 1. Use a lightweight Python 3.11 base image
FROM python:3.11-slim

# 2. Hardening: Create a non-root user for OpenShell execution
RUN useradd -m -s /bin/bash agentuser

# 3. Set up the working directory and mount points
WORKDIR /app
RUN mkdir /data /results && chown agentuser:agentuser /data /results

# 4. Copy the entire monorepo
COPY . /app/

# 5. Install benchmarking tools via requirements
RUN pip install --no-cache-dir -r requirements-dev.txt

# 6. Install the GlassBox ecosystem in editable mode to bypass auto-discovery bugs
RUN pip install --no-cache-dir -e .

# 7. Enforce the non-root user
USER agentuser

# 8. Default execution command
CMD ["python", "verify.py"]