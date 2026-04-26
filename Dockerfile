# 1. Use a lightweight Python 3.11 base image
FROM python:3.11-slim

# 2. Hardening: Create a non-root user for execution
RUN useradd -m -s /bin/bash agentuser

# 3. Set up the working directory and mount points
WORKDIR /app
RUN mkdir /data /results && chown agentuser:agentuser /data /results

# 4. Copy the entire isolated monorepo
COPY . /app/

# 5. Install benchmarking/testing requirements
RUN pip install --no-cache-dir -r requirements-dev.txt

# 6. Absolute Path Injection (Bypasses setuptools completely)
ENV PYTHONPATH="/app/packages/glassbox-autofit/src:\
    /app/packages/glassbox-benchmark/src:\
    /app/packages/glassbox-eda/src:\
    /app/packages/glassbox-meta/src:\
    /app/packages/glassbox-ml/src:\
    /app/packages/glassbox-numpandas/src:\
    /app/packages/glassbox-optimization/src:\
    /app/packages/glassbox-pipeline/src:\
    /app/packages/glassbox-preprocessing/src:\
    /app/packages/glassbox-split/src"

# 7. Enforce the non-root user
USER agentuser

# 8. Default execution command
CMD ["python", "verify.py"]