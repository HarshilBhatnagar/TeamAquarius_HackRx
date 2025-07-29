# ---- Builder Stage ----
# Use a full Python image which has the necessary tools to build dependencies
FROM python:3.11 as builder

WORKDIR /app

# Create and activate a virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---- Final Stage ----
# Use a slim image for the final, lightweight application
FROM python:3.11-slim

WORKDIR /app

# Copy the installed packages from the builder stage
COPY --from=builder /app/venv /app/venv

# Copy the application source code
COPY . .

# Activate the virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Expose the port and run the application
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]