# Stage 1: Build stage with dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install poetry for dependency management
RUN pip install poetry

# Copy only dependency files to leverage Docker cache
COPY poetry.lock pyproject.toml ./

# Install dependencies into a virtual environment
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --no-dev

# Stage 2: Final application stage
FROM python:3.11-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv ./.venv

# Set the PATH to use the virtual environment's binaries
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application source code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]