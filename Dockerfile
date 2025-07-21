FROM python:3.13-slim

# Install Poetry.
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy the application into the container.
COPY . /app

# Install the application dependencies using Poetry.
WORKDIR /app
RUN poetry config virtualenvs.create false && poetry install --no-root 

ENV PORT=8080

# CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
CMD [ "sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}" ]