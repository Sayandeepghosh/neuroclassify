FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

# Run train & benchmark on container startup or run interactive CLI
ENTRYPOINT ["neuroclassify"]
CMD ["benchmark"]
