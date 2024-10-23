# Use the specified base image
FROM python:3.11.5-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

COPY ["*.py", "*.bin", "./"]

EXPOSE 9696

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "app:app"]