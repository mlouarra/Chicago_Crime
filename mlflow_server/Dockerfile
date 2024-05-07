FROM python:3.9

RUN pip install mlflow

EXPOSE 8080

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "8080"]

