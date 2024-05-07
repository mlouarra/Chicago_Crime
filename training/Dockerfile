FROM python:3.9

COPY . .

#RUN pip install mlflow

#CMD ["python3", "main.py"] #lancement api

RUN pip install mlflow fastapi uvicorn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

