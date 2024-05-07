from fastapi import FastAPI
from run_train import train_once

app = FastAPI()


@app.get("/train")
def first_training():
    train_once()
