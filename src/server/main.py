from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Message(BaseModel):
    text: str


@app.get("/info")
async def info():
    return {"Hello": "World"}


@app.post("/predict")
async def predict(message: Message):
    if message.text == "Hello":
        return {"prediction": "ham"}
    return {"prediction": "spam"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
