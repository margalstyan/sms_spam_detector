from fastapi import FastAPI, HTTPException, Request
from .models import Message
from src.model.main import nb

app = FastAPI()
routes_to_reroute = ['/']


@app.middleware('http')
async def middleware(request: Request, call_next):
    if request.url.path in routes_to_reroute:
        request.scope['path'] = '/docs'

    return await call_next(request)


@app.get("/info", tags=["Model Information"])
async def info():
    """
    **Information about the algorithm.**

    * **Returns:** algorithm name, version, status, status code, message, and error.
    """
    return {"algorithm_name": "Naive Bayes",
            "version": "0.0.1",
            "status": "up",
            "status_code": 200,
            "message": "OK",
            "error": ""}


@app.post("/predict", tags=["Prediction"])
async def predict(message: Message):
    """
    **Predicts whether a message is spam or ham.**

    * **Parameter:** message to classify.
    * **Returns:** prediction.
    """
    if message.message:
        prediction = nb.predict([message.message])
        if prediction[0] == 0:
            return {"prediction": "ham"}
        return {"prediction": "spam"}

    raise HTTPException(status_code=400, detail="Message should not be empty")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
