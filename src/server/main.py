from fastapi import FastAPI, HTTPException, Request
from .models import Message, Messages
from src.model.main import nb, X_test, y_test

app = FastAPI()
routes_to_reroute = ['/']


@app.middleware('http')
async def middleware(request: Request, call_next):
    """
    Middleware to reroute the root path to the /docs path.

    Args:
        request (Request): request object.
        call_next: next middleware function.

    Returns:
        response: response object.
    """
    if request.url.path in routes_to_reroute:
        request.scope['path'] = '/docs'

    return await call_next(request)


@app.get("/info", tags=["Model Information"])
async def info():
    """
    **Information about the algorithm.**

    * **Returns:** algorithm name, related research papers, version, training data source.
    """
    return {"algorithm_name": "Naive Bayes",
            "model_score": round(nb.score(X_test, y_test), 3),
            "related_research_papers": [
                "https://www.researchgate.net/publication/228845263_An_Empirical_Study_of_the_Naive_Bayes_Classifier"],
            "version": "0.0.1",
            "training_data": "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset",
            }


@app.post("/predict", tags=["Prediction"])
async def predict(message: Message):
    """
    **Predicts whether a message is spam or ham.**

    * **Parameter:** message to classify.

    * **Returns:** prediction.
    """
    message = message.message.strip()
    if message:
        prediction = nb.predict([message])
        if prediction[0] == 0:
            return {"prediction": "ham"}
        return {"prediction": "spam"}

    raise HTTPException(status_code=400, detail="Message should not be empty")


@app.post("/predict_all", tags=["Prediction"])
async def predict_all(messages: Messages, ignore_empty: bool = False):
    """
    **Predicts whether a list of messages are spam or ham.**

    * **Parameter:** list of messages to classify.

    * **Returns:** list of predictions.

    * **Optional Parameter:** ignore_empty (default: False) - whether to ignore empty message(s).
    """
    message_list = [message.strip() for message in messages.messages if message.strip()]
    if message_list and not any(message.strip() == "" for message in messages.messages) or ignore_empty:
        predictions = nb.predict(message_list)
        for i in range(len(predictions)):
            if predictions[i] == 0:
                predictions[i] = "ham"
            else:
                predictions[i] = "spam"
        return {"predictions": predictions}

    raise HTTPException(status_code=400, detail="Message(s) should not be empty")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
