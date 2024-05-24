from pydantic import BaseModel


class Message(BaseModel):
    """
    Message model for the /predict endpoint.
    """
    message: str = "Hi there!"
