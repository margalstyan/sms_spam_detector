from pydantic import BaseModel


class Message(BaseModel):
    """
    Message model for the /predict endpoint.
    """
    message: str = "Hi there!"


class Messages(BaseModel):
    """
    Message model for the /predict endpoint.
    """
    messages: list = ["Hi there!", "How are you?", "Click on link below to earn $5000"]
