from pydantic import BaseModel

class HackRxResponse(BaseModel):
    """
    Pydantic model for the final /hackrx/run response body, as required by the hackathon.
    """
    answers: list[str]