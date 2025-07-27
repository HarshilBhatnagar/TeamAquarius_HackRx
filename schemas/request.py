from pydantic import BaseModel, HttpUrl

class HackRxRequest(BaseModel):
    """
    Pydantic model for the /hackrx/run request body.
    """
    documents: HttpUrl
    questions: list[str]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "documents": "https://arxiv.org/pdf/1706.03762.pdf",
                    "questions": [
                        "What is the title of the paper?",
                        "Who are the authors?"
                    ]
                }
            ]
        }
    }