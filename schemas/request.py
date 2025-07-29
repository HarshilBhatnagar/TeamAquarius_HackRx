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
                    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
                    "questions": [
                        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                        "What is the waiting period for pre-existing diseases (PED) to be covered?"
                    ]
                }
            ]
        }
    }