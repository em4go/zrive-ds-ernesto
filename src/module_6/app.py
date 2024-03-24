import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from basket_model.feature_store import FeatureStore
from basket_model.basket_model import BasketModel
from basket_model.exceptions import UserNotFoundException, PredictionException
import pathlib
from loguru import logger
import time
import uuid


working_dir = pathlib.Path(__file__).parent.resolve()
logger.add(working_dir / "log.txt", rotation="100 MB")
logger.add(working_dir / "log.json", format="{time} {level} {message}", serialize=True)


class PredictionRequest(BaseModel):
    USER_ID: str


# Create an instance of FastAPI
app = FastAPI()

feature_store = FeatureStore()
basket_model = BasketModel()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = uuid.uuid4()
    logger.info(f"rid={rid} start request path={request.url.path}")
    start_time = time.time()

    request.state.rid = rid

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    formatted_process_time = "{0:.2f}".format(process_time)
    logger.info(
        f"rid={rid} completed_in={formatted_process_time}ms status_code={response.status_code}"
    )

    return response


@app.get("/")
async def read_root():
    return {"message": "Hello, module 6!"}


@app.get("/status")
async def read_status():
    return {"status": 200}


@app.post("/predict")
def predict(request: Request, user_id: PredictionRequest):
    try:
        user_features = feature_store.get_features(user_id.USER_ID)
        prediction = basket_model.predict(user_features)
        logger.info(f"rid={request.state.rid} prediction={prediction.tolist()}")
        return {
            "prediction": prediction.tolist(),
        }
    except UserNotFoundException as e:
        logger.error(f"rid={request.state.rid} {str(e)}")
        raise HTTPException(status_code=404, detail=(str(e)))
    except PredictionException as e:
        logger.error(f"rid={request.state.rid} {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# This block allows you to run the application using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
