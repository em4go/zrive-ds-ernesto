import json
from push_model import PushModel
from utils import load_data, create_model_path, load_model
import pandas as pd


def handler_fit(event, _):
    model_parametrisation = event["model_parametrisation"]
    threshold = event["threshold"]

    model = PushModel(model_parametrisation, threshold)

    df = load_data()
    df_train, _ = model.feature_label_split(df)

    model.fit(df_train)

    try:
        model_path = create_model_path("push_model")
        model.save(model_path)
    except Exception as e:
        return {
            "statusCode": "500",
            "body": json.dumps(
                {
                    "error": f"An error occurred while saving the model: {str(e)}",
                }
            ),
        }

    return {
        "statusCode": "200",
        "body": json.dumps(
            {
                "model_path": [model_path],
            }
        ),
    }


def handler_predict(event, _):
    data_to_predict = pd.DataFrame.from_dict(json.loads(event["users"]), orient="index")

    try:
        model = load_model(event["model_path"])
    except FileNotFoundError:
        return {
            "statusCode": "500",
            "body": json.dumps(
                {
                    "error": "Model not found",
                }
            ),
        }

    predictions = model.predict(data_to_predict)

    pred_dictionary = {}
    for user, prediction in zip(data_to_predict.index, predictions):
        pred_dictionary[user] = prediction

    return {
        "statusCode": "200",
        "body": json.dumps({"prediction": {pred_dictionary}}),
    }
