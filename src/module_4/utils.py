from push_model import PushModel
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

DATA_PATH = "../../data/box_builder_dataset/feature_frame.csv"


def load_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def create_model_path(model_folder_path: str, model_name: str = "push.joblib") -> str:
    model_name_without_extension = Path(model_name).stem
    extension = ".joblib"

    model_path = Path(model_folder_path) / model_name_without_extension + extension

    if model_path.exists():
        date = datetime.now().strftime("%yyyy_%mm_%dd")
        model_path = (
            Path(model_folder_path) / f"{model_name_without_extension}_{date}"
            + extension
        )

    return str(model_path)


def load_model(model_path: str) -> PushModel:
    return joblib.load(model_path)
