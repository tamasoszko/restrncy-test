import os
from agents import set_tracing_disabled
import dotenv
import mlflow


dotenv.load_dotenv()


def init_mlflow_tracing():

    MLFLOW_TRACING_URL = os.getenv("MLFLOW_TRACING_URL")


    if MLFLOW_TRACING_URL is not None:
        mlflow.openai.autolog()
        mlflow.set_tracking_uri(MLFLOW_TRACING_URL)
        mlflow.set_experiment("OpenAI Agent")
        print(f"Using MLFlow tracing: {MLFLOW_TRACING_URL}")
    else:
        print("MLFlow tracing is not enabled")

    # Disable default tracing
    set_tracing_disabled(True)

