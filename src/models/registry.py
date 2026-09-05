import os
import glob
import joblib
import hopsworks
from src.config import HOPSWORKS_API_KEY,CONFIG
from src.utils.logger import logger


class HopsworksModelRegistry:
    """
    Class-based wrapper around the Hopsworks Model Registry.
    Used by the Streamlit dashboard to load the latest trained model.
    """

    def __init__(self, model_name: str = "islamabad_aqi_model_24h"):
        self.model_name = model_name
        self._project = None
        self._mr = None

    def _connect(self):
        """Lazily connects to Hopsworks on first use."""
        if self._mr is None:
            if not HOPSWORKS_API_KEY:
                raise ValueError("HOPSWORKS_API_KEY is not set.")
            self.project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY,project=CONFIG["feature_store"]["project_name"])
            self._mr = self._project.get_model_registry()

    def load_latest_model(self):
        """
        Downloads and loads the latest version of the model from the registry.
        Falls back to a locally saved model if Hopsworks is unreachable.
        Returns None if no model is found anywhere.
        """
        # 1. Try Hopsworks first
        try:
            self._connect()
            logger.info(f"Fetching latest '{self.model_name}' from Hopsworks registry...")
            model = self._mr.get_model(self.model_name)

            local_dir = os.path.join("saved_models", "inference")
            os.makedirs(local_dir, exist_ok=True)

            model_path = model.download(local_dir)

            # Find the .pkl or .joblib file inside the downloaded artifact
            pkl_path = os.path.join(model_path, "model.pkl")
            if not os.path.exists(pkl_path):
                candidates = (
                    glob.glob(os.path.join(model_path, "*.pkl")) +
                    glob.glob(os.path.join(model_path, "*.joblib"))
                )
                if not candidates:
                    raise FileNotFoundError(f"No model file found in {model_path}")
                pkl_path = candidates[0]

            logger.info(f"Loading model from {pkl_path}")
            return joblib.load(pkl_path)

        except Exception as e:
            logger.warning(f"Could not load model from Hopsworks: {e}")

        # 2. Fall back to local saved model
        local_fallback = os.path.join("saved_models", "model.pkl")
        if os.path.exists(local_fallback):
            logger.info(f"Loading local fallback model from {local_fallback}")
            return joblib.load(local_fallback)

        logger.warning("No trained model found in registry or local storage.")
        return None


# ── Standalone functions (used by train.py and predict.py) ──────────────────

def upload_model(model, model_name: str, metrics: dict, description: str):
    """Saves model locally and uploads it to Hopsworks Model Registry."""
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set.")

    local_dir = "saved_models"
    os.makedirs(local_dir, exist_ok=True)
    model_path = os.path.join(local_dir, "model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved locally to {model_path}")

    logger.info("Authenticating with Hopsworks...")
    project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project=CONFIG["feature_store"]["project_name"])

    logger.info(f"Uploading model '{model_name}' to registry...")
    hs_model = mr.python.create_model(
        name=model_name,
        metrics=metrics,
        description=description
    )
    hs_model.save(local_dir)
    logger.info(f"Model '{model_name}' successfully registered!")


def download_model(model_name: str, version: int = None):
    """Downloads a specific model version from the registry."""
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set.")

    project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project=CONFIG["feature_store"]["project_name"])
    mr = project.get_model_registry()

    model = mr.get_model(model_name, version=version) if version else mr.get_model(model_name)

    local_dir = os.path.join("saved_models", "inference")
    os.makedirs(local_dir, exist_ok=True)

    model_path = model.download(local_dir)
    pkl_path = os.path.join(model_path, "model.pkl")
    return joblib.load(pkl_path)