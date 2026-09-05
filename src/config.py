import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
class _Location:
    def __init__(self, d):
        self.city      = d["name"]
        self.latitude  = d["latitude"]
        self.longitude = d["longitude"]

class _Settings:
    def __init__(self, cfg):
        self.location = _Location(cfg["location"])



CONFIG = load_config()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
AQICN_API_KEY = os.getenv("AQICN_API_KEY")
settings = _Settings(CONFIG)