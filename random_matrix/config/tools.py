from pathlib import Path
import json

_config_path = Path(__file__).parent / "config.json"

def load_config() -> dict:
    if not _config_path.exists():
        return {"data_directory": str(Path.cwd())}
    with open(_config_path) as f:
        return json.load(f)

def save_config(config: dict) -> None:
    with open(_config_path, "w") as f:
        json.dump(config, f, indent=2)

def get_data_directory() -> Path:
    config = load_config()
    return Path(config["data_directory"])

def set_data_directory(new_path: str | Path) -> None:
    config = load_config()
    config["data_directory"] = str(Path(new_path))
    save_config(config)

def print_data_directory() -> None:
    print(f"DATA_DIRECTORY = {get_data_directory()}")