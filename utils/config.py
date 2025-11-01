import yaml
from pathlib import Path
from typing import Dict, Any
import os

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    config_path = Path(config_path)
    
    if not config_path.exists():
        default_config = {
            "audio": {
                "target_sr": 22050,
                "chunk_duration": 30.0,
                "normalize_volume": True,
                "remove_silence": True,
                "denoise": True
            },
            "models": {
                "voice_cloner": "tacotron2_waveglow",
                "emotion_controller": "neural",
                "embedding_extractor": "wav2vec2"
            },
            "generation": {
                "default_emotion": "neutral",
                "emotion_strength": 0.7,
                "speaking_rate": 1.0,
                "pitch_shift": 0.0
            },
            "ui": {
                "theme": "dark",
                "max_audio_duration": 60,
                "show_analytics": True
            }
        }
        save_config(default_config, config_path)
        return default_config
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], config_path: str = "config.yaml"):
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_audio_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("audio", {})

def get_model_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("models", {})

def update_config(section: str, key: str, value: Any):
    config = load_config()
    
    if section not in config:
        config[section] = {}
    
    config[section][key] = value
    save_config(config)

def get_default_generation_params() -> Dict[str, Any]:
    config = load_config()
    generation = config.get("generation", {})
    
    return {
        "emotion": generation.get("default_emotion", "neutral"),
        "emotion_strength": generation.get("emotion_strength", 0.7),
        "speaking_rate": generation.get("speaking_rate", 1.0),
        "pitch_shift": generation.get("pitch_shift", 0.0)
    }