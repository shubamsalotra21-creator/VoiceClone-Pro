import torch
import torchaudio
import numpy as np
import librosa
from typing import Dict, Any
import torch.nn as nn

class EmotionEmbedding:
    def __init__(self):
        self.emotion_profiles = {
            "neutral": {
                "pitch_shift": 0.0,
                "speaking_rate": 1.0,
                "energy": 1.0,
                "spectral_tilt": 0.0
            },
            "happy": {
                "pitch_shift": 2.0,
                "speaking_rate": 1.2,
                "energy": 1.3,
                "spectral_tilt": 0.5
            },
            "sad": {
                "pitch_shift": -2.0,
                "speaking_rate": 0.8,
                "energy": 0.7,
                "spectral_tilt": -0.3
            },
            "angry": {
                "pitch_shift": 1.5,
                "speaking_rate": 1.4,
                "energy": 1.5,
                "spectral_tilt": 0.8
            },
            "excited": {
                "pitch_shift": 3.0,
                "speaking_rate": 1.3,
                "energy": 1.4,
                "spectral_tilt": 0.6
            },
            "calm": {
                "pitch_shift": -1.0,
                "speaking_rate": 0.9,
                "energy": 0.9,
                "spectral_tilt": -0.2
            }
        }
    
    def get_emotion_profile(self, emotion: str, strength: float = 1.0) -> Dict[str, float]:
        base_profile = self.emotion_profiles.get(emotion, self.emotion_profiles["neutral"])
        
        adjusted_profile = {}
        for key, value in base_profile.items():
            if key == "pitch_shift":
                adjusted_profile[key] = value * strength
            elif key == "speaking_rate":
                adjusted_profile[key] = 1.0 + (value - 1.0) * strength
            elif key == "energy":
                adjusted_profile[key] = 1.0 + (value - 1.0) * strength
            elif key == "spectral_tilt":
                adjusted_profile[key] = value * strength
        
        return adjusted_profile

class EmotionController:
    def __init__(self, device="auto"):
        self.device = self._setup_device(device)
        self.emotion_embedding = EmotionEmbedding()
    
    def _setup_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def apply_emotion(self, audio: np.ndarray, emotion: str, strength: float = 1.0) -> np.ndarray:
        emotion_profile = self.emotion_embedding.get_emotion_profile(emotion, strength)
        
        processed_audio = audio.copy()
        
        sr = 22050
        
        if emotion_profile["pitch_shift"] != 0.0:
            processed_audio = self._apply_pitch_shift(processed_audio, sr, emotion_profile["pitch_shift"])
        
        if emotion_profile["speaking_rate"] != 1.0:
            processed_audio = self._apply_speaking_rate(processed_audio, sr, emotion_profile["speaking_rate"])
        
        if emotion_profile["energy"] != 1.0:
            processed_audio = self._apply_energy_adjustment(processed_audio, emotion_profile["energy"])
        
        if emotion_profile["spectral_tilt"] != 0.0:
            processed_audio = self._apply_spectral_tilt(processed_audio, sr, emotion_profile["spectral_tilt"])
        
        return processed_audio
    
    def _apply_pitch_shift(self, audio: np.ndarray, sr: int, shift: float) -> np.ndarray:
        try:
            shifted_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift)
            return shifted_audio
        except:
            return audio
    
    def _apply_speaking_rate(self, audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
        try:
            stretched_audio = librosa.effects.time_stretch(audio, rate=rate)
            return stretched_audio
        except:
            return audio
    
    def _apply_energy_adjustment(self, audio: np.ndarray, energy: float) -> np.ndarray:
        return audio * energy
    
    def _apply_spectral_tilt(self, audio: np.ndarray, sr: int, tilt: float) -> np.ndarray:
        try:
            stft = librosa.stft(audio)
            
            frequencies = np.fft.fftfreq(stft.shape[0], 1/sr)[:stft.shape[0]]
            
            tilt_filter = 1.0 + tilt * (frequencies / (sr/2))
            
            tilted_stft = stft * tilt_filter[:, np.newaxis]
            
            processed_audio = librosa.istft(tilted_stft)
            
            return processed_audio
        except:
            return audio
    
    def extract_emotion_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        features = {}
        
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        features["pitch_mean"] = pitch_mean
        
        rms_energy = librosa.feature.rms(y=audio)
        features["energy_mean"] = np.mean(rms_energy)
        
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features["spectral_centroid"] = np.mean(spectral_centroids)
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        features["zero_crossing_rate"] = np.mean(zero_crossing_rate)
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features["mfcc_mean"] = np.mean(mfccs, axis=1)
        
        return features

class NeuralEmotionController(EmotionController):
    def __init__(self, device="auto"):
        super().__init__(device)
        self.emotion_predictor = None
    
    def load_emotion_model(self):
        pass
    
    def predict_emotion_from_audio(self, audio: np.ndarray, sr: int) -> str:
        features = self.extract_emotion_features(audio, sr)
        
        energy = features["energy_mean"]
        pitch = features["pitch_mean"]
        spectral_centroid = features["spectral_centroid"]
        
        if energy > 0.1 and pitch > 200 and spectral_centroid > 3000:
            return "excited"
        elif energy > 0.08 and pitch > 180:
            return "happy"
        elif energy < 0.05 and pitch < 150:
            return "sad"
        elif energy > 0.12 and spectral_centroid > 4000:
            return "angry"
        else:
            return "neutral"