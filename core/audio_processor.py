import torchaudio
import numpy as np
import librosa
import noisereduce as nr
from scipy import signal
from typing import Union, List

class AudioProcessor:
    def __init__(self, target_sr: int = 22050):
        self.target_sr = target_sr
    
    def preprocess_audio(self, audio_path: str, duration: float = 30.0) -> str:
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        
        if len(audio) > duration * sr:
            audio = audio[:int(duration * sr)]
        
        audio = self.remove_silence(audio, sr)
        
        audio = self.normalize_volume(audio)
        
        audio = self.denoise(audio, sr)
        
        output_path = "temp_processed.wav"
        torchaudio.save(output_path, torch.from_numpy(audio).unsqueeze(0), sr)
        
        return output_path
    
    def remove_silence(self, audio: np.ndarray, sr: int, threshold: float = 0.02) -> np.ndarray:
        intervals = librosa.effects.split(audio, top_db=30)
        
        if len(intervals) == 0:
            return audio
        
        non_silent_audio = np.concatenate([audio[start:end] for start, end in intervals])
        
        return non_silent_audio
    
    def denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        try:
            denoised_audio = nr.reduce_noise(y=audio, sr=sr)
            return denoised_audio
        except:
            return audio
    
    def normalize_volume(self, audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 10**(target_level / 20)
            gain = target_rms / rms
            normalized_audio = audio * gain
            return np.clip(normalized_audio, -1.0, 1.0)
        return audio
    
    def change_speed(self, audio: np.ndarray, speed_factor: float) -> np.ndarray:
        try:
            stretched_audio = librosa.effects.time_stretch(audio, rate=speed_factor)
            return stretched_audio
        except:
            return audio
    
    def shift_pitch(self, audio: np.ndarray, n_steps: float) -> np.ndarray:
        try:
            shifted_audio = librosa.effects.pitch_shift(audio, sr=self.target_sr, n_steps=n_steps)
            return shifted_audio
        except:
            return audio
    
    def apply_high_pass_filter(self, audio: np.ndarray, cutoff: float) -> np.ndarray:
        nyquist = self.target_sr / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        filtered_audio = signal.filtfilt(b, a, audio)
        return filtered_audio
    
    def apply_low_pass_filter(self, audio: np.ndarray, cutoff: float) -> np.ndarray:
        nyquist = self.target_sr / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        filtered_audio = signal.filtfilt(b, a, audio)
        return filtered_audio
    
    def extract_audio_features(self, audio: np.ndarray, sr: int) -> dict:
        features = {}
        
        features["duration"] = len(audio) / sr
        
        features["rms_energy"] = np.sqrt(np.mean(audio**2))
        
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features["spectral_centroid"] = np.mean(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features["spectral_rolloff"] = np.mean(spectral_rolloff)
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        features["zero_crossing_rate"] = np.mean(zero_crossing_rate)
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features["mfccs"] = np.mean(mfccs, axis=1)
        
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[pitches > 0]
        features["pitch_mean"] = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        
        return features

class RealTimeAudioProcessor(AudioProcessor):
    def __init__(self, target_sr: int = 22050, chunk_size: int = 1024):
        super().__init__(target_sr)
        self.chunk_size = chunk_size
        self.buffer = np.array([])
    
    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        
        if len(self.buffer) >= self.chunk_size * 4:
            processed_chunk = self.denoise(self.buffer[:self.chunk_size], self.target_sr)
            self.buffer = self.buffer[self.chunk_size:]
            return processed_chunk
        
        return np.zeros_like(audio_chunk)
    
    def reset_buffer(self):
        self.buffer = np.array([])