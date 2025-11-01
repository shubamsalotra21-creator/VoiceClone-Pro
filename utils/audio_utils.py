import soundfile as sf
import numpy as np
import io
from pydub import AudioSegment
from pydub.generators import Sine
import tempfile
import os

def save_audio(audio: np.ndarray, filepath: str, sr: int = 22050):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sf.write(filepath, audio, sr)

def load_audio(filepath: str, sr: int = 22050) -> np.ndarray:
    audio, _ = librosa.load(filepath, sr=sr)
    return audio

def play_audio(audio: np.ndarray, sr: int = 22050):
    import IPython.display as ipd
    return ipd.Audio(audio, rate=sr)

def convert_audio_format(input_path: str, output_path: str, format: str = "wav"):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format=format)

def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)

def audio_to_bytes(audio: np.ndarray, sr: int = 22050) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)
    return buffer.getvalue()

def bytes_to_audio(audio_bytes: bytes, sr: int = 22050) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file.flush()
        audio, _ = librosa.load(tmp_file.name, sr=sr)
        os.unlink(tmp_file.name)
    return audio

def generate_sine_wave(frequency: float, duration: float, sr: int = 22050) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio

def mix_audios(audio1: np.ndarray, audio2: np.ndarray, mix_ratio: float = 0.5) -> np.ndarray:
    min_len = min(len(audio1), len(audio2))
    mixed = mix_ratio * audio1[:min_len] + (1 - mix_ratio) * audio2[:min_len]
    return mixed

def calculate_rms(audio: np.ndarray) -> float:
    return np.sqrt(np.mean(audio**2))

def normalize_audio(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    current_rms = calculate_rms(audio)
    if current_rms > 0:
        gain = target_rms / current_rms
        return np.clip(audio * gain, -1.0, 1.0)
    return audio

def trim_silence(audio: np.ndarray, sr: int, silence_threshold: float = 0.01) -> np.ndarray:
    intervals = librosa.effects.split(audio, top_db=30)
    if len(intervals) > 0:
        return np.concatenate([audio[start:end] for start, end in intervals])
    return audio

def create_audio_chunks(audio: np.ndarray, chunk_duration: float, sr: int = 22050):
    chunk_size = int(chunk_duration * sr)
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
    return chunks