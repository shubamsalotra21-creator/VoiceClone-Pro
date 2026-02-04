import torchaudio
import torch
import numpy as np
import librosa
import noisereduce as nr
from scipy import signal
from typing import Union, List, Dict, Any
import re

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
    
    def apply_eq_boost(self, audio: np.ndarray, low_freq: float = 60, high_freq: float = 150, 
                       gain_db: float = 4.0) -> np.ndarray:
        """Apply EQ boost to a specific frequency range for sub-bass weight.
        
        Args:
            audio: Input audio signal
            low_freq: Lower frequency bound in Hz (default: 60)
            high_freq: Upper frequency bound in Hz (default: 150)
            gain_db: Gain to apply in dB (default: 4.0)
            
        Returns:
            Audio with EQ boost applied
        """
        try:
            # Convert to frequency domain
            stft = librosa.stft(audio)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=self.target_sr)
            
            # Create EQ filter
            gain_linear = 10 ** (gain_db / 20)
            eq_filter = np.ones(len(freqs))
            
            # Apply gain to target frequency range
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            eq_filter[freq_mask] = gain_linear
            
            # Apply EQ to STFT
            boosted_stft = stft * eq_filter[:, np.newaxis]
            
            # Convert back to time domain
            boosted_audio = librosa.istft(boosted_stft)
            
            return boosted_audio
        except Exception as e:
            print(f"EQ boost failed: {e}")
            return audio
    
    def apply_compression(self, audio: np.ndarray, threshold: float = -20.0, 
                         ratio: float = 4.0, attack: float = 0.005, 
                         release: float = 0.1) -> np.ndarray:
        """Apply dynamic range compression for intimate 'whisper-to-roar' presence.
        
        Args:
            audio: Input audio signal
            threshold: Threshold in dB (default: -20.0)
            ratio: Compression ratio (default: 4.0 for 4:1)
            attack: Attack time in seconds (default: 0.005)
            release: Release time in seconds (default: 0.1)
            
        Returns:
            Compressed audio
        """
        try:
            # Convert threshold from dB to linear
            threshold_linear = 10 ** (threshold / 20)
            
            # Calculate envelope
            audio_abs = np.abs(audio)
            
            # Simple envelope follower
            envelope = np.zeros_like(audio_abs)
            attack_coef = np.exp(-1.0 / (attack * self.target_sr))
            release_coef = np.exp(-1.0 / (release * self.target_sr))
            
            for i in range(1, len(audio_abs)):
                if audio_abs[i] > envelope[i-1]:
                    envelope[i] = attack_coef * envelope[i-1] + (1 - attack_coef) * audio_abs[i]
                else:
                    envelope[i] = release_coef * envelope[i-1] + (1 - release_coef) * audio_abs[i]
            
            # Calculate gain reduction
            gain = np.ones_like(envelope)
            over_threshold = envelope > threshold_linear
            
            # Apply compression ratio
            gain[over_threshold] = (threshold_linear + (envelope[over_threshold] - threshold_linear) / ratio) / envelope[over_threshold]
            
            # Apply gain
            compressed_audio = audio * gain
            
            return compressed_audio
        except Exception as e:
            print(f"Compression failed: {e}")
            return audio
    
    def apply_reverb(self, audio: np.ndarray, room_size: float = 0.4, 
                    damping: float = 0.5, wet_level: float = 0.3) -> np.ndarray:
        """Apply reverb effect (Small Dark Room / Plate style).
        
        Args:
            audio: Input audio signal
            room_size: Reverb decay time in seconds (default: 0.4)
            damping: High frequency damping (default: 0.5)
            wet_level: Mix level of reverb (default: 0.3)
            
        Returns:
            Audio with reverb applied
        """
        try:
            # Create impulse response for reverb
            ir_length = int(room_size * self.target_sr)
            
            # Generate exponentially decaying noise as simple reverb IR
            decay = np.exp(-np.arange(ir_length) / (room_size * self.target_sr * 0.5))
            noise = np.random.randn(ir_length)
            impulse_response = noise * decay
            
            # Apply damping (low-pass filter)
            if damping > 0:
                cutoff = self.target_sr / 2 * (1 - damping * 0.8)
                nyquist = self.target_sr / 2
                normal_cutoff = cutoff / nyquist
                b, a = signal.butter(2, normal_cutoff, btype='low')
                impulse_response = signal.filtfilt(b, a, impulse_response)
            
            # Normalize impulse response
            impulse_response = impulse_response / np.max(np.abs(impulse_response))
            
            # Convolve audio with impulse response
            reverb_audio = signal.fftconvolve(audio, impulse_response, mode='same')
            
            # Mix dry and wet signals
            output_audio = (1 - wet_level) * audio + wet_level * reverb_audio
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(output_audio))
            if max_val > 1.0:
                output_audio = output_audio / max_val
            
            return output_audio
        except Exception as e:
            print(f"Reverb failed: {e}")
            return audio
    
    def apply_profile_post_processing(self, audio: np.ndarray, 
                                     post_processing_config: Dict[str, Any]) -> np.ndarray:
        """Apply post-processing based on voice profile configuration.
        
        Args:
            audio: Input audio signal
            post_processing_config: Dictionary containing post-processing parameters
            
        Returns:
            Audio with all post-processing effects applied
        """
        processed_audio = audio.copy()
        
        # Apply EQ boost if specified
        eq_boost = post_processing_config.get("eq_boost", "")
        if "60Hz-150Hz" in eq_boost or "60" in eq_boost:
            # Extract gain value if present (e.g., "+4dB")
            gain_db = 4.0  # default
            if "dB" in eq_boost:
                match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*dB', eq_boost)
                if match:
                    gain_db = float(match.group(1))
            
            processed_audio = self.apply_eq_boost(processed_audio, 60, 150, gain_db)
        
        # Apply compression if specified
        compression = post_processing_config.get("compression", "")
        if compression:
            # Extract ratio if present (e.g., "4:1 ratio")
            ratio = 4.0  # default
            if ":" in compression:
                match = re.search(r'(\d+(?:\.\d+)?):1', compression)
                if match:
                    ratio = float(match.group(1))
            
            processed_audio = self.apply_compression(processed_audio, ratio=ratio)
        
        # Apply reverb if specified
        reverb = post_processing_config.get("reverb", "")
        if reverb:
            # Extract decay time if present (e.g., "0.4s decay")
            decay_time = 0.4  # default
            if "s" in reverb:
                match = re.search(r'(\d+(?:\.\d+)?)\s*s', reverb)
                if match:
                    decay_time = float(match.group(1))
            
            # Adjust wet level based on room type
            wet_level = 0.3  # default
            if "Dark Room" in reverb or "Plate" in reverb:
                wet_level = 0.35  # slightly more wet for these types
            
            processed_audio = self.apply_reverb(processed_audio, room_size=decay_time, wet_level=wet_level)
        
        return processed_audio

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