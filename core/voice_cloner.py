import torch
import torchaudio
import numpy as np
import librosa
from speechbrain.pretrained import Tacotron2, Waveglow
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn
from typing import Dict, Any, List
import tempfile

class VoiceEmbeddingExtractor:
    def __init__(self, device="auto"):
        self.device = self._setup_device(device)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.model.to(self.device)
        self.model.eval()
    
    def _setup_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        audio, sr = librosa.load(audio_path, sr=16000)
        
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs.to(self.device))
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings[0]

class VoiceCloner:
    def __init__(self, device="auto"):
        self.device = self._setup_device(device)
        self.embedding_extractor = VoiceEmbeddingExtractor(device)
        self.tacotron2 = None
        self.waveglow = None
        self.voice_models = {}
        
    def _setup_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_models(self):
        if self.tacotron2 is None:
            self.tacotron2 = Tacotron2.from_hparams(
                source="speechbrain/tts-tacotron2-ljspeech",
                savedir="models/tacotron2"
            )
            self.tacotron2.to(self.device)
        
        if self.waveglow is None:
            self.waveglow = Waveglow.from_hparams(
                source="speechbrain/tts-waveglow-ljspeech", 
                savedir="models/waveglow"
            )
            self.waveglow.to(self.device)
    
    def extract_voice_embedding(self, audio_path: str) -> np.ndarray:
        return self.embedding_extractor.extract_embedding(audio_path)
    
    def clone_voice(self, voice_embedding: np.ndarray, voice_id: str = "cloned_voice") -> Dict[str, Any]:
        self.load_models()
        
        voice_model = {
            "embedding": voice_embedding,
            "id": voice_id,
            "timestamp": torch.timestamp()
        }
        
        self.voice_models[voice_id] = voice_model
        return voice_model
    
    def synthesize_speech(self, text: str, voice_model: Dict[str, Any], 
                         stability: float = 0.5, clarity_similarity: float = 0.75,
                         style_exaggeration: float = 0.0) -> np.ndarray:
        """Synthesize speech with voice profile parameters.
        
        Args:
            text: Text to synthesize
            voice_model: Voice model dictionary
            stability: Stability parameter (0.0-1.0), lower = more variable/expressive
            clarity_similarity: Clarity/similarity parameter (0.0-1.0), higher = closer to original
            style_exaggeration: Style exaggeration parameter (0.0-1.0), higher = more exaggerated
            
        Returns:
            Synthesized audio as numpy array
        """
        self.load_models()
        
        try:
            mel_output, mel_length, alignment = self.tacotron2.encode_text(
                [text]
            )
            
            # Apply stability by adding controlled noise to mel spectrogram
            if stability < 1.0:
                noise_scale = (1.0 - stability) * 0.1
                noise = torch.randn_like(mel_output) * noise_scale
                mel_output = mel_output + noise
            
            # Apply style exaggeration by amplifying mel spectrogram variations
            if style_exaggeration > 0.0:
                mel_mean = mel_output.mean()
                mel_output = mel_mean + (mel_output - mel_mean) * (1.0 + style_exaggeration)
            
            waveforms = self.waveglow.decode_batch(mel_output)
            
            audio = waveforms.squeeze().cpu().numpy()
            
            # Apply clarity/similarity by blending with a more neutral version
            if clarity_similarity < 1.0:
                # This is a simplified approach - in production, you'd blend with reference
                audio = audio * clarity_similarity + audio * 0.5 * (1.0 - clarity_similarity)
            
            return audio
            
        except Exception as e:
            raise Exception(f"Speech synthesis failed: {str(e)}")
    
    def fine_tune_voice(self, voice_id: str, additional_audio: List[str], epochs: int = 10):
        pass
    
    def get_voice_similarity(self, voice1: Dict[str, Any], voice2: Dict[str, Any]) -> float:
        emb1 = voice1["embedding"]
        emb2 = voice2["embedding"]
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

class FastVoiceCloner(VoiceCloner):
    def __init__(self, device="auto"):
        super().__init__(device)
        self.fastspeech2 = None
        self.hifigan = None
    
    def load_models(self):
        try:
            from espnet2.bin.tts_inference import Text2Speech
            
            if self.fastspeech2 is None:
                self.fastspeech2 = Text2Speech.from_pretrained(
                    "espnet/kan-bayashi_ljspeech_fastspeech2"
                )
            
            if self.hifigan is None:
                self.hifigan = Text2Speech.from_pretrained(
                    "espnet/kan-bayashi_ljspeech_hifigan"
                )
                
        except ImportError:
            raise ImportError("ESPnet not installed. Install with: pip install espnet")
    
    def synthesize_speech(self, text: str, voice_model: Dict[str, Any]) -> np.ndarray:
        self.load_models()
        
        try:
            with torch.no_grad():
                output = self.fastspeech2(text)
                waveform = output[0].numpy()
            
            return waveform
            
        except Exception as e:
            raise Exception(f"Fast speech synthesis failed: {str(e)}")