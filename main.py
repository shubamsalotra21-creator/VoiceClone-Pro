import streamlit as st
import torch
import numpy as np
import soundfile as sf
import io
import os
from pathlib import Path
import tempfile
from core.voice_cloner import VoiceCloner
from core.emotion_controller import EmotionController
from core.audio_processor import AudioProcessor
from core.voice_profile_manager import VoiceProfileManager
from utils.audio_utils import save_audio, load_audio, play_audio
from utils.config import load_config

st.set_page_config(
    page_title="VoiceClone Pro - Advanced Voice Cloning - wasif",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    if 'voice_cloner' not in st.session_state:
        st.session_state.voice_cloner = None
    if 'emotion_controller' not in st.session_state:
        st.session_state.emotion_controller = None
    if 'audio_processor' not in st.session_state:
        st.session_state.audio_processor = None
    if 'voice_profile_manager' not in st.session_state:
        st.session_state.voice_profile_manager = VoiceProfileManager()
    if 'cloned_voice' not in st.session_state:
        st.session_state.cloned_voice = None
    if 'generated_audio' not in st.session_state:
        st.session_state.generated_audio = []
    if 'selected_profile' not in st.session_state:
        st.session_state.selected_profile = None

def load_models():
    with st.spinner("🔄 Loading AI models..."):
        if st.session_state.voice_cloner is None:
            st.session_state.voice_cloner = VoiceCloner()
        if st.session_state.emotion_controller is None:
            st.session_state.emotion_controller = EmotionController()
        if st.session_state.audio_processor is None:
            st.session_state.audio_processor = AudioProcessor()

def main():
    st.title("🎙️ VoiceClone Pro - Advanced Voice Cloning")
    st.markdown("Clone any voice with just 30 seconds of audio and control emotions in real-time!")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Voice Profile Selection
        st.subheader("🎭 Voice Profile")
        available_profiles = st.session_state.voice_profile_manager.list_profiles()
        
        if available_profiles:
            profile_options = ["None (Custom)"] + available_profiles
            selected_profile_name = st.selectbox(
                "Select Voice Profile",
                profile_options,
                help="Choose a pre-configured voice profile"
            )
            
            if selected_profile_name != "None (Custom)":
                st.session_state.selected_profile = selected_profile_name
                profile_data = st.session_state.voice_profile_manager.load_profile(selected_profile_name)
                
                # Display profile information
                with st.expander("📋 Profile Details"):
                    voice_info = profile_data.get("voice_profile", {})
                    st.write(f"**Character:** {voice_info.get('character_name', 'N/A')}")
                    st.write(f"**Gender:** {voice_info.get('gender', 'N/A')}")
                    st.write(f"**Accent:** {voice_info.get('accent', 'N/A')}")
                    st.write(f"**Archetype:** {voice_info.get('archetype', 'N/A')}")
                    
                    # Show masterpiece script if available
                    script = st.session_state.voice_profile_manager.get_masterpiece_script(selected_profile_name)
                    if script:
                        st.write("**Sample Script:**")
                        st.info(script.get('hindi', ''))
                        st.write(f"*Directive:* {script.get('performance_directive', '')}")
            else:
                st.session_state.selected_profile = None
        else:
            st.info("No voice profiles found. Using custom settings.")
            st.session_state.selected_profile = None
        
        st.divider()
        
        model_choice = st.selectbox(
            "Voice Model",
            ["Tacotron2 + WaveGlow", "FastSpeech2 + HiFiGAN", "VITS"],
            help="Select the voice synthesis model"
        )
        
        # Get default values from profile if selected
        if st.session_state.selected_profile:
            perf_attrs = st.session_state.voice_profile_manager.get_performance_attributes(st.session_state.selected_profile)
            # Map pacing to speed
            pacing = perf_attrs.get("pacing", "")
            if "slow" in pacing.lower():
                default_speed = 0.8
            elif "fast" in pacing.lower():
                default_speed = 1.2
            else:
                default_speed = 1.0
        else:
            default_speed = 1.0
        
        emotion_options = {
            "Neutral": "neutral",
            "Happy": "happy", 
            "Sad": "sad",
            "Angry": "angry",
            "Excited": "excited",
            "Calm": "calm"
        }
        selected_emotion = st.selectbox("Emotion", list(emotion_options.keys()))
        
        emotion_strength = st.slider("Emotion Strength", 0.1, 1.0, 0.7, 0.1)
        
        speed_factor = st.slider("Speech Speed", 0.5, 2.0, default_speed, 0.1)
        
        pitch_shift = st.slider("Pitch Adjustment", -5.0, 5.0, 0.0, 0.5)
        
        st.subheader("Voice Quality")
        enable_denoise = st.checkbox("Noise Removal", value=True)
        enable_normalize = st.checkbox("Volume Normalization", value=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎤 Record Voice", "📁 Upload Voice", "🗣️ Generate Speech", "📊 Voice Analytics"])
    
    with tab1:
        st.header("Record Reference Voice")
        st.markdown("Record at least 30 seconds of clean audio for best results")
        
        from streamlit_audio_recorder import audio_recorder
        
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            sample_rate=22050,
        )
        
        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                st.session_state.reference_audio = tmp_file.name
            
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("🎯 Clone Voice from Recording", type="primary"):
                clone_voice_from_audio(st.session_state.reference_audio)
    
    with tab2:
        st.header("Upload Reference Voice")
        st.markdown("Upload a clean audio file (WAV, MP3, FLAC)")
        
        uploaded_file = st.file_uploader(
            "Choose audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload 30+ seconds of clean speech"
        )
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.reference_audio = tmp_file.name
            
            st.audio(uploaded_file.getvalue(), format="audio/wav")
            
            if st.button("🎯 Clone Voice from Upload", type="primary"):
                clone_voice_from_audio(st.session_state.reference_audio)
    
    with tab3:
        st.header("Generate Speech")
        st.markdown("Enter text and generate speech with your cloned voice")
        
        if 'cloned_voice' not in st.session_state or st.session_state.cloned_voice is None:
            st.warning("Please clone a voice first using Record or Upload tabs")
        else:
            # Use profile script if available
            default_text = "Hello! This is your cloned voice speaking with advanced emotion control."
            if st.session_state.selected_profile:
                script = st.session_state.voice_profile_manager.get_masterpiece_script(
                    st.session_state.selected_profile
                )
                if script and 'hindi' in script:
                    default_text = script['hindi']
                    if 'performance_directive' in script:
                        st.info(f"📝 Performance Directive: {script['performance_directive']}")
            
            text_input = st.text_area(
                "Enter text to synthesize",
                default_text,
                height=100
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎵 Generate Speech", type="primary"):
                    generate_speech(
                        text_input,
                        emotion_options[selected_emotion],
                        emotion_strength,
                        speed_factor,
                        pitch_shift,
                        enable_denoise,
                        enable_normalize
                    )
            
            with col2:
                if st.button("🎭 Generate Emotional Variations"):
                    generate_emotional_variations(text_input)
    
    with tab4:
        st.header("Voice Analytics")
        st.markdown("Analyze and compare voice characteristics")
        
        if st.session_state.generated_audio:
            st.subheader("Generated Audio History")
            for idx, (audio_path, text, emotion) in enumerate(st.session_state.generated_audio[-5:]):
                with st.expander(f"Audio {idx+1}: {emotion}"):
                    st.write(f"Text: {text}")
                    st.audio(audio_path, format="audio/wav")
                    
                    if st.button(f"Download Audio {idx+1}", key=f"download_{idx}"):
                        with open(audio_path, "rb") as f:
                            st.download_button(
                                label="Download WAV",
                                data=f,
                                file_name=f"voiceclone_{emotion}_{idx+1}.wav",
                                mime="audio/wav"
                            )
        else:
            st.info("No audio generated yet")

def clone_voice_from_audio(audio_path):
    load_models()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Processing reference audio...")
        processed_audio = st.session_state.audio_processor.preprocess_audio(audio_path)
        progress_bar.progress(20)
        
        status_text.text("🎯 Extracting voice characteristics...")
        voice_embedding = st.session_state.voice_cloner.extract_voice_embedding(processed_audio)
        progress_bar.progress(50)
        
        status_text.text("🤖 Training voice model...")
        cloned_voice = st.session_state.voice_cloner.clone_voice(voice_embedding)
        st.session_state.cloned_voice = cloned_voice
        progress_bar.progress(80)
        
        status_text.text("✅ Voice cloning complete!")
        progress_bar.progress(100)
        
        st.success("🎉 Voice successfully cloned! You can now generate speech in the Generate Speech tab.")
        
    except Exception as e:
        st.error(f"❌ Voice cloning failed: {str(e)}")

def generate_speech(text, emotion, emotion_strength, speed, pitch, denoise, normalize):
    if st.session_state.cloned_voice is None:
        st.error("Please clone a voice first")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Get voice profile parameters if a profile is selected
        stability = 0.5
        clarity_similarity = 0.75
        style_exaggeration = 0.0
        post_processing_config = {}
        
        if st.session_state.selected_profile:
            synthesis_config = st.session_state.voice_profile_manager.get_synthesis_config(
                st.session_state.selected_profile
            )
            stability = synthesis_config.get("stability", 0.5)
            clarity_similarity = synthesis_config.get("clarity_similarity", 0.75)
            style_exaggeration = synthesis_config.get("style_exaggeration", 0.0)
            
            post_processing_config = st.session_state.voice_profile_manager.get_post_processing_config(
                st.session_state.selected_profile
            )
        
        status_text.text("🔄 Generating speech with profile parameters...")
        raw_audio = st.session_state.voice_cloner.synthesize_speech(
            text, 
            st.session_state.cloned_voice,
            stability=stability,
            clarity_similarity=clarity_similarity,
            style_exaggeration=style_exaggeration
        )
        progress_bar.progress(30)
        
        status_text.text("🎭 Applying emotion...")
        emotional_audio = st.session_state.emotion_controller.apply_emotion(
            raw_audio, 
            emotion, 
            emotion_strength
        )
        progress_bar.progress(50)
        
        status_text.text("⚙️ Processing audio...")
        processed_audio = emotional_audio
        
        if speed != 1.0:
            processed_audio = st.session_state.audio_processor.change_speed(processed_audio, speed)
        
        if pitch != 0.0:
            processed_audio = st.session_state.audio_processor.shift_pitch(processed_audio, pitch)
        
        if denoise:
            processed_audio = st.session_state.audio_processor.denoise(processed_audio, st.session_state.audio_processor.target_sr)
        
        if normalize:
            processed_audio = st.session_state.audio_processor.normalize_volume(processed_audio)
        
        progress_bar.progress(70)
        
        # Apply profile-specific post-processing if available
        if post_processing_config:
            status_text.text("🎚️ Applying profile post-processing (EQ, Compression, Reverb)...")
            processed_audio = st.session_state.audio_processor.apply_profile_post_processing(
                processed_audio,
                post_processing_config
            )
        
        progress_bar.progress(90)
        
        output_path = f"outputs/generated_{len(st.session_state.generated_audio)}.wav"
        save_audio(processed_audio, output_path, 22050)
        
        st.session_state.generated_audio.append((output_path, text, emotion))
        
        progress_bar.progress(100)
        status_text.text("✅ Speech generation complete!")
        
        st.audio(output_path, format="audio/wav")
        
        st.session_state.generated_audio.append((output_path, text, emotion))
        
        progress_bar.progress(100)
        status_text.text("✅ Speech generation complete!")
        
        st.audio(output_path, format="audio/wav")
        st.success("🎉 Speech generated successfully!")
        
    except Exception as e:
        st.error(f"❌ Speech generation failed: {str(e)}")

def generate_emotional_variations(text):
    emotions = ["neutral", "happy", "sad", "angry", "excited"]
    
    st.write("Generating emotional variations...")
    
    for emotion in emotions:
        with st.spinner(f"Generating {emotion} version..."):
            try:
                raw_audio = st.session_state.voice_cloner.synthesize_speech(
                    text, 
                    st.session_state.cloned_voice
                )
                
                emotional_audio = st.session_state.emotion_controller.apply_emotion(
                    raw_audio, 
                    emotion, 
                    0.8
                )
                
                output_path = f"outputs/variation_{emotion}.wav"
                save_audio(emotional_audio, output_path, 22050)
                
                st.write(f"**{emotion.capitalize()}:**")
                st.audio(output_path, format="audio/wav")
                
            except Exception as e:
                st.error(f"Failed to generate {emotion} version: {str(e)}")

if __name__ == "__main__":
    main()