# Voice Profiles

VoiceClone Pro now supports pre-configured **Voice Profiles** that allow you to apply sophisticated voice characteristics, performance attributes, and post-processing effects to synthesized speech.

## Overview

Voice Profiles are JSON configuration files that define:

- **Voice Profile Characteristics**: Character name, age, gender, ethnicity, accent, archetype, and vocal texture
- **Performance Attributes**: Pacing, pauses, breathing, emotional range, and language nuance  
- **Technical Engine Parameters**: Stability, clarity/similarity, style exaggeration
- **Post-Processing Configuration**: EQ boost, compression, reverb effects
- **Sample Scripts**: Masterpiece script segments with performance directives

## Available Profiles

### The Delhi Emperor

A commanding, cinematic voice profile with deep bass resonance and authentic Delhi accent.

**Character Details:**
- Age: 30-40 years
- Gender: Male
- Ethnicity: Indian
- Accent: Authentic Delhi (Urban/Strong)
- Archetype: Alpha / Anti-Hero / Commanding Leader

**Vocal Characteristics:**
- **Depth**: Extreme bass / Sub-frequency resonance (85Hz-110Hz)
- **Timbre**: Rich, husky, heavy chest-weighted
- **Weight**: Cinematic (Thanos-level authority)
- **Finish**: Slightly gritty, warm, intimidatingly attractive

**Performance Style:**
- **Pacing**: Slow-moderate cinematic
- **Pauses**: Strategic, dramatic, weighted
- **Breathing**: Audible, controlled, close-mic proximity
- **Emotional Range**: 100% (Rage, dominance, vulnerability, sarcasm, triumph)
- **Language**: Pure Hindi with natural Delhi rhythm and colloquial 'Dilli' bite

**Technical Parameters:**
- **Stability**: 0.45 (more variable/expressive)
- **Clarity/Similarity**: 0.90 (high fidelity to original)
- **Style Exaggeration**: 0.15 (subtle enhancement)

**Post-Processing:**
- **EQ Boost**: 60Hz-150Hz (+4dB) for sub-bass weight
- **Compression**: 4:1 ratio for intimate 'whisper-to-roar' presence
- **Reverb**: Small Dark Room / Plate (0.4s decay)

**Sample Script:**
```
Dilli ki hawa mein na... ek ajeeb sa bhari-pan hai. Log kehte hain ye dhuaan hai. 
Main kehta hoon... ye un logon ki cheekhein hain jo mere raaste mein aaye thhe. 
Beta... raj-paat virasat mein milta hoga, lekin dabdaba... dabdaba kamana padta hai.
```

**Performance Directive:**
*Start with a deep intake of breath. Speak close to the mic. Treat words like 'DABDABA' with heavy chest resonance. End with a slow, commanding finality.*

## Using Voice Profiles

### In the Web Interface

1. **Select Profile**: In the sidebar under "Voice Profile", choose from available profiles
2. **View Details**: Click the expander to see full profile information and sample script
3. **Clone Voice**: Upload or record reference audio to create a voice model
4. **Generate Speech**: The profile's parameters are automatically applied during synthesis
   - Technical parameters (stability, clarity, style) affect the base synthesis
   - Post-processing effects (EQ, compression, reverb) are applied to the final audio
   - Sample script is auto-populated in the text area

### Programmatic Usage

```python
from core.voice_profile_manager import VoiceProfileManager
from core.voice_cloner import VoiceCloner
from core.audio_processor import AudioProcessor

# Initialize components
profile_manager = VoiceProfileManager()
voice_cloner = VoiceCloner()
audio_processor = AudioProcessor()

# Load profile
profile = profile_manager.load_profile('delhi_emperor')

# Get synthesis parameters
synthesis_config = profile_manager.get_synthesis_config('delhi_emperor')
stability = synthesis_config['stability']
clarity = synthesis_config['clarity_similarity']
style = synthesis_config['style_exaggeration']

# Synthesize with profile parameters
audio = voice_cloner.synthesize_speech(
    text="Your text here",
    voice_model=your_cloned_voice,
    stability=stability,
    clarity_similarity=clarity,
    style_exaggeration=style
)

# Apply post-processing
post_config = profile_manager.get_post_processing_config('delhi_emperor')
processed_audio = audio_processor.apply_profile_post_processing(
    audio,
    post_config
)
```

## Creating Custom Profiles

Create a new JSON file in the `voice_profiles/` directory with the following structure:

```json
{
  "voice_profile": {
    "character_name": "Your Character",
    "age_range": "25-35 years",
    "gender": "Male/Female/Other",
    "ethnicity": "Your Ethnicity",
    "accent": "Your Accent",
    "archetype": "Your Archetype",
    "vocal_texture": {
      "depth": "Bass/Tenor/Alto/Soprano",
      "timbre": "Description",
      "weight": "Light/Medium/Heavy",
      "finish": "Description"
    }
  },
  "performance_attributes": {
    "pacing": "Slow/Moderate/Fast",
    "pauses": "Description",
    "breathing": "Description",
    "emotional_range": "Description",
    "language_nuance": "Description"
  },
  "technical_engine_parameters": {
    "stability": 0.5,
    "clarity_similarity": 0.75,
    "style_exaggeration": 0.0,
    "post_processing": {
      "eq_boost": "60Hz-150Hz (+4dB) for sub-bass weight",
      "compression": "4:1 ratio for description",
      "reverb": "Room Type (decay time)"
    }
  },
  "masterpiece_script_segment": {
    "hindi": "Your sample text",
    "performance_directive": "Performance instructions"
  }
}
```

### Parameter Guidelines

**Stability** (0.0 - 1.0):
- Lower values (0.3-0.5): More variable and expressive
- Medium values (0.5-0.7): Balanced
- Higher values (0.7-0.9): More consistent and stable

**Clarity/Similarity** (0.0 - 1.0):
- Higher values (0.8-0.95): Closer to reference voice
- Medium values (0.6-0.8): Balanced quality
- Lower values (0.4-0.6): More variation from reference

**Style Exaggeration** (0.0 - 1.0):
- 0.0: No exaggeration (neutral)
- 0.1-0.2: Subtle enhancement
- 0.3-0.5: Moderate enhancement
- 0.5+: Strong stylistic emphasis

**Post-Processing:**
- **EQ Boost**: Specify frequency range and gain (e.g., "60Hz-150Hz (+4dB)")
- **Compression**: Specify ratio (e.g., "4:1 ratio")
- **Reverb**: Specify room type and decay time (e.g., "Small Room (0.3s decay)")

## Voice Profile Architecture

```
voice_profiles/
├── delhi_emperor.json          # Delhi Emperor profile
└── your_profile.json            # Your custom profiles

core/
└── voice_profile_manager.py     # Profile management system

VoiceProfileManager Methods:
├── load_profile()               # Load a profile by name
├── list_profiles()              # List all available profiles
├── get_technical_parameters()   # Get synthesis parameters
├── get_post_processing_config() # Get post-processing settings
├── get_voice_characteristics()  # Get voice profile info
├── get_masterpiece_script()     # Get sample script
└── save_profile()               # Save a new profile
```

## Technical Implementation

The voice profile system integrates with three core components:

1. **VoiceCloner**: Uses technical parameters (stability, clarity, style) during speech synthesis
2. **EmotionController**: Applies emotional characteristics based on performance attributes
3. **AudioProcessor**: Applies post-processing effects (EQ, compression, reverb) to final audio

### Audio Post-Processing Details

**EQ Boost (Parametric Equalizer):**
- Applies frequency-selective gain to specific ranges
- Uses FFT-based filtering for precise control
- Ideal for enhancing bass presence or reducing harshness

**Dynamic Range Compression:**
- Reduces dynamic range for consistent loudness
- Attack/release envelope following
- Maintains whisper-to-roar presence

**Reverb (Convolution):**
- Simulates acoustic spaces
- Exponentially decaying impulse response
- Configurable room size and damping

## Best Practices

1. **Reference Audio**: Use high-quality, clean audio when cloning voices for profile-based synthesis
2. **Profile Selection**: Choose profiles that match the desired vocal characteristics
3. **Parameter Tuning**: Start with profile defaults, then fine-tune using UI sliders
4. **Post-Processing Order**: EQ → Compression → Reverb (automatically applied in correct order)
5. **Language Compatibility**: Ensure your TTS model supports the profile's target language

## Future Enhancements

- Additional pre-configured voice profiles
- Profile inheritance and composition
- Real-time parameter adjustment
- A/B testing for profile comparison
- Profile-specific emotion mappings
- Multi-language profile support

## Support

For issues or questions about voice profiles, please refer to the main README or open an issue on the GitHub repository.
