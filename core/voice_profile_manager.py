import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

class VoiceProfileManager:
    """Manager for loading and managing voice profiles."""
    
    def __init__(self, profiles_dir: str = "voice_profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_cache = {}
        
    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Load a voice profile from JSON file.
        
        Args:
            profile_name: Name of the profile (without .json extension)
            
        Returns:
            Dictionary containing the voice profile configuration
        """
        if profile_name in self.profiles_cache:
            return self.profiles_cache[profile_name]
        
        profile_path = self.profiles_dir / f"{profile_name}.json"
        
        if not profile_path.exists():
            raise FileNotFoundError(f"Voice profile '{profile_name}' not found at {profile_path}")
        
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        
        self.profiles_cache[profile_name] = profile
        return profile
    
    def list_profiles(self) -> List[str]:
        """List all available voice profiles.
        
        Returns:
            List of profile names (without .json extension)
        """
        if not self.profiles_dir.exists():
            return []
        
        profiles = [
            f.stem for f in self.profiles_dir.glob("*.json")
        ]
        return sorted(profiles)
    
    def get_technical_parameters(self, profile_name: str) -> Dict[str, Any]:
        """Get technical engine parameters from a profile.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Dictionary containing technical parameters
        """
        profile = self.load_profile(profile_name)
        return profile.get("technical_engine_parameters", {})
    
    def get_performance_attributes(self, profile_name: str) -> Dict[str, Any]:
        """Get performance attributes from a profile.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Dictionary containing performance attributes
        """
        profile = self.load_profile(profile_name)
        return profile.get("performance_attributes", {})
    
    def get_voice_characteristics(self, profile_name: str) -> Dict[str, Any]:
        """Get voice profile characteristics.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Dictionary containing voice characteristics
        """
        profile = self.load_profile(profile_name)
        return profile.get("voice_profile", {})
    
    def get_masterpiece_script(self, profile_name: str) -> Optional[Dict[str, str]]:
        """Get the masterpiece script segment if available.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Dictionary containing script and directive, or None if not available
        """
        profile = self.load_profile(profile_name)
        return profile.get("masterpiece_script_segment")
    
    def save_profile(self, profile_name: str, profile_data: Dict[str, Any]):
        """Save a voice profile to JSON file.
        
        Args:
            profile_name: Name of the profile
            profile_data: Profile configuration dictionary
        """
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        profile_path = self.profiles_dir / f"{profile_name}.json"
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)
        
        self.profiles_cache[profile_name] = profile_data
    
    def get_synthesis_config(self, profile_name: str) -> Dict[str, Any]:
        """Get synthesis configuration from technical parameters.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Dictionary with synthesis configuration
        """
        tech_params = self.get_technical_parameters(profile_name)
        
        return {
            "stability": tech_params.get("stability", 0.5),
            "clarity_similarity": tech_params.get("clarity_similarity", 0.75),
            "style_exaggeration": tech_params.get("style_exaggeration", 0.0),
        }
    
    def get_post_processing_config(self, profile_name: str) -> Dict[str, Any]:
        """Get post-processing configuration from technical parameters.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Dictionary with post-processing configuration
        """
        tech_params = self.get_technical_parameters(profile_name)
        return tech_params.get("post_processing", {})
