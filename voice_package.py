import os
import requests
from dotenv import load_dotenv

class ElevenLabsVoice:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self.api_key = os.getenv('elevenlabs_api_key')
        self.base_url = "https://api.elevenlabs.io/v1"
        
        if not self.api_key:
            raise ValueError("ElevenLabs API key not found. Please set it in your .env file.")
    
    # eleven_monolingual_v1
    #eleven_multilingual_v2
    def generate_voice(self, text, voice_id="21m00Tcm4TlvDq8ikWAM", model_id="eleven_turbo_v2"):
        """
        Generate voice from text using ElevenLabs API
        
        Args:
            text (str): The text to convert to speech
            voice_id (str): The voice ID to use (default is a male voice)
            model_id (str): The model ID to use
            
        Returns:
            bytes: Audio data in bytes or None if generation failed
        """
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error generating voice: {e}")
            return None
    
    def save_audio(self, audio_data, output_path="output.mp3"):
        """
        Save audio data to a file
        
        Args:
            audio_data (bytes): The audio data to save
            output_path (str): The path to save the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not audio_data:
            return False
        
        try:
            with open(output_path, "wb") as f:
                f.write(audio_data)
            return True
        except Exception as e:
            print(f"Error saving audio file: {e}")
            return False

    

if __name__ == "__main__":
    voice = ElevenLabsVoice()
    audio_data = voice.generate_voice("Hello, how are you?")
    voice.save_audio(audio_data)

