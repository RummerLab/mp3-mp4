#!/usr/bin/env python3
"""
MP3 to MP4 Converter for Social Media
Converts MP3 files to MP4 videos with logos and auto-generated captions
Optimized for portrait format (reels/shorts)
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from typing import List, Optional
import subprocess
import json
import re
from datetime import datetime
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Continue without dotenv if not available

# Video processing libraries
try:
    import cv2
    import numpy as np
    from moviepy.editor import *
    from moviepy.video.fx import resize
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install required packages: pip install opencv-python moviepy pillow numpy")
    sys.exit(1)

# Audio processing libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: Librosa not available. Audio visualization will use basic analysis.")

# Try to import whisper (optional for captions)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available. Videos will be created without captions.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MP3ToMP4Converter:
    def __init__(self, input_folder: str = "input", output_folder: str = "output"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        
        # Load configuration
        self.config = self.load_config()
        
        # Create directories if they don't exist
        self.input_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        
        # Video settings from config
        self.video_width = self.config["video_settings"]["width"]
        self.video_height = self.config["video_settings"]["height"]
        self.fps = self.config["video_settings"]["fps"]
        
        # Check if captions are enabled
        self.enable_captions = os.getenv("ENABLE_CAPTIONS", "true").lower() == "true"
        self.enable_logos = os.getenv("ENABLE_LOGOS", "true").lower() == "true"
        
        # Initialize Whisper model for transcription
        if WHISPER_AVAILABLE and self.enable_captions:
            try:
                model_name = self.config["whisper"]["model"]
                self.whisper_model = whisper.load_model(model_name)
                logger.info(f"Whisper model '{model_name}' loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load Whisper model: {e}")
                self.whisper_model = None
        else:
            self.whisper_model = None
            if not self.enable_captions:
                logger.info("Captions disabled via environment variable")
            else:
                logger.info("Whisper not available - videos will be created without captions")
    
    def load_config(self) -> dict:
        """Load configuration from config.json and environment variables"""
        config_path = Path("config.json")
        config = {}
        
        # Load from config.json if it exists
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config.json: {e}")
        
        # Override with environment variables
        config = self._override_with_env(config)
        
        return config
    
    def _override_with_env(self, config: dict) -> dict:
        """Override configuration with environment variables"""
        # Video settings
        if "video_settings" not in config:
            config["video_settings"] = {}
        
        config["video_settings"]["width"] = int(os.getenv("VIDEO_WIDTH", config.get("video_settings", {}).get("width", 480)))
        config["video_settings"]["height"] = int(os.getenv("VIDEO_HEIGHT", config.get("video_settings", {}).get("height", 854)))
        config["video_settings"]["fps"] = int(os.getenv("VIDEO_FPS", config.get("video_settings", {}).get("fps", 30)))
        
        # Background colors
        if "background" not in config:
            config["background"] = {"type": "gradient", "colors": {}}
        if "colors" not in config["background"]:
            config["background"]["colors"] = {}
        
        # Check for transparent background setting
        config["transparent_background"] = os.getenv("TRANSPARENT_BACKGROUND", "false").lower() == "true"
        
        config["background"]["colors"]["top"] = [
            int(os.getenv("BG_TOP_RED", config.get("background", {}).get("colors", {}).get("top", [100, 50, 20])[0])),
            int(os.getenv("BG_TOP_GREEN", config.get("background", {}).get("colors", {}).get("top", [100, 50, 20])[1])),
            int(os.getenv("BG_TOP_BLUE", config.get("background", {}).get("colors", {}).get("top", [100, 50, 20])[2]))
        ]
        config["background"]["colors"]["bottom"] = [
            int(os.getenv("BG_BOTTOM_RED", config.get("background", {}).get("colors", {}).get("bottom", [200, 100, 50])[0])),
            int(os.getenv("BG_BOTTOM_GREEN", config.get("background", {}).get("colors", {}).get("bottom", [200, 100, 50])[1])),
            int(os.getenv("BG_BOTTOM_BLUE", config.get("background", {}).get("colors", {}).get("bottom", [200, 100, 50])[2]))
        ]
        
        # Logo URLs
        if "logos" not in config:
            config["logos"] = {}
        if "rummerlab" not in config["logos"]:
            config["logos"]["rummerlab"] = {}
        if "physioshark" not in config["logos"]:
            config["logos"]["physioshark"] = {}
        
        config["logos"]["rummerlab"]["url"] = os.getenv("RUMMERLAB_LOGO_URL", 
            config.get("logos", {}).get("rummerlab", {}).get("url", "https://rummerlab.com/images/rummerlab_logo_transparent.png"))
        config["logos"]["physioshark"]["url"] = os.getenv("PHYSIOSHARK_LOGO_URL", 
            config.get("logos", {}).get("physioshark", {}).get("url", "https://physioshark.org/images/logo-physioshark-project.png"))
        
        # Caption settings
        if "captions" not in config:
            config["captions"] = {}
        
        config["captions"]["font_size"] = int(os.getenv("CAPTION_FONT_SIZE", config.get("captions", {}).get("font_size", 60)))
        config["captions"]["font_color"] = os.getenv("CAPTION_FONT_COLOR", config.get("captions", {}).get("font_color", "white"))
        config["captions"]["stroke_color"] = os.getenv("CAPTION_STROKE_COLOR", config.get("captions", {}).get("stroke_color", "black"))
        config["captions"]["stroke_width"] = int(os.getenv("CAPTION_STROKE_WIDTH", config.get("captions", {}).get("stroke_width", 3)))
        
        # Whisper settings
        if "whisper" not in config:
            config["whisper"] = {}
        
        config["whisper"]["model"] = os.getenv("WHISPER_MODEL", config.get("whisper", {}).get("model", "base"))
        config["whisper"]["language"] = os.getenv("WHISPER_LANGUAGE", config.get("whisper", {}).get("language"))
        
        # Set default values for missing keys
        if "position" not in config["captions"]:
            config["captions"]["position"] = "bottom"
        if "max_width" not in config["captions"]:
            config["captions"]["max_width"] = 980
        if "words_per_segment" not in config["captions"]:
            config["captions"]["words_per_segment"] = 5
        
        if "position" not in config["logos"]["rummerlab"]:
            config["logos"]["rummerlab"]["position"] = "top-left"
        if "max_height" not in config["logos"]["rummerlab"]:
            config["logos"]["rummerlab"]["max_height"] = 200
        if "margin" not in config["logos"]["rummerlab"]:
            config["logos"]["rummerlab"]["margin"] = 50
        
        if "position" not in config["logos"]["physioshark"]:
            config["logos"]["physioshark"]["position"] = "top-right"
        if "max_height" not in config["logos"]["physioshark"]:
            config["logos"]["physioshark"]["max_height"] = 200
        if "margin" not in config["logos"]["physioshark"]:
            config["logos"]["physioshark"]["margin"] = 50
        
        if "codec" not in config["video_settings"]:
            config["video_settings"]["codec"] = "libx264"
        if "audio_codec" not in config["video_settings"]:
            config["video_settings"]["audio_codec"] = "aac"
        
        return config
    
    def download_logos(self) -> dict:
        """Download logos from URLs and return paths to local files"""
        if not self.enable_logos:
            logger.info("Logos disabled via environment variable")
            return {}
        
        logo_paths = {}
        
        for name, logo_config in self.config["logos"].items():
            # Download logos to root directory instead of output folder
            logo_path = Path(f"{name}_logo.png")
            
            if not logo_path.exists():
                try:
                    logger.info(f"Downloading {name} logo...")
                    response = requests.get(logo_config["url"], timeout=30)
                    response.raise_for_status()
                    
                    with open(logo_path, 'wb') as f:
                        f.write(response.content)
                    
                    logger.info(f"Downloaded {name} logo successfully to {logo_path}")
                except Exception as e:
                    logger.error(f"Failed to download {name} logo: {e}")
                    continue
            
            logo_paths[name] = logo_path
        
        return logo_paths
    
    def transcribe_audio(self, audio_path: Path) -> str:
        """Transcribe audio using Whisper (local or API)"""
        if not self.enable_captions:
            logger.info("Captions disabled, skipping transcription")
            return ""
        
        # Check for OpenAI API key first
        openai_api_key = os.getenv("OPENAI_API_KEY")
        whisper_api_key = os.getenv("WHISPER_API_KEY")
        
        if openai_api_key or whisper_api_key:
            return self._transcribe_with_api(audio_path, openai_api_key or whisper_api_key)
        
        # Fall back to local Whisper model
        if not self.whisper_model:
            logger.warning("Whisper model not available, skipping transcription")
            return ""
        
        try:
            logger.info(f"Transcribing {audio_path.name} with local Whisper...")
            # Use language from config if specified
            language = self.config["whisper"]["language"]
            result = self.whisper_model.transcribe(str(audio_path), language=language)
            return result["text"]
        except Exception as e:
            logger.error(f"Local transcription failed: {e}")
            return ""
    
    def _transcribe_with_api(self, audio_path: Path, api_key: str) -> str:
        """Transcribe audio using OpenAI API"""
        try:
            logger.info(f"Transcribing {audio_path.name} with OpenAI API...")
            
            # Prepare the API request
            api_url = os.getenv("WHISPER_API_URL", "https://api.openai.com/v1/audio/transcriptions")
            
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
            with open(audio_path, "rb") as audio_file:
                files = {
                    "file": (audio_path.name, audio_file, "audio/mpeg"),
                    "model": (None, "whisper-1"),
                    "language": (None, self.config["whisper"]["language"] or "en")
                }
                
                response = requests.post(api_url, headers=headers, files=files, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                return result.get("text", "")
                
        except Exception as e:
            logger.error(f"API transcription failed: {e}")
            return ""
    
    def create_captions(self, text: str, duration: float) -> List[dict]:
        """Create caption segments from transcribed text"""
        if not text:
            return []
        
        # Simple word-based segmentation
        words = text.split()
        words_per_segment = self.config["captions"]["words_per_segment"]
        segments = []
        
        for i in range(0, len(words), words_per_segment):
            segment_words = words[i:i + words_per_segment]
            segment_text = " ".join(segment_words)
            
            start_time = (i / len(words)) * duration
            end_time = min(((i + words_per_segment) / len(words)) * duration, duration)
            
            segments.append({
                "text": segment_text,
                "start": start_time,
                "end": end_time
            })
        
        return segments
    
    def create_caption_clip(self, text: str, start_time: float, end_time: float) -> TextClip:
        """Create a text clip for captions"""
        try:
            caption_config = self.config["captions"]
            
            # Create caption with styling from config
            txt_clip = TextClip(
                text,
                fontsize=caption_config["font_size"],
                color=caption_config["font_color"],
                font='Arial-Bold',
                stroke_color=caption_config["stroke_color"],
                stroke_width=caption_config["stroke_width"],
                method='caption',
                size=(caption_config["max_width"], None)
            ).set_position(('center', caption_config["position"])).set_duration(end_time - start_time).set_start(start_time)
            
            return txt_clip
        except Exception as e:
            logger.error(f"Failed to create caption clip: {e}")
            return None
    
    def create_audio_visualization(self, audio: AudioFileClip, duration: float) -> VideoClip:
        """Create advanced audio visualization using STFT frequency analysis"""
        viz_config = self.config.get("audio_visualization", {})
        
        if not viz_config.get("enabled", True):
            return None
        
        # Parameters from config
        num_bars = viz_config.get("num_bars", 60)
        bar_width = viz_config.get("bar_width", 6)
        bar_spacing = viz_config.get("bar_spacing", 3)
        bar_color = viz_config.get("bar_color", [100, 200, 255])
        bar_alpha = viz_config.get("bar_alpha", 0.8)
        viz_height = viz_config.get("height", 120)
        sensitivity = viz_config.get("sensitivity", 2.0)
        
        # Enhanced visualization features
        gradient_enabled = viz_config.get("gradient", {}).get("enabled", True)
        glow_enabled = viz_config.get("glow", {}).get("enabled", True)
        background_enabled = viz_config.get("background", {}).get("enabled", True)
        
        # Background settings
        bg_height = viz_config.get("background", {}).get("height", 140)
        bg_color = viz_config.get("background", {}).get("color", [0, 0, 0, 150])
        bg_blur = viz_config.get("background", {}).get("blur", 10)
        
        # Advanced audio analysis using STFT
        frequency_data = None
        time_data = None
        spectrogram = None
        
        if LIBROSA_AVAILABLE:
            try:
                # Save audio to temporary file for librosa processing
                temp_audio_path = "temp_audio_for_analysis.wav"
                audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                
                # Load audio with librosa
                y, sr = librosa.load(temp_audio_path, sr=None)
                
                # Advanced STFT analysis (as described in the article)
                hop_length = 512
                n_fft = 2048 * 4  # 4x larger for better accuracy
                
                # Get STFT matrix (frequency vs time)
                stft = np.abs(librosa.stft(y, hop_length=hop_length, n_fft=n_fft))
                
                # Convert to decibel scale
                spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
                
                # Get frequency and time arrays
                frequencies = librosa.core.fft_frequencies(n_fft=n_fft)
                times = librosa.core.frames_to_time(np.arange(spectrogram.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)
                
                # Calculate ratios for index mapping
                time_index_ratio = len(times) / times[-1] if len(times) > 0 else 1
                frequencies_index_ratio = len(frequencies) / frequencies[-1] if len(frequencies) > 0 else 1
                
                # Store for later use
                frequency_data = {
                    'frequencies': frequencies,
                    'frequencies_index_ratio': frequencies_index_ratio
                }
                time_data = {
                    'times': times,
                    'time_index_ratio': time_index_ratio
                }
                
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                
                logger.info("Successfully analyzed audio with advanced STFT")
                
            except Exception as e:
                logger.warning(f"Could not analyze audio with STFT: {e}")
                frequency_data = None
                time_data = None
                spectrogram = None
        
        # Fallback to basic analysis if STFT fails
        if spectrogram is None:
            try:
                # Extract audio array at 44100 Hz for analysis
                audio_array = audio.to_soundarray(fps=44100)
                
                # Handle stereo to mono conversion
                if len(audio_array.shape) > 1:
                    audio_array = np.mean(audio_array, axis=1)
                
                audio_array = np.asarray(audio_array).flatten()
                
                # Calculate audio levels for each time segment
                samples_per_frame = int(44100 / self.fps)
                num_frames = int(duration * self.fps)
                
                audio_levels = []
                for frame_idx in range(num_frames):
                    start_sample = frame_idx * samples_per_frame
                    end_sample = min(start_sample + samples_per_frame, len(audio_array))
                    
                    if end_sample > start_sample:
                        frame_audio = audio_array[start_sample:end_sample]
                        rms = np.sqrt(np.mean(frame_audio**2))
                        audio_levels.append(rms)
                    else:
                        audio_levels.append(0.0)
                
                # Normalize audio levels
                max_level = max(audio_levels) if audio_levels else 1.0
                if max_level > 0:
                    audio_levels = [level / max_level for level in audio_levels]
                
            except Exception as e:
                logger.warning(f"Could not analyze audio for visualization: {e}")
                audio_levels = None
        
        # Calculate total width needed for bars
        total_bar_width = num_bars * (bar_width + bar_spacing) - bar_spacing
        start_x = (self.video_width - total_bar_width) // 2
        
        def get_decibel(target_time, freq):
            """Get decibel value for specific time and frequency (from article)"""
            if spectrogram is not None and frequency_data and time_data:
                try:
                    freq_idx = int(freq * frequency_data['frequencies_index_ratio'])
                    time_idx = int(target_time * time_data['time_index_ratio'])
                    
                    # Ensure indices are within bounds
                    freq_idx = max(0, min(freq_idx, spectrogram.shape[0] - 1))
                    time_idx = max(0, min(time_idx, spectrogram.shape[1] - 1))
                    
                    return spectrogram[freq_idx][time_idx]
                except:
                    return -80  # Default low value
            return -80
        
        def make_viz_frame(t):
            # Create frame with background if enabled
            if background_enabled:
                # Create larger frame to accommodate background
                frame = np.zeros((bg_height, self.video_width, 3), dtype=np.uint8)
                
                # Create background bar
                bg_y_start = bg_height - viz_height - 20
                bg_y_end = bg_height - 10
                
                # Draw semi-transparent background
                for y in range(bg_y_start, bg_y_end):
                    for x in range(self.video_width):
                        # Create gradient background
                        bg_ratio = (y - bg_y_start) / (bg_y_end - bg_y_start)
                        bg_r = int(bg_color[0] * bg_ratio)
                        bg_g = int(bg_color[1] * bg_ratio)
                        bg_b = int(bg_color[2] * bg_ratio)
                        frame[y, x] = [bg_b, bg_g, bg_r]  # BGR order
                
                # Set visualization area
                viz_y_offset = bg_y_start
            else:
                frame = np.zeros((viz_height, self.video_width, 3), dtype=np.uint8)
                viz_y_offset = 0
            
            # Create frequency-based bars
            for i in range(num_bars):
                # Map bar index to frequency range (100Hz to 8000Hz)
                freq = 100 + (i / num_bars) * 7900
                
                # Get decibel value for this frequency and time
                decibel = get_decibel(t, freq)
                
                # Convert decibel to height
                min_decibel = -80
                max_decibel = 0
                min_height = 3
                max_height = viz_height * 0.9
                
                # Calculate height based on decibel
                if decibel > min_decibel:
                    decibel_height_ratio = (max_height - min_height) / (max_decibel - min_decibel)
                    desired_height = decibel * decibel_height_ratio + max_height
                    wave_height = int(max(min_height, min(max_height, desired_height)))
                else:
                    wave_height = min_height
                
                # Apply sensitivity
                wave_height = int(wave_height * sensitivity)
                
                if wave_height > 0:
                    x = start_x + i * (bar_width + bar_spacing)
                    
                    # Draw bar with enhanced effects
                    for y in range(max(0, wave_height)):
                        for x_offset in range(bar_width):
                            if x + x_offset < self.video_width and y < viz_height:
                                # Create gradient effect - brighter at the top
                                gradient_factor = 1.0 - (y / max(wave_height, 1))
                                
                                # Enhanced color calculation
                                if gradient_enabled:
                                    top_color = viz_config.get("gradient", {}).get("top_color", [150, 220, 255])
                                    bottom_color = viz_config.get("gradient", {}).get("bottom_color", [80, 160, 220])
                                    
                                    # Interpolate between top and bottom colors
                                    r = int(top_color[0] + gradient_factor * (bottom_color[0] - top_color[0]))
                                    g = int(top_color[1] + gradient_factor * (bottom_color[1] - top_color[1]))
                                    b = int(top_color[2] + gradient_factor * (bottom_color[2] - top_color[2]))
                                else:
                                    # Use single color with intensity variation
                                    color_intensity = bar_alpha * (0.6 + 0.4 * gradient_factor)
                                    r = int(bar_color[0] * color_intensity)
                                    g = int(bar_color[1] * color_intensity)
                                    b = int(bar_color[2] * color_intensity)
                                
                                # Add glow effect
                                if glow_enabled:
                                    glow_intensity = viz_config.get("glow", {}).get("intensity", 0.3)
                                    glow_color = viz_config.get("glow", {}).get("color", [100, 200, 255])
                                    
                                    # Add glow around the bar
                                    glow_factor = max(0, 1.0 - (y / max(wave_height, 1))) * glow_intensity
                                    r = min(255, r + int(glow_color[0] * glow_factor))
                                    g = min(255, g + int(glow_color[1] * glow_factor))
                                    b = min(255, b + int(glow_color[2] * glow_factor))
                                
                                # Add frequency-based color variation
                                freq_factor = i / num_bars
                                r = min(255, r + int(20 * freq_factor))
                                b = min(255, b + int(20 * (1 - freq_factor)))
                                
                                # Clamp values
                                r = max(0, min(255, r))
                                g = max(0, min(255, g))
                                b = max(0, min(255, b))
                                
                                # Set pixel
                                frame[viz_y_offset + viz_height - 1 - y, x + x_offset] = [b, g, r]  # BGR order
                                
                                # Add subtle glow effect to surrounding pixels
                                if glow_enabled and y < wave_height * 0.3:  # Only for top portion
                                    for dx in [-1, 1]:
                                        for dy in [-1, 1]:
                                            glow_x = x + x_offset + dx
                                            glow_y = viz_y_offset + viz_height - 1 - y + dy
                                            if (0 <= glow_x < self.video_width and 
                                                0 <= glow_y < frame.shape[0]):
                                                # Blend with existing pixel
                                                existing = frame[glow_y, glow_x]
                                                glow_blend = 0.3
                                                frame[glow_y, glow_x] = [
                                                    int(existing[0] * (1 - glow_blend) + b * glow_blend),
                                                    int(existing[1] * (1 - glow_blend) + g * glow_blend),
                                                    int(existing[2] * (1 - glow_blend) + r * glow_blend)
                                                ]
            
            return frame
        
        viz_clip = VideoClip(make_viz_frame, duration=duration)
        return viz_clip
    
    def create_background_video(self, duration: float) -> VideoClip:
        """Create a background video with enhanced gradient and subtle effects"""
        bg_config = self.config["background"]
        
        # Check if we want transparent background
        transparent_bg = self.config.get("transparent_background", False)
        
        if transparent_bg:
            # Create transparent background
            def make_frame(t):
                # Create a transparent frame (RGBA with alpha=0)
                frame = np.zeros((self.video_height, self.video_width, 4), dtype=np.uint8)
                frame[:, :, 3] = 0  # Set alpha to 0 (transparent)
                return frame
            
            background = VideoClip(make_frame, duration=duration)
        else:
            # Create enhanced gradient background with subtle effects
            def make_frame(t):
                # Create a gradient from top to bottom
                frame = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
                
                # Use colors from config
                top_color = bg_config["colors"]["top"]
                bottom_color = bg_config["colors"]["bottom"]
                
                # Add subtle animation to the gradient
                time_factor = np.sin(t * 0.5) * 0.05  # Subtle breathing effect
                
                for y in range(self.video_height):
                    ratio = y / self.video_height
                    
                    # Add subtle horizontal variation
                    x_factor = np.sin(y * 0.01 + t * 0.3) * 0.02
                    
                    # Interpolate between top and bottom colors with subtle variations
                    red = int(top_color[0] + ratio * (bottom_color[0] - top_color[0]) + time_factor * 10 + x_factor * 5)
                    green = int(top_color[1] + ratio * (bottom_color[1] - top_color[1]) + time_factor * 8 + x_factor * 3)
                    blue = int(top_color[2] + ratio * (bottom_color[2] - top_color[2]) + time_factor * 12 + x_factor * 4)
                    
                    # Clamp values to valid range
                    red = max(0, min(255, red))
                    green = max(0, min(255, green))
                    blue = max(0, min(255, blue))
                    
                    frame[y, :] = [blue, green, red]  # OpenCV uses BGR order
                
                # Add subtle overlay pattern if enabled
                if bg_config.get("overlay", {}).get("enabled", False):
                    overlay_opacity = bg_config["overlay"].get("opacity", 0.1)
                    pattern_type = bg_config["overlay"].get("type", "subtle_pattern")
                    
                    if pattern_type == "subtle_pattern":
                        # Create subtle diagonal lines
                        for y in range(0, self.video_height, 20):
                            for x in range(0, self.video_width, 20):
                                if (x + y) % 40 == 0:
                                    # Add subtle diagonal pattern
                                    for i in range(10):
                                        px = x + i
                                        py = y + i
                                        if px < self.video_width and py < self.video_height:
                                            # Add subtle brightness variation
                                            frame[py, px] = [
                                                min(255, frame[py, px, 0] + int(overlay_opacity * 20)),
                                                min(255, frame[py, px, 1] + int(overlay_opacity * 20)),
                                                min(255, frame[py, px, 2] + int(overlay_opacity * 20))
                                            ]
                
                return frame
            
            background = VideoClip(make_frame, duration=duration)
        
        return background
    
    def add_logos_to_video(self, video: VideoClip, logo_paths: dict) -> VideoClip:
        """Add logos to the video with enhanced styling"""
        if not logo_paths:
            return video
        
        logo_clips = []
        
        for name, logo_path in logo_paths.items():
            if logo_path.exists():
                try:
                    logo_config = self.config["logos"][name]
                    
                    # Load and resize logo
                    logo_img = Image.open(logo_path)
                    
                    # Resize logo based on config with auto-scaling
                    max_height = logo_config["max_height"]
                    max_width = logo_config.get("max_width", self.video_width * 0.4)
                    
                    # Calculate aspect ratio
                    aspect_ratio = logo_img.width / logo_img.height
                    
                    # Scale based on height first
                    new_height = min(max_height, logo_img.height)
                    new_width = int(new_height * aspect_ratio)
                    
                    # If width exceeds max_width, scale down proportionally
                    if new_width > max_width:
                        new_width = int(max_width)
                        new_height = int(new_width / aspect_ratio)
                    
                    logo_img = logo_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Add shadow effect if enabled
                    shadow_config = logo_config.get("shadow", {})
                    if shadow_config.get("enabled", False):
                        # Create shadow image
                        shadow_offset = shadow_config.get("offset", 2)
                        shadow_blur = shadow_config.get("blur", 4)
                        shadow_color = shadow_config.get("color", [0, 0, 0, 100])
                        
                        # Create shadow by duplicating the logo with shadow color
                        shadow_img = logo_img.copy()
                        
                        # Apply shadow color to non-transparent pixels
                        if shadow_img.mode == 'RGBA':
                            data = np.array(shadow_img)
                            # Create shadow effect for non-transparent pixels
                            alpha_mask = data[:, :, 3] > 0
                            data[alpha_mask, 0] = shadow_color[0]  # R
                            data[alpha_mask, 1] = shadow_color[1]  # G
                            data[alpha_mask, 2] = shadow_color[2]  # B
                            data[alpha_mask, 3] = shadow_color[3]  # A
                            shadow_img = Image.fromarray(data)
                        
                        # Create shadow clip
                        shadow_array = np.array(shadow_img)
                        shadow_clip = ImageClip(shadow_array).set_duration(video.duration)
                        
                        # Position shadow slightly offset
                        position = logo_config["position"]
                        margin = logo_config["margin"]
                        
                        if "left" in position:
                            x = margin + shadow_offset
                        elif "right" in position:
                            x = self.video_width - new_width - margin + shadow_offset
                        elif "center" in position:
                            x = (self.video_width - new_width) // 2 + shadow_offset
                        else:
                            x = (self.video_width - new_width) // 2 + shadow_offset
                        
                        if "top" in position:
                            y = margin + shadow_offset
                        elif "bottom" in position:
                            y = self.video_height - new_height - margin + shadow_offset
                        elif "center" in position and "top" not in position and "bottom" not in position:
                            y = (self.video_height - new_height) // 2 + shadow_offset
                        else:
                            y = (self.video_height - new_height) // 2 + shadow_offset
                        
                        shadow_clip = shadow_clip.set_position((x, y))
                        logo_clips.append(shadow_clip)
                    
                    # Convert to numpy array
                    logo_array = np.array(logo_img)
                    
                    # Handle transparency - keep RGBA for transparency
                    if logo_array.shape[2] == 4:  # RGBA
                        # Keep the RGBA format for transparency
                        pass
                    else:
                        # Convert RGB to RGBA if needed
                        rgba_array = np.zeros((new_height, new_width, 4), dtype=np.uint8)
                        rgba_array[:, :, :3] = logo_array
                        rgba_array[:, :, 3] = 255  # Full alpha
                        logo_array = rgba_array
                    
                    # Create video clip from logo
                    logo_clip = ImageClip(logo_array).set_duration(video.duration)
                    
                    # Position logo based on config
                    position = logo_config["position"]
                    margin = logo_config["margin"]
                    
                    if "left" in position:
                        x = margin
                    elif "right" in position:
                        x = self.video_width - new_width - margin
                    elif "center" in position:
                        x = (self.video_width - new_width) // 2
                    else:  # center
                        x = (self.video_width - new_width) // 2
                    
                    if "top" in position:
                        y = margin
                    elif "bottom" in position:
                        y = self.video_height - new_height - margin
                    elif "center" in position and "top" not in position and "bottom" not in position:
                        y = (self.video_height - new_height) // 2
                    else:  # center
                        y = (self.video_height - new_height) // 2
                    
                    logo_clip = logo_clip.set_position((x, y))
                    logo_clips.append(logo_clip)
                    
                except Exception as e:
                    logger.error(f"Failed to add logo {name}: {e}")
        
        # Composite all clips
        if logo_clips:
            final_video = CompositeVideoClip([video] + logo_clips)
            return final_video
        
        return video
    
    def convert_mp3_to_mp4(self, mp3_path: Path, force: bool = False) -> bool:
        """Convert a single MP3 file to MP4"""
        # Generate output filename
        output_filename = mp3_path.stem + ".mp4"
        output_path = self.output_folder / output_filename
        
        # Check if output already exists
        if output_path.exists() and not force:
            logger.info(f"Skipping {mp3_path.name} - output already exists")
            return True
        
        try:
            logger.info(f"Converting {mp3_path.name} to MP4...")
            
            # Load audio
            audio = AudioFileClip(str(mp3_path))
            duration = audio.duration
            
            # Create background video
            background = self.create_background_video(duration)
            
            # Download logos
            logo_paths = self.download_logos()
            
            # Add logos to background
            video_with_logos = self.add_logos_to_video(background, logo_paths)
            
            # Create audio visualization
            audio_viz = self.create_audio_visualization(audio, duration)
            
            # Transcribe audio and create captions
            transcription = self.transcribe_audio(mp3_path)
            caption_segments = self.create_captions(transcription, duration)
            
            # Create caption clips
            caption_clips = []
            for segment in caption_segments:
                caption_clip = self.create_caption_clip(
                    segment["text"], 
                    segment["start"], 
                    segment["end"]
                )
                if caption_clip:
                    caption_clips.append(caption_clip)
            
            # Combine video, audio visualization, and captions
            video_clips = [video_with_logos]
            if audio_viz:
                # Position audio visualization at the bottom with proper spacing
                viz_config = self.config.get("audio_visualization", {})
                background_enabled = viz_config.get("background", {}).get("enabled", True)
                
                if background_enabled:
                    # Position with background - it will extend below the main video
                    viz_y = self.video_height - 20  # 20px margin from bottom
                else:
                    # Position without background
                    viz_y = self.video_height - audio_viz.h - 50  # 50px margin from bottom
                
                audio_viz = audio_viz.set_position((0, viz_y))
                video_clips.append(audio_viz)
            video_clips.extend(caption_clips)
            
            final_video = CompositeVideoClip(video_clips)
            final_video = final_video.set_audio(audio)
            
            # Write output file
            logger.info(f"Writing {output_filename}...")
            
            # Check if we need transparent output
            transparent_bg = self.config.get("transparent_background", False)
            
            if transparent_bg:
                # Use codec that supports alpha channel
                codec = "libx264"  # or "prores_ks" for better alpha support
                # Note: For true alpha support, you might want to use .mov format
                # and prores_ks codec, but libx264 can work with some limitations
                output_path = output_path.with_suffix('.mov')
                logger.info(f"Writing transparent video to {output_path.name}...")
            else:
                codec = self.config["video_settings"]["codec"]
            
            final_video.write_videofile(
                str(output_path),
                fps=self.fps,
                codec=codec,
                audio_codec=self.config["video_settings"]["audio_codec"],
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Clean up
            audio.close()
            final_video.close()
            background.close()
            
            logger.info(f"Successfully converted {mp3_path.name} to {output_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert {mp3_path.name}: {e}")
            
            # Clean up any partial output file that may have been created
            if output_path.exists():
                try:
                    output_path.unlink()
                    logger.info(f"Cleaned up failed output file: {output_filename}")
                except Exception as cleanup_error:
                    logger.warning(f"Could not clean up failed output file {output_filename}: {cleanup_error}")
            
            return False
    
    def process_all_files(self, force: bool = False) -> None:
        """Process all MP3 files in the input folder"""
        mp3_files = list(self.input_folder.glob("*.mp3"))
        
        if not mp3_files:
            logger.info("No MP3 files found in input folder")
            return
        
        logger.info(f"Found {len(mp3_files)} MP3 files to process")
        
        successful = 0
        failed = 0
        
        for mp3_file in mp3_files:
            if self.convert_mp3_to_mp4(mp3_file, force):
                successful += 1
            else:
                failed += 1
        
        logger.info(f"Conversion complete: {successful} successful, {failed} failed")

def main():
    parser = argparse.ArgumentParser(description="Convert MP3 files to MP4 videos for social media")
    parser.add_argument("-i", "--input", default="input", help="Input folder containing MP3 files")
    parser.add_argument("-o", "--output", default="output", help="Output folder for MP4 files")
    parser.add_argument("-f", "--force", action="store_true", help="Force conversion even if output exists")
    
    args = parser.parse_args()
    
    # Create converter instance
    converter = MP3ToMP4Converter(args.input, args.output)
    
    # Process all files
    converter.process_all_files(args.force)

if __name__ == "__main__":
    main()
