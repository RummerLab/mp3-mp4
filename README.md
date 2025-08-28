# MP3 to MP4 Converter for Social Media

A Python script that converts MP3 audio files to MP4 videos optimized for social media platforms like YouTube Shorts, Instagram Reels, and TikTok. Features include:

- **Portrait format** (9:16 aspect ratio) optimized for mobile viewing
- **Auto-generated captions** using OpenAI's Whisper speech recognition
- **Brand logos** automatically added to videos
- **Ocean-themed gradient background** fitting for marine biology content
- **Transparent background support** for overlay videos
- **Batch processing** with skip/force options

## Features

- Converts MP3 files to MP4 videos in portrait format (1080x1920)
- Automatically downloads and adds RummerLab and PhysioShark logos
- Generates captions from audio using Whisper AI
- Creates beautiful ocean-themed gradient backgrounds
- Processes multiple files in batch
- Skips already converted files (unless forced)
- Comprehensive logging and error handling

## Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd mp3-mp4
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (required by MoviePy)
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (CentOS/RHEL)

4. **Configure environment variables** (optional)
   ```bash
   python setup_env.py
   ```
   Or manually copy `env.template` to `.env` and edit the values.

## Usage

### Basic Usage

1. **Create input and output folders** (automatically created if they don't exist)
   ```
   mp3-mp4/
   ├── input/          # Place your MP3 files here
   ├── output/         # Converted MP4 files will be saved here
   └── mp3_to_mp4_converter.py
   ```

2. **Place your MP3 files in the `input` folder**

3. **Run the converter**
   ```bash
   python mp3_to_mp4_converter.py
   ```

### Advanced Usage

```bash
# Use custom input/output folders
python mp3_to_mp4_converter.py -i /path/to/input -o /path/to/output

# Force conversion (overwrite existing files)
python mp3_to_mp4_converter.py -f

# Combine options
python mp3_to_mp4_converter.py -i /path/to/input -o /path/to/output -f
```

### Command Line Options

- `-i, --input`: Input folder containing MP3 files (default: "input")
- `-o, --output`: Output folder for MP4 files (default: "output")
- `-f, --force`: Force conversion even if output file already exists

## Output Specifications

- **Resolution**: 1080x1920 (portrait format)
- **Frame rate**: 30 FPS
- **Codec**: H.264 video, AAC audio
- **Aspect ratio**: 9:16 (optimized for mobile/social media)
- **Background**: Ocean-themed gradient (blue tones) or transparent
- **Logos**: RummerLab (top-left) and PhysioShark (top-right)
- **Captions**: White text with black outline, positioned at bottom
- **Transparency**: Optional alpha channel support for overlay videos

## Logo Sources

The script automatically downloads logos from:
- **RummerLab**: https://rummerlab.com/images/rummerlab_logo_transparent.png
- **PhysioShark**: https://physioshark.org/images/logo-physioshark-project.png

## Environment Variables

The converter supports environment variables for configuration. Create a `.env` file or use the setup script:

```bash
python setup_env.py
```

### Key Environment Variables

- **`OPENAI_API_KEY`**: Your OpenAI API key for Whisper transcription
- **`VIDEO_WIDTH`**: Video width (default: 480)
- **`VIDEO_HEIGHT`**: Video height (default: 854)
- **`ENABLE_CAPTIONS`**: Enable/disable captions (true/false)
- **`ENABLE_LOGOS`**: Enable/disable logos (true/false)
- **`BG_TOP_RED/GREEN/BLUE`**: Background gradient top color (RGB)
- **`BG_BOTTOM_RED/GREEN/BLUE`**: Background gradient bottom color (RGB)
- **`TRANSPARENT_BACKGROUND`**: Enable transparent background (true/false)

See `env.template` for all available options.

## Caption Generation

The script uses OpenAI's Whisper model to:
1. Transcribe the audio content (local model or API)
2. Segment the text into readable chunks
3. Time-sync captions with the audio
4. Display captions with professional styling

### Caption Options

- **Local Whisper**: Uses local model (requires more disk space)
- **OpenAI API**: Uses cloud API (requires API key, faster)
- **Disabled**: Skip captions entirely

### Transparent Background

To create videos with transparent backgrounds (useful for overlays):

1. **Set environment variable**:
   ```bash
   export TRANSPARENT_BACKGROUND=true
   ```

2. **Or modify config.json**:
   ```json
   {
     "transparent_background": true
   }
   ```

3. **Output format**: Videos will be saved as `.mov` files with alpha channel support

**Note**: Transparent videos are ideal for:
- Overlaying on other videos
- Creating video effects
- Professional video editing workflows

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Install FFmpeg and ensure it's in your system PATH
   - Restart your terminal after installation

2. **Missing dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Whisper model download issues**
   - The script will continue without captions if Whisper fails to load
   - Check your internet connection for initial model download

4. **Logo download failures**
   - The script will continue without logos if downloads fail
   - Check your internet connection and logo URLs

### Performance Tips

- **First run**: May take longer as Whisper model downloads (~1GB)
- **Large files**: Consider processing during off-peak hours
- **Storage**: Ensure sufficient disk space for video output
- **Memory**: Video processing can be memory-intensive

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place MP3 files in input folder
cp /path/to/your/interviews/*.mp3 input/

# 3. Run conversion
python mp3_to_mp4_converter.py

# 4. Check output folder for MP4 files
ls output/
```

## File Structure

```
mp3-mp4/
├── input/                          # MP3 input files
│   ├── interview1.mp3
│   ├── interview2.mp3
│   └── ...
├── output/                         # MP4 output files
│   ├── interview1.mp4
│   ├── interview2.mp4
│   ├── rummerlab_logo.png         # Downloaded logos
│   └── physioshark_logo.png
├── mp3_to_mp4_converter.py        # Main script
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## License

This project is open source. Feel free to modify and distribute as needed.

## Support

For issues or questions, please check the troubleshooting section above or create an issue in the repository.
