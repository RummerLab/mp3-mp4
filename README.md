# MP3 to MP4 Converter for Social Media

Turns an audio clip (an interview, a statement for the media, a podcast excerpt) into a
vertical video ready for YouTube Shorts, Instagram Reels and TikTok:

- **Portrait 1080x1920** (9:16), H.264 + AAC, 30 fps
- **RummerLab and PhysioShark logos** on a white card, over an ocean-blue gradient
- **Title card**: label, name, subtitle, date and a topic line
- **Audio visualiser**: mirrored frequency bars that move with the voice
- **Word-by-word captions** from a local [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  transcription (no API key needed), with the spoken word highlighted
- **`.srt` subtitles and a `.txt` transcript** alongside the video, for YouTube's subtitle
  upload and the video description

Rendering is done by FFmpeg, so a two-minute clip renders in about half a minute.
Transcripts are cached, so re-rendering after changing the title text is quick.

## Installation

1. **Install FFmpeg** (it must include libass, which the standard builds do)
   - **Windows**: `winget install Gyan.FFmpeg` (or `scoop install ffmpeg`)
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

2. **Install the Python dependencies** (Python 3.10+)
   ```bash
   python -m venv .venv
   .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

The first run downloads the Whisper model (`medium.en`, about 1.5 GB) and the two logos.

## Making a video

1. Put the audio file in `input/` (mp3, m4a, wav, aac, flac, ogg and opus all work).
2. Run the converter, giving the title-card text for this clip:
   ```bash
   python mp3_to_mp4_converter.py input/jodie-rummer-bbc-statement-2026-10-05.m4a \
       --label "Statement for the BBC" \
       --date "5 October 2026" \
       --topic "Shark bites, climate change & physiology"
   ```
3. Collect the results from `output/`:
   - `<name>.mp4`: the video
   - `<name>.srt`: subtitles to upload in YouTube Studio (Subtitles → Upload file)
   - `<name>.txt`: the transcript, handy for writing the description
   - `<name>.transcript.json`: the cached word-timed transcript

Check the captions before you upload. Whisper can mishear names: add fixes to
`captions.corrections` in `config.json`, then re-run with `-f`. The cached transcript is
reused, so this takes seconds.

### Title-card text

Each line is optional and is skipped if empty:

| Field      | Example                                   | Default (config.json)                     |
|------------|-------------------------------------------|-------------------------------------------|
| `label`    | Statement for the BBC (shown in capitals) |                                           |
| `title`    | Prof. Jodie Rummer                        | Prof. Jodie Rummer                        |
| `subtitle` | Marine biologist · James Cook University  | Marine biologist · James Cook University  |
| `date`     | 5 October 2026                            |                                           |
| `topic`    | Shark bites, climate change & physiology  |                                           |
| `footer`   | jodierummer.com                           | jodierummer.com                           |

You can set them in three places. Each one overrides the one before:

1. `text` in `config.json`: the defaults for every video
2. A JSON file next to the audio with the same name (`input/<name>.json`), or one passed
   with `--meta path/to/file.json`. See [`examples/`](examples/) for the BBC statement's.
3. Command-line flags: `--label`, `--title`, `--subtitle`, `--date`, `--topic`, `--footer`

### Command-line options

```bash
python mp3_to_mp4_converter.py                  # convert everything in input/
python mp3_to_mp4_converter.py input/clip.mp3   # convert specific files
python mp3_to_mp4_converter.py -f               # re-render even if the .mp4 exists
python mp3_to_mp4_converter.py --retranscribe   # ignore the cached transcript
python mp3_to_mp4_converter.py -i in -o out     # custom input/output folders
python mp3_to_mp4_converter.py --config other.json
```

## Configuration (`config.json`)

| Section               | What it controls                                                        |
|-----------------------|-------------------------------------------------------------------------|
| `video`               | Frame rate, x264 quality (`crf`, lower is better), audio bitrate        |
| `background`          | Gradient colours, top and bottom, as RGB                                |
| `logos`               | Logo URLs shown on the white card, left to right (downloaded to `.cache/`) |
| `text`                | Default title-card text (see above)                                     |
| `fonts`               | `regular`/`bold` `.ttf` paths. If unset, uses Segoe UI (Windows), Arial (macOS) or DejaVu Sans (Linux). |
| `captions`            | On/off, size, colours, line length, vertical position, name `corrections` |
| `audio_visualization` | On/off, position, size, number of bars, frequency range, colours        |
| `whisper`             | Model (`small.en` is faster, `medium.en` more accurate), language, and an `initial_prompt` that helps with names |

## Uploading to YouTube

- Clips of up to 3 minutes in this vertical format are published as **Shorts**.
- Upload the `.srt` under **Subtitles** so viewers can turn on captions and search can index
  the transcript. The burned-in captions stay on screen either way.
- YouTube's Shorts buttons and title cover the bottom fifth and the right edge of the
  frame, so the title card, visualiser and captions sit in the upper two-thirds.

## Troubleshooting

- **`ffmpeg not found`**: install FFmpeg and restart your terminal so it is on `PATH`.
- **`No bold font found`**: set `fonts.bold` and `fonts.regular` in `config.json` to `.ttf` files.
- **Logo download failures**: the video is still made, just without that logo. Check the
  URLs in `config.json`.
- **Wrong words in the captions**: add them to `captions.corrections`, or put the right
  spelling in `whisper.initial_prompt` and re-run with `--retranscribe`.
