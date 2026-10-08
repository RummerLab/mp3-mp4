#!/usr/bin/env python3
"""
MP3 to MP4 Converter for Social Media

Turns an audio clip (mp3, m4a, wav, ...) into a vertical 1080x1920 video for
YouTube Shorts, Instagram Reels and TikTok: ocean gradient, RummerLab and
PhysioShark logos, a title card, an audio visualiser and word-by-word captions.

Captions come from a local faster-whisper transcription (cached next to the
output), and frames are rendered by ffmpeg, so re-rendering after a text tweak
takes seconds.
"""

import argparse
import copy
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
    import requests
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus"}
TEXT_FIELDS = ("label", "title", "subtitle", "date", "topic", "footer")

# Portrait 9:16. The layout below is in these pixel coordinates.
W, H = 1080, 1920
SAMPLE_RATE = 16000  # whisper's rate; also plenty for the visualiser

DEFAULT_CONFIG = {
    "video": {"fps": 30, "crf": 18, "audio_bitrate": "192k"},
    "background": {"top": [50, 100, 200], "bottom": [12, 38, 84]},
    "logos": {},
    "text": {field: "" for field in TEXT_FIELDS},
    "fonts": {"regular": None, "bold": None},
    "captions": {
        "enabled": True,
        "font_size": 76,
        "color": [255, 255, 255],
        "highlight_color": [150, 212, 255],
        "max_chars": 26,
        "y": 1290,
        "corrections": {},
    },
    "audio_visualization": {
        "enabled": True,
        "y": 905,
        "width": 900,
        "height": 260,
        "num_bars": 48,
        "bar_width": 12,
        "min_hz": 100,
        "max_hz": 8000,
        "center_color": [150, 220, 255],
        "edge_color": [90, 170, 235],
    },
    "whisper": {"model": "medium.en", "language": "en", "initial_prompt": None},
}

FONT_CANDIDATES = {
    "regular": [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf",
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ],
    "bold": [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeuib.ttf",
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ],
}


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict:
    config = DEFAULT_CONFIG
    if path.exists():
        config = deep_merge(config, json.loads(path.read_text(encoding="utf-8")))
    else:
        logger.warning(f"{path} not found - using built-in defaults")
    return config


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH. Install FFmpeg (with libass) and try again.")


def probe_duration(audio_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio_path)],
        capture_output=True, encoding="utf-8", check=True)
    return float(result.stdout.strip())


def decode_audio(audio_path: Path) -> np.ndarray:
    """Decode any ffmpeg-readable audio to mono float32 at SAMPLE_RATE."""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(audio_path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


@functools.lru_cache(maxsize=1)
def load_whisper(model_name: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper is not installed: pip install -r requirements.txt")
    logger.info(f"Loading Whisper model '{model_name}' (first use downloads it)...")
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def apply_corrections(text: str, corrections: dict) -> str:
    for wrong, right in corrections.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text)
    return text


def srt_timestamp(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ass_timestamp(t: float) -> str:
    cs = int(round(max(t, 0) * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_colour(rgb) -> str:
    r, g, b = rgb
    return f"&H{b:02X}{g:02X}{r:02X}&"


def find_font(kind: str, configured) -> Path:
    candidates = [Path(configured)] if configured else FONT_CANDIDATES[kind]
    for path in candidates:
        if path.exists():
            return path
    sys.exit(f"No {kind} font found. Set fonts.{kind} in config.json to a .ttf file.")


class MP3ToMP4Converter:
    def __init__(self, input_folder: str = "input", output_folder: str = "output",
                 config_path: Path = ROOT / "config.json"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.input_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)

        self.config = load_config(config_path)
        self.fps = self.config["video"]["fps"]
        self.font_regular = find_font("regular", self.config["fonts"]["regular"])
        self.font_bold = find_font("bold", self.config["fonts"]["bold"])
        self.cache_folder = ROOT / ".cache"

    # ------------------------------------------------------------ inputs

    def download_logos(self) -> list:
        """Download logos (cached in .cache/) and return their local paths, in config order."""
        self.cache_folder.mkdir(exist_ok=True)
        paths = []
        for name, logo in self.config["logos"].items():
            path = self.cache_folder / f"{name}_logo.png"
            if not path.exists():
                try:
                    logger.info(f"Downloading {name} logo...")
                    response = requests.get(logo["url"], timeout=30)
                    response.raise_for_status()
                    path.write_bytes(response.content)
                except Exception as e:
                    logger.error(f"Failed to download {name} logo: {e}")
                    continue
            paths.append(path)
        return paths

    def card_text(self, audio_path: Path, meta_path, overrides: dict) -> dict:
        """Title-card text: config.json defaults < <audio>.json sidecar (or --meta) < CLI flags."""
        text = dict(self.config["text"])
        meta_path = Path(meta_path) if meta_path else audio_path.with_suffix(".json")
        if meta_path.exists():
            logger.info(f"Using title-card text from {meta_path}")
            text.update(json.loads(meta_path.read_text(encoding="utf-8")))
        text.update({k: v for k, v in overrides.items() if v is not None})
        return text

    def transcribe(self, audio_path: Path, pcm: np.ndarray, retranscribe: bool) -> list:
        """Word-timed transcript, cached as output/<name>.transcript.json."""
        cache = self.output_folder / f"{audio_path.stem}.transcript.json"
        if cache.exists() and not retranscribe:
            logger.info(f"Using cached transcript {cache.name} (pass --retranscribe to redo it)")
            return json.loads(cache.read_text(encoding="utf-8"))["segments"]

        whisper_cfg = self.config["whisper"]
        model = load_whisper(whisper_cfg["model"])
        logger.info(f"Transcribing {audio_path.name}...")
        segments, _ = model.transcribe(
            pcm, language=whisper_cfg["language"], initial_prompt=whisper_cfg["initial_prompt"],
            word_timestamps=True, vad_filter=True, beam_size=5)
        result = [{
            "start": s.start, "end": s.end, "text": s.text.strip(),
            "words": [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])],
        } for s in segments]

        with open(cache, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"model": whisper_cfg["model"], "segments": result}, f, indent=2, ensure_ascii=False)
        return result

    def caption_words(self, segments: list) -> list:
        """Flatten the transcript into corrected, timed words."""
        corrections = self.config["captions"]["corrections"]
        words = []
        for s in segments:
            for i, w in enumerate(s["words"]):
                token = apply_corrections(w["word"].strip(), corrections)
                seg_end = i == len(s["words"]) - 1
                if token.startswith("-") and words:  # "long" + "-term" -> "long-term"
                    words[-1].update(t=words[-1]["t"] + token, end=w["end"], seg_end=seg_end)
                    continue
                words.append({"t": token, "start": w["start"], "end": w["end"], "seg_end": seg_end})
        return words

    @staticmethod
    def chunk_words(words: list, max_chars: int, break_after: str, min_words: int = 3) -> list:
        """Group words into lines, breaking after punctuation, at segment ends, or once long enough."""
        chunks, current = [], []
        for w in words:
            current.append(w)
            line = " ".join(x["t"] for x in current)
            if ((w["t"] and w["t"][-1] in break_after and len(current) >= min_words) or w["seg_end"]
                    or len(line) >= max_chars):
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)
        return chunks

    def write_transcripts(self, segments: list, stem: str) -> None:
        """SRT for YouTube's subtitle upload, plain text for the video description."""
        words = self.caption_words(segments)
        with open(self.output_folder / f"{stem}.srt", "w", encoding="utf-8", newline="\n") as f:
            for i, cue in enumerate(self.chunk_words(words, max_chars=70, break_after=".?!", min_words=1), 1):
                text = " ".join(w["t"] for w in cue)
                if len(text) > 42:  # two balanced lines read better than one long one
                    middle = len(text) // 2
                    split = min((m.start() for m in re.finditer(" ", text)), key=lambda p: abs(p - middle))
                    text = text[:split] + "\n" + text[split + 1:]
                f.write(f"{i}\n{srt_timestamp(cue[0]['start'])} --> {srt_timestamp(cue[-1]['end'])}\n{text}\n\n")
        with open(self.output_folder / f"{stem}.txt", "w", encoding="utf-8", newline="\n") as f:
            f.write(" ".join(w["t"] for w in words) + "\n")

    # ------------------------------------------------------------ rendering

    def build_background(self, text: dict, logo_paths: list) -> Image.Image:
        bg_cfg = self.config["background"]
        ramp = np.linspace(0, 1, H)[:, None]
        top, bottom = np.array(bg_cfg["top"]), np.array(bg_cfg["bottom"])
        gradient = (top + (bottom - top) * ramp).astype(np.uint8)
        image = Image.fromarray(np.broadcast_to(gradient[:, None, :], (H, W, 3)).copy()).convert("RGBA")
        draw = ImageDraw.Draw(image)

        # Logos sit on a white card - their black/navy artwork disappears on the gradient.
        if logo_paths:
            cx0, cy0, cx1, cy1 = 60, 110, W - 60, 370
            draw.rounded_rectangle((cx0, cy0, cx1, cy1), radius=36, fill=(255, 255, 255, 255))
            slot_w = (cx1 - cx0 - 80) / len(logo_paths)
            for i, path in enumerate(logo_paths):
                x0 = cx0 + 40 + i * slot_w
                if i:
                    draw.line((x0, cy0 + 40, x0, cy1 - 40), fill=(210, 220, 230), width=3)
                logo = Image.open(path).convert("RGBA")
                box_w, box_h = slot_w - 20, cy1 - cy0 - 60
                scale = min(box_w / logo.width, box_h / logo.height)
                logo = logo.resize((int(logo.width * scale), int(logo.height * scale)), Image.LANCZOS)
                image.alpha_composite(logo, (int(x0 + (slot_w - logo.width) / 2),
                                             int(cy0 + (cy1 - cy0 - logo.height) / 2)))

        def fitted(path: Path, size: int, content: str, max_w: int) -> ImageFont.FreeTypeFont:
            font = ImageFont.truetype(str(path), size)
            while draw.textlength(content, font=font) > max_w and size > 20:
                size -= 2
                font = ImageFont.truetype(str(path), size)
            return font

        def centered(content: str, y: int, font: ImageFont.FreeTypeFont, fill) -> None:
            draw.text(((W - draw.textlength(content, font=font)) / 2, y), content, font=font, fill=fill)

        y = 455
        if text["label"]:
            centered(text["label"].upper(), y, fitted(self.font_bold, 46, text["label"].upper(), 960),
                     (150, 220, 255))
            y += 65
        if text["title"]:
            centered(text["title"], y, fitted(self.font_bold, 92, text["title"], 960), (255, 255, 255))
            y += 130
        for field in ("subtitle", "date"):
            if text[field]:
                centered(text[field], y, fitted(self.font_regular, 44, text[field], 960), (220, 235, 250))
                y += 62
        if text["topic"]:
            y += 26
            font = fitted(self.font_bold, 42, text["topic"], 880)
            text_w = draw.textlength(text["topic"], font=font)
            pill = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(pill).rounded_rectangle(
                ((W - text_w) / 2 - 36, y, (W + text_w) / 2 + 36, y + 72), radius=36,
                fill=(255, 255, 255, 38), outline=(150, 220, 255, 160), width=2)
            image.alpha_composite(pill)
            centered(text["topic"], y + 8, font, (255, 255, 255))
        if text["footer"]:
            centered(text["footer"], 1500, fitted(self.font_bold, 40, text["footer"], 960), (190, 225, 250))
        return image.convert("RGB")

    def build_captions(self, segments: list, font_family: str) -> str:
        """ASS subtitles: short lines, with the word being spoken highlighted."""
        cap = self.config["captions"]
        words = self.caption_words(segments)
        for w in words:  # braces and backslashes are ASS override syntax
            w["t"] = w["t"].replace("{", "(").replace("}", ")").replace("\\", "/")
        chunks = self.chunk_words(words, max_chars=cap["max_chars"], break_after=".?!,;")

        highlight, normal = f"{{\\c{ass_colour(cap['highlight_color'])}}}", f"{{\\c{ass_colour(cap['color'])}}}"
        position = f"{{\\pos({W // 2},{cap['y']})}}"
        events = []
        for ci, chunk in enumerate(chunks):
            next_start = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else chunk[-1]["end"] + 0.8
            chunk_end = min(next_start, chunk[-1]["end"] + 0.8)
            for wi in range(len(chunk)):
                start = chunk[wi]["start"] if wi else chunk[0]["start"]
                end = chunk[wi + 1]["start"] if wi + 1 < len(chunk) else chunk_end
                line = " ".join(highlight + x["t"] + normal if j == wi else x["t"] for j, x in enumerate(chunk))
                events.append(f"Dialogue: 0,{ass_timestamp(start)},{ass_timestamp(end)},Cap,,0,0,0,,"
                              f"{position}{line}")

        primary = "&H00" + ass_colour(cap["color"])[2:-1]  # styles take &HAABBGGRR
        return "\n".join([
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {W}",
            f"PlayResY: {H}",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Cap,{font_family},{cap['font_size']},{primary},{primary},&H00301A08,&H96000000,"
            "-1,0,0,0,100,100,0,0,1,5,3,5,110,110,0,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            *events,
            "",
        ])

    def visualizer_frames(self, pcm: np.ndarray, n_frames: int):
        """Mirrored frequency bars (low frequencies in the middle), as raw RGBA frames."""
        viz = self.config["audio_visualization"]
        viz_w, viz_h, n_bars, bar_w = viz["width"], viz["height"], viz["num_bars"], viz["bar_width"]

        n_fft = 2048
        padded = np.concatenate([np.zeros(n_fft // 2, np.float32), pcm, np.zeros(n_fft, np.float32)])
        starts = (np.arange(n_frames) * SAMPLE_RATE / self.fps).astype(int)
        windows = np.stack([padded[s:s + n_fft] for s in starts]) * np.hanning(n_fft).astype(np.float32)
        magnitude = np.abs(np.fft.rfft(windows, axis=1))
        freqs = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)
        edges = np.geomspace(viz["min_hz"], viz["max_hz"], n_bars + 1)
        bands = np.stack([magnitude[:, (freqs >= lo) & (freqs < hi)].mean(axis=1)
                          for lo, hi in zip(edges[:-1], edges[1:])], axis=1)
        db = 20 * np.log10(bands + 1e-9)
        level = np.clip((db - (np.percentile(db, 99.5) - 50)) / 50, 0, 1) ** 1.5

        smooth = np.zeros_like(level)  # fast attack, slow release
        for t in range(n_frames):
            prev = smooth[t - 1] if t else 0
            smooth[t] = np.where(level[t] > prev, 0.6 * level[t] + 0.4 * prev, 0.82 * prev + 0.18 * level[t])
        heights = np.concatenate([smooth[:, ::-1], smooth], axis=1)

        n_cols = heights.shape[1]
        gap = (viz_w - n_cols * bar_w) / (n_cols - 1)
        xs = np.arange(viz_w)
        column = np.floor(xs / (bar_w + gap)).astype(int).clip(0, n_cols - 1)
        in_bar = (xs - column * (bar_w + gap)) < bar_w
        dist = np.abs(np.arange(viz_h) - (viz_h - 1) / 2)[:, None]
        mix = np.clip(1 - dist / (viz_h / 2), 0, 1)[..., None]
        center, edge = np.array(viz["center_color"]), np.array(viz["edge_color"])
        rgb = np.broadcast_to((edge + (center - edge) * mix).astype(np.uint8), (viz_h, viz_w, 3))

        for t in range(n_frames):
            half = 4 + heights[t][column] * (viz_h / 2 - 6)
            frame = np.zeros((viz_h, viz_w, 4), np.uint8)
            frame[..., :3] = rgb
            frame[..., 3] = ((dist <= half[None, :]) & in_bar[None, :]) * 235
            yield frame.tobytes()

    # ------------------------------------------------------------ pipeline

    def convert(self, audio_path: Path, force: bool = False, retranscribe: bool = False,
                meta_path=None, text_overrides: dict = None) -> bool:
        output_path = self.output_folder / f"{audio_path.stem}.mp4"
        if output_path.exists() and not force:
            logger.info(f"Skipping {audio_path.name} - {output_path.name} already exists (use -f to redo)")
            return True

        try:
            logger.info(f"Converting {audio_path.name}...")
            duration = probe_duration(audio_path)
            n_frames = int(np.ceil(duration * self.fps))
            pcm = decode_audio(audio_path)
            text = self.card_text(audio_path, meta_path, text_overrides or {})

            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                self.build_background(text, self.download_logos()).save(work / "background.png")

                inputs = ["-loop", "1", "-framerate", str(self.fps), "-i", "background.png",
                          "-i", str(audio_path.resolve())]
                video_chain = "[0:v]"
                filters = []
                viz = self.config["audio_visualization"]
                if not viz["enabled"]:
                    inputs.insert(0, "-nostdin")
                else:
                    inputs += ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{viz['width']}x{viz['height']}",
                               "-framerate", str(self.fps), "-i", "-"]
                    filters.append(f"{video_chain}[2:v]overlay=(W-w)/2:{viz['y']}[withviz]")
                    video_chain = "[withviz]"
                if self.config["captions"]["enabled"]:
                    segments = self.transcribe(audio_path, pcm, retranscribe)
                    self.write_transcripts(segments, audio_path.stem)
                    (work / "fonts").mkdir()
                    shutil.copy(self.font_bold, work / "fonts")
                    family = ImageFont.truetype(str(self.font_bold), 10).getname()[0]
                    with open(work / "captions.ass", "w", encoding="utf-8", newline="\n") as f:
                        f.write(self.build_captions(segments, family))
                    # Relative paths: Windows drive letters break ffmpeg's filter syntax.
                    filters.append(f"{video_chain}subtitles=captions.ass:fontsdir=fonts[withcaps]")
                    video_chain = "[withcaps]"
                filters.append(f"{video_chain}format=yuv420p[v]")

                video = self.config["video"]
                cmd = ["ffmpeg", "-y", "-v", "error", "-stats", *inputs,
                       "-filter_complex", ";".join(filters), "-map", "[v]", "-map", "1:a",
                       "-c:v", "libx264", "-preset", "medium", "-crf", str(video["crf"]), "-r", str(self.fps),
                       "-c:a", "aac", "-b:a", video["audio_bitrate"], "-ar", "48000",
                       "-t", f"{duration:.3f}", "-movflags", "+faststart", str(output_path.resolve())]

                logger.info(f"Rendering {output_path.name}...")
                proc = subprocess.Popen(cmd, cwd=work, stdin=subprocess.PIPE if viz["enabled"] else None)
                if viz["enabled"]:
                    try:
                        for frame in self.visualizer_frames(pcm, n_frames):
                            proc.stdin.write(frame)
                    except BrokenPipeError:
                        pass  # ffmpeg exited early; its error is reported below
                    finally:
                        proc.stdin.close()
                if proc.wait():
                    raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

            logger.info(f"Successfully converted {audio_path.name} to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to convert {audio_path.name}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False

    def process_all_files(self, files=None, **options) -> None:
        files = files or sorted(p for p in self.input_folder.iterdir() if p.suffix.lower() in AUDIO_EXTS)
        if not files:
            logger.info(f"No audio files found in {self.input_folder}/")
            return

        logger.info(f"Found {len(files)} audio file(s) to process")
        results = [self.convert(Path(f), **options) for f in files]
        logger.info(f"Conversion complete: {sum(results)} successful, {len(results) - sum(results)} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio clips to vertical captioned MP4 videos for social media")
    parser.add_argument("files", nargs="*", help="Audio files to convert (default: everything in the input folder)")
    parser.add_argument("-i", "--input", default="input", help="Input folder containing audio files")
    parser.add_argument("-o", "--output", default="output", help="Output folder for MP4 files")
    parser.add_argument("-f", "--force", action="store_true", help="Force conversion even if output exists")
    parser.add_argument("--retranscribe", action="store_true", help="Ignore the cached transcript")
    parser.add_argument("--config", default=str(ROOT / "config.json"), help="Path to config.json")
    parser.add_argument("--meta", help="JSON file with title-card text (default: <audio name>.json next to the audio)")
    for field in TEXT_FIELDS:
        parser.add_argument(f"--{field}", help=f"Title-card {field} (overrides config and --meta)")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    require_ffmpeg()

    converter = MP3ToMP4Converter(args.input, args.output, Path(args.config))
    converter.process_all_files(
        [Path(f) for f in args.files], force=args.force, retranscribe=args.retranscribe,
        meta_path=args.meta, text_overrides={field: getattr(args, field) for field in TEXT_FIELDS})


if __name__ == "__main__":
    main()
