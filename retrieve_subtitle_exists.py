import time
import csv
import argparse
import sys
import re
import random
import os
import yt_dlp
import torch
import subprocess
import string
import re
import librosa
import silero_vad
import numpy as np
from pathlib import Path
from jiwer import wer, cer
from parsnorm import ParsNorm
from util import make_video_url
from nemo.collections.asr.models import ASRModel
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve video metadata and subtitle availability status.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("lang", type=str, help="language code (ja, en, ...)")
    parser.add_argument("videoidlist", type=str, help="filename of video ID list")
    parser.add_argument("--outdir", type=str, default="sub", help="dirname to save results")
    parser.add_argument("--checkpoint", type=str, default=None, help="filename of list checkpoint (for restart retrieving)")
    return parser.parse_args(sys.argv[1:])

def load_audio(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=16000)
    # convert to mono
    waveform = librosa.to_mono(waveform)
    # Normalize and convert to float32
    if waveform.dtype == 'int16':
        waveform = waveform.astype('float32') / 32768.0
    elif waveform.dtype == 'int32':
        waveform = waveform.astype('float32') / 2147483648.0
    elif waveform.dtype == 'uint8':
        waveform = (waveform.astype('float32') - 128) / 128.0
    else:
        # If already float32, ensure no further normalization is done
        waveform = waveform.astype('float32')

    return waveform, sample_rate

def segment_audio_with_vad(file_path, vad_model):
    waveform, sample_rate = load_audio(file_path)

    # Get speech timestamps
    speech_timestamps = silero_vad.get_speech_timestamps(
        waveform,
        vad_model,
        sampling_rate=sample_rate,
        min_speech_duration_ms=250,  # Minimum speech duration in ms
        min_silence_duration_ms=100  # Minimum silence duration in ms
    )

    # Extract speech segments
    speech_segments = []
    for segment in speech_timestamps:
        start = segment['start']
        end = segment['end']
        speech_segment = waveform[start:end]
        speech_segments.append(speech_segment)

    return speech_segments

def transcribe_chunk(chunk, model):
    transcription = model.transcribe([chunk], batch_size=1, verbose=False)
    return transcription[1][0]

def transcribe_audio(file_path, model, vad_model):
    # chunks = chunk_audio(file_path)
    chunks = segment_audio_with_vad(file_path, vad_model)
    transcriptions = []
    for chunk in chunks:
        transcription = transcribe_chunk(chunk, model)
        transcriptions.append(transcription)
    return ' '.join(transcriptions)

def load_model(model_path:str="/content/drive/MyDrive/stt_fa_fastconformer_hybrid_large_dataset_v30.nemo"):
    model = ASRModel.restore_from(restore_path=model_path)
    return model

vad_model = silero_vad.load_silero_vad(onnx=False)
normalizer = ParsNorm()
model = load_model()

def is_english(text):
    """Returns True if the text contains more than 50% English alphabet characters."""
    # Count English alphabet characters
    english_chars = sum(1 for char in text if char in string.ascii_letters)
    # Total characters in the text
    total_chars = len(text)
    # Avoid division by zero
    if total_chars == 0:
        return False
    # Calculate percentage of English characters
    return (english_chars / total_chars) > 0.5

def count_common_punctuations(text):
    """Count common punctuation marks in text."""
    common_punctuation_marks = r'[؟،]'
    matches = re.findall(common_punctuation_marks, text)
    return len(matches)

def count_other_punctuations(text):
    other_punctuation_marks = r'[!؛:]'
    matches = re.findall(other_punctuation_marks, text)
    return len(matches)

def parse_timestamp(timestamp):
    """Convert WebVTT timestamp to seconds."""
    # Format: HH:MM:SS.mmm
    hours, minutes, seconds = timestamp.split(':')
    seconds, milliseconds = seconds.split('.')
    total_seconds = (int(hours) * 3600 + 
                    int(minutes) * 60 + 
                    int(seconds) + 
                    int(milliseconds) / 1000)
    return total_seconds

def calculate_subtitle_duration(subtitle_file):
    """Calculate total duration covered by subtitles."""
    total_duration = 0
    try:
        with open(subtitle_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[3:]
            for line in lines:
                if '-->' in line:
                    # Extract start and end times
                    start, end = line.strip().split(' --> ')
                    start_time = parse_timestamp(start)
                    end_time = parse_timestamp(end)
                    duration = end_time - start_time
                    total_duration += duration
    except Exception as e:
        print(f"Error calculating subtitle duration: {e}")
        return 0
    return total_duration

def extract_text_from_subtitle(subtitle_file):
    """Extract plain text from subtitle file, removing timings."""
    text = ""
    try:
        with open(subtitle_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[3:]
            for line in lines:
                # Skip timeline patterns and empty lines
                if not line.strip():
                    continue
                if re.match(r'^\d+$', line.strip()):
                    continue
                if '-->' in line:
                    continue
                # Add non-empty, non-timeline lines to text
                if line.strip():
                    text += line.strip() + " "
    except Exception as e:
        print(f"Error reading subtitle file: {e}")
        return ""
    return text.strip()

def extract_subtitle_text(subtitle_file: str) -> str:
    if not subtitle_file or not os.path.exists(subtitle_file):
        return None
    with open(subtitle_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    text = " ".join(line.strip() for line in lines[3:] if not line.startswith('WEBVTT') and not line.strip().startswith('0') and line.strip())
    # remove text between parenthesis
    text = re.sub(r'\([^)]*\)', '', text)
    # remove text between square brackets
    text = re.sub(r'\[[^\]]*\]', '', text)
    # remove text between asterisks *
    text = re.sub(r'\*[^*]*\*', '', text)
    text = normalizer.normalize(text)
    return text

def download_video(video_id: str):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    os.makedirs('videos', exist_ok=True)
    output_template = f"videos/{video_id}.%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best',         # Download best audio quality
        'outtmpl': output_template,
        'skip_download': False,             # Download the audio
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        audio_file = ydl.prepare_filename(info).replace('.%(ext)s', info['ext'])

        return audio_file

def download_captions(video_id, lang):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    os.makedirs('subtitles', exist_ok=True)
    output_template = f"subtitles/{video_id}.%(ext)s"
    if lang == 'fa':
        lang = ['fa', 'fa-IR']
    else:
        lang = [lang]

    ydl_opts = {
        'outtmpl': output_template,
        'writesubtitles': True,             # Write manual subtitles
        'writeautomaticsubs': False,        # Explicitly disable auto-generated subtitles
        'subtitleslangs': lang,    # Only download Persian subtitles
        'skip_download': True,             # Download the audio
        'cookies': 'cookies.txt',
        'quiet': True,
        'list_subs': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)

        # Look for Persian subtitle file
        subtitle_file = None
        for i in lang:
            potential_file = f"subtitles/{video_id}.{i}.vtt"
            if os.path.exists(potential_file):
                subtitle_file = potential_file
                break

        return subtitle_file, info

def process_video(videoid, query_phrase, lang, processed_channels):
    """Process a single video to get metadata, download Persian subtitles, and analyze punctuation."""
    url = make_video_url(videoid)
    entry = {
        "videoid": videoid,
        "videourl": url,
        "good_sub": "False",
        "sub": "False",
        "title": "",
        "query_phrase": query_phrase,
        "channel": "",
        "channel_id": "",
        "channel_url": "",
        "channel_follower_count": "",
        "upload_date": "",
        "duration": "",
        "view_count": "",
        "categories": [],
        "like_count": "",
        "punctuation_count": 0,
        "subtitle_duration": 0,  # New field for subtitle duration
        "cer": "",
        "wer": "",
    }


    try:
        # First request: Get subtitle info
        subtitle_filename, metadata = download_captions(videoid, lang)
        manu_lang = list(metadata['subtitles'].keys())
        has_subtitle = lang in manu_lang and len(manu_lang) < 5
        entry["sub"] = str(has_subtitle)
        try:
            entry.update({
                'title': metadata.get('title', ''),
                'channel': metadata.get('uploader', ''),
                'channel_id': metadata.get('uploader_id', ''),
                'channel_url': metadata.get('uploader_url', ''),
                'channel_follower_count': metadata.get('channel_follower_count', ''),
                'upload_date': metadata.get('upload_date', ''),
                'duration': metadata.get('duration', ''),
                'view_count': metadata.get('view_count', ''),
                'categories': metadata.get('categories', []),
                'like_count': metadata.get('like_count', '')
            })
        except Exception as e:
            print(f"Error updating metadata: {e}") 

        if has_subtitle:
            print(f"Downloaded subtitle for video {videoid} to {subtitle_filename}")

            # Extract text and count punctuations
            if Path(subtitle_filename).exists():
                subtitle_text = extract_text_from_subtitle(subtitle_filename)
                common_punct = count_common_punctuations(subtitle_text)
                other_punct = count_other_punctuations(subtitle_text)
                punct_count = common_punct + other_punct
                entry["punctuation_count"] = punct_count
                
                # Calculate total subtitle duration
                subtitle_duration = calculate_subtitle_duration(subtitle_filename)
                entry["subtitle_duration"] = round(subtitle_duration, 2)  # Round to 2 decimal places
                if (entry["subtitle_duration"] > 10) and (not is_english(subtitle_text)) and (common_punct > 5 or other_punct > 1):
                    print(f"Downloading and processing audio for video {videoid}")
                    print(url)
                    audio_file = download_video(videoid)
                    auto_transcription = transcribe_audio(audio_file, model, vad_model)
                    auto_transcription = re.sub(' +', ' ', auto_transcription)
                    manual_transcription = extract_subtitle_text(subtitle_filename)
                    manual_transcription = re.sub(' +', ' ', manual_transcription)
                    word_error_rate = wer(manual_transcription, auto_transcription)
                    character_error_rate = cer(manual_transcription, auto_transcription)
                    entry["wer"] = word_error_rate
                    entry["cer"] = character_error_rate
                    if word_error_rate < 0.8 and character_error_rate < 0.2:
                        entry["good_sub"] = str(True)


    except subprocess.CalledProcessError as e:
        print(f"Error processing video {videoid}. stdout: {e.stdout}, stderr: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error processing video {videoid}: {str(e)}")

    return entry

def retrieve_subtitle_exists(lang, fn_videoid, outdir="sub", wait_sec=0.2, fn_checkpoint=None):
    fn_sub = Path(outdir) / f"{Path(fn_videoid).stem}.csv"
    fn_sub.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if provided
    subtitle_exists = []
    processed_videoids = set()
    processed_channels = set()
    if fn_checkpoint and Path(fn_checkpoint).exists():
        with open(fn_checkpoint, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subtitle_exists.append(row)
                processed_videoids.add(row["videoid"])
                processed_channels.add(row["channel_id"])

    # Load video ID list
    video_ids = []
    with open(fn_videoid, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_ids.append((row["video_id"], row['word']))
    random.shuffle(video_ids)

    # Define fieldnames for CSV
    fieldnames = ["videoid", "videourl", "good_sub", "sub", "title", 
                  "query_phrase",
                 "channel", "channel_id", "channel_url",
                 "punctuation_count", "subtitle_duration",
                 "wer", "cer",
                 "channel_follower_count", "upload_date", "duration", 
                 "view_count", "categories", "like_count", "subtitle_coverage"]


    # Process videos
    for videoid, query_phrase in tqdm(video_ids):
        if videoid in processed_videoids:
            continue

        entry = process_video(videoid, query_phrase, lang, processed_channels)
        subtitle_exists.append(entry)

        if wait_sec > 0.01:
            time.sleep(wait_sec)

        # Write current result every 2 videos
        if len(subtitle_exists) % 2 == 0:
            with open(fn_sub, "w", newline="", encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(subtitle_exists)

    # Final write
    with open(fn_sub, "w", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subtitle_exists)

    return fn_sub

if __name__ == "__main__":
    args = parse_args()
    filename = retrieve_subtitle_exists(
        args.lang, 
        args.videoidlist, 
        args.outdir, 
        fn_checkpoint=args.checkpoint
    )
    print(f"Saved {args.lang.upper()} subtitle info, metadata, and punctuation counts to {filename}.")