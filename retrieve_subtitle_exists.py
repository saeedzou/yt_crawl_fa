import time
import csv
import argparse
import sys
import re
import json
import random
import subprocess
from pathlib import Path
from util import make_video_url, get_subtitle_language
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

def run_command(cmd):
    """Run command and return stdout, handling both stdout and stderr."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)
    return stdout, stderr

def count_punctuations(text):
    """Count both English and Persian punctuations in text."""
    # Combined English and Persian punctuation marks
    punctuation_marks = r'[،؛؟!\.,:;\?\!]'
    matches = re.findall(punctuation_marks, text)
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
            lines = f.readlines()
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
            lines = f.readlines()
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

def process_video(videoid, lang):
    """Process a single video to get metadata, download Persian subtitles, and analyze punctuation."""
    url = make_video_url(videoid)
    entry = {
        "videoid": videoid,
        "videourl": url,
        "good_sub": "False",
        "sub": "False",
        "title": "",
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
        "subtitle_duration": 0  # New field for subtitle duration
    }


    try:
        # First request: Get subtitle info
        cmd = f"yt-dlp --list-subs --skip-download {url} --cookies cookies.txt"
        stdout, stderr = run_command(cmd)
        auto_lang, manu_lang = get_subtitle_language(stdout)
        has_subtitle = lang in manu_lang and len(manu_lang) < 5
        entry["sub"] = str(has_subtitle)

        if has_subtitle:
            # Download Persian subtitle
            subtitle_filename = f"subtitles/{videoid}.{lang}.vtt"
            Path("subtitles").mkdir(exist_ok=True)
            download_cmd = f"yt-dlp --skip-download --sub-lang {lang} --write-sub --convert-subs vtt --cookies cookies.txt -o 'subtitles/%(id)s' {url}"
            stdout, stderr = run_command(download_cmd)

            # Extract text and count punctuations
            if Path(subtitle_filename).exists():
                subtitle_text = extract_text_from_subtitle(subtitle_filename)
                punct_count = count_punctuations(subtitle_text)
                entry["punctuation_count"] = punct_count
                
                # Calculate total subtitle duration
                subtitle_duration = calculate_subtitle_duration(subtitle_filename)
                entry["subtitle_duration"] = round(subtitle_duration, 2)  # Round to 2 decimal places
                if entry["subtitle_duration"] > 600 and punct_count > 5:
                    entry["good_sub"] = str(True)
                    # Get metadata
                    metadata_cmd = f"yt-dlp -j {url} --cookies cookies.txt"
                    stdout, stderr = run_command(metadata_cmd)
                    try:
                        metadata = json.loads(stdout)
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
                    except json.JSONDecodeError as e:
                        print(f"Error parsing metadata JSON for {videoid}. stdout: {stdout[:100]}...")

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
    if fn_checkpoint and Path(fn_checkpoint).exists():
        with open(fn_checkpoint, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subtitle_exists.append(row)
                processed_videoids.add(row["videoid"])

    # Load video ID list
    video_ids = []
    with open(fn_videoid, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_ids.append(row["video_id"])
    random.shuffle(video_ids)

    # Define fieldnames for CSV
    fieldnames = ["videoid", "videourl", "good_sub", "sub", "title", 
                 "channel", "channel_id", "channel_url", "channel_follower_count", 
                 "upload_date", "duration", "view_count", "categories", "like_count",
                 "punctuation_count", "subtitle_duration", "subtitle_coverage"]  # Added new fields


    # Process videos
    for videoid in tqdm(video_ids):
        if videoid in processed_videoids:
            continue

        entry = process_video(videoid, lang)
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