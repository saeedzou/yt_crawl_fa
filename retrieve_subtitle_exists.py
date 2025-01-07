import time
import csv
import argparse
import sys
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

def process_video(videoid, lang):
    """Process a single video to get metadata and subtitle info (without downloading subtitles)."""
    url = make_video_url(videoid)
    entry = {
        "videoid": videoid,
        "videourl": url,
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
        "like_count":"",

    }

    try:
        # First request: Get subtitle info (without downloading them)
        cmd = f"yt-dlp --list-subs --skip-download {url} --cookies cookies.txt"
        stdout, stderr = run_command(cmd)
        auto_lang, manu_lang = get_subtitle_language(stdout)
        has_subtitle = lang in manu_lang and len(manu_lang) < 5
        entry["sub"] = str(has_subtitle)

        # If subtitle exists or we need metadata, make second request to fetch metadata
        if has_subtitle:  # Always get metadata
            metadata_cmd = f"yt-dlp -j {url} --cookies cookies.txt"
            stdout, stderr = run_command(metadata_cmd)
            try:
                import json
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
    fieldnames = ["videoid", "videourl", "sub", "title", 
                  "channel", "channel_id", "channel_url", "channel_follower_count", 
                  "upload_date", "duration", "view_count", "categories", "like_count"]

    # Process videos
    for videoid in tqdm(video_ids):
        if videoid in processed_videoids:
            continue

        # Call process_video without download_subs, as we are not downloading subtitles
        entry = process_video(videoid, lang)
        subtitle_exists.append(entry)

        # Sleep to avoid overloading requests
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
    print(f"Saved {args.lang.upper()} subtitle info and metadata to {filename}.")
