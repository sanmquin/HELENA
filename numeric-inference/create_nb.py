import json

def create_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# YouTube Dataset Builder for Numeric Inference\n",
                    "\n",
                    "This notebook collects video data (titles and views) from specific YouTube channels to build a dataset for predicting video popularity.\n",
                    "\n",
                    "## 1. Setup and Authentication\n",
                    "\n",
                    "Install the YouTube API client and mount Google Drive."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!pip install -q google-api-python-client\n",
                    "\n",
                    "import os\n",
                    "import json\n",
                    "import time\n",
                    "from datetime import datetime\n",
                    "from googleapiclient.discovery import build\n",
                    "from googleapiclient.errors import HttpError\n",
                    "from google.colab import drive, userdata\n",
                    "\n",
                    "try:\n",
                    "    drive.mount('/content/drive')\n",
                    "except ImportError:\n",
                    "    print('Not running in Colab or drive mount failed.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Configuration\n",
                    "\n",
                    "Retrieve your API key from Colab secrets and set up paths."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Configuration\n",
                    "try:\n",
                    "    YOUTUBE_API_KEY = userdata.get('YOUTUBE_API_KEY')\n",
                    "except Exception:\n",
                    "    print(\"Error: Could not retrieve YOUTUBE_API_KEY from Colab secrets.\")\n",
                    "    YOUTUBE_API_KEY = None\n",
                    "\n",
                    "INPUT_CHANNELS_PATH = '/content/drive/MyDrive/Graphiko/exports/base_data/latest/channels_structured.json'\n",
                    "OUTPUT_DATA_PATH = '/content/drive/MyDrive/Graphiko/exports/base_data/latest/channels_structured_v2.json'\n",
                    "CACHE_DIR = '/content/drive/MyDrive/youtube_api_cache/'\n",
                    "MAX_VIDEOS_PER_CHANNEL = 1000\n",
                    "\n",
                    "if not os.path.exists(CACHE_DIR):\n",
                    "    os.makedirs(CACHE_DIR, exist_ok=True)\n",
                    "\n",
                    "if YOUTUBE_API_KEY:\n",
                    "    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)\n",
                    "else:\n",
                    "    print(\"YouTube service NOT initialized. Please set YOUTUBE_API_KEY in Colab secrets.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Utility Functions\n",
                    "\n",
                    "Functions to interact with the YouTube API efficiently."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def get_longform_playlist_id(channel_id):\n",
                    "    \"\"\"Derive the 'Long-form' playlist ID from a Channel ID.\"\"\"\n",
                    "    if channel_id.startswith('UC'):\n",
                    "        return 'UULF' + channel_id[2:]\n",
                    "    return None\n",
                    "\n",
                    "def fetch_video_ids(playlist_id, max_results=1000):\n",
                    "    \"\"\"Fetch video IDs from a playlist using playlistItems.list.\"\"\"\n",
                    "    video_ids = []\n",
                    "    next_page_token = None\n",
                    "    \n",
                    "    while len(video_ids) < max_results:\n",
                    "        request = youtube.playlistItems().list(\n",
                    "            part=\"contentDetails\",\n",
                    "            playlistId=playlist_id,\n",
                    "            maxResults=min(50, max_results - len(video_ids)),\n",
                    "            pageToken=next_page_token\n",
                    "        )\n",
                    "        try:\n",
                    "            response = request.execute()\n",
                    "            video_ids.extend([item['contentDetails']['videoId'] for item in response.get('items', [])])\n",
                    "            next_page_token = response.get('nextPageToken')\n",
                    "            if not next_page_token:\n",
                    "                break\n",
                    "        except HttpError as e:\n",
                    "            print(f\"An error occurred: {e}\")\n",
                    "            break\n",
                    "            \n",
                    "    return video_ids\n",
                    "\n",
                    "def fetch_video_details(video_ids):\n",
                    "    \"\"\"Fetch video details (title, stats, publishedAt) in batches of 50.\"\"\"\n",
                    "    all_details = []\n",
                    "    for i in range(0, len(video_ids), 50):\n",
                    "        batch = video_ids[i:i+50]\n",
                    "        request = youtube.videos().list(\n",
                    "            part=\"snippet,statistics\",\n",
                    "            id=\",\".join(batch)\n",
                    "        )\n",
                    "        try:\n",
                    "            response = request.execute()\n",
                    "            for item in response.get('items', []):\n",
                    "                all_details.append({\n",
                    "                    'id': item['id'],\n",
                    "                    'title': item['snippet']['title'],\n",
                    "                    'views': int(item['statistics'].get('viewCount', 0)),\n",
                    "                    'publishedAt': item['snippet']['publishedAt']\n",
                    "                })\n",
                    "        except HttpError as e:\n",
                    "            print(f\"An error occurred: {e}\")\n",
                    "            \n",
                    "    return all_details"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Main Collection Loop\n",
                    "\n",
                    "Iterate through channels, fetch data, and cache results."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def load_channels():\n",
                    "    with open(INPUT_CHANNELS_PATH, 'r') as f:\n",
                    "        return json.load(f)\n",
                    "\n",
                    "def get_cache_path(channel_id):\n",
                    "    return os.path.join(CACHE_DIR, f\"{channel_id}.json\")\n",
                    "\n",
                    "def save_to_cache(channel_id, data):\n",
                    "    with open(get_cache_path(channel_id), 'w') as f:\n",
                    "        json.dump(data, f)\n",
                    "\n",
                    "def load_from_cache(channel_id):\n",
                    "    path = get_cache_path(channel_id)\n",
                    "    if os.path.exists(path):\n",
                    "        with open(path, 'r') as f:\n",
                    "            return json.load(f)\n",
                    "    return None\n",
                    "\n",
                    "channels_data = load_channels()\n",
                    "final_dataset = []\n",
                    "\n",
                    "for channel in channels_data:\n",
                    "    channel_id = channel['channel_id']\n",
                    "    channel_name = channel['channel_name']\n",
                    "    print(f\"Processing channel: {channel_name} ({channel_id})\")\n",
                    "    \n",
                    "    cached_videos = load_from_cache(channel_id)\n",
                    "    if cached_videos:\n",
                    "        print(f\"  Loaded {len(cached_videos)} videos from cache.\")\n",
                    "        final_dataset.append({\n",
                    "            'channel_id': channel_id,\n",
                    "            'channel_name': channel_name,\n",
                    "            'videos': cached_videos\n",
                    "        })\n",
                    "        continue\n",
                    "        \n",
                    "    playlist_id = get_longform_playlist_id(channel_id)\n",
                    "    if not playlist_id:\n",
                    "        print(f\"  Could not derive playlist ID for {channel_id}\")\n",
                    "        continue\n",
                    "        \n",
                    "    print(f\"  Fetching video IDs from playlist {playlist_id}...\")\n",
                    "    video_ids = fetch_video_ids(playlist_id, MAX_VIDEOS_PER_CHANNEL)\n",
                    "    \n",
                    "    print(f\"  Fetching details for {len(video_ids)} videos...\")\n",
                    "    videos_details = fetch_video_details(video_ids)\n",
                    "    \n",
                    "    save_to_cache(channel_id, videos_details)\n",
                    "    final_dataset.append({\n",
                    "        'channel_id': channel_id,\n",
                    "        'channel_name': channel_name,\n",
                    "        'videos': videos_details\n",
                    "    })\n",
                    "    print(f\"  Done. Collected {len(videos_details)} videos.\")\n",
                    "\n",
                    "# Save final dataset\n",
                    "with open(OUTPUT_DATA_PATH, 'w') as f:\n",
                    "    json.dump(final_dataset, f, indent=2)\n",
                    "print(f\"\\nFinal dataset saved to {OUTPUT_DATA_PATH}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Statistics Reporting\n",
                    "\n",
                    "Report statistics per channel and aggregate."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "total_videos = 0\n",
                    "print(f\"{'Channel Name':<30} | {'Videos':<10} | {'Avg Views':<15}\")\n",
                    "print(\"-\" * 60)\n",
                    "for channel in final_dataset:\n",
                    "    count = len(channel['videos'])\n",
                    "    total_videos += count\n",
                    "    avg_views = sum(v['views'] for v in channel['videos']) / count if count > 0 else 0\n",
                    "    print(f\"{channel['channel_name']:<30} | {count:<10} | {avg_views:<15,.0f}\")\n",
                    "\n",
                    "print(\"-\" * 60)\n",
                    "print(f\"Total videos across all channels: {total_videos}\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open('numeric-inference/youtube_dataset_builder.ipynb', 'w') as f:
        json.dump(nb, f, indent=2)

if __name__ == "__main__":
    create_notebook()
