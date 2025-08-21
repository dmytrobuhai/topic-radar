---
title: Topic Radar
emoji: 📚
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 5.43.1
app_file: app.py
pinned: false
---

# Topic Radar

Topic Radar is a lightweight Gradio app that lets you search **recent papers, news, and videos** about any topic and see them **side-by-side**.  
It fetches from arXiv, Crossref, OpenAlex, hf, selected tech RSS feeds, and YouTube, then ranks results by **lexical coverage + PRF expansion + recency**.  
You control the **percent mix** (e.g., 70% papers, 15% videos, 15% news), per-source split for papers (arXiv/Crossref/OpenAlex/HF), and fine-tune scoring **weights**.

## Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/dmytrobuhai/topic-radar.git
cd topic-radar
```
### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys (optional but recommended)
```bash
setx YT_API_KEY "your_youtube_api_key" # Windows
setx EMAIL "your_email" # Windows
export YT_API_KEY="your_youtube_api_key" # Linux / macOS
export EMAIL="your_email" # Linux / macOS
```

### 5. Run the app
```bash
python app.py
```
Each parser can also be executed independently in debug/CLI mode.
```bash
python -m papers.arxiv_parser "computer vision industrial automation" --days 30 --max 10 --debug

python -m papers.crossref_parser "machine learning robotics" --days 30 --max 5 --debug

python -m papers.openalex_parser "deep learning optimization" --days 30 --max 5 --debug

python -m papers.hf_parser "transformers natural language processing" --days 30 --max 5 --debug

python -m videos.youtube_api "computer vision industrial automation" --days 10 --max 10 --debug

python -m news.news_parser "artificial intelligence regulation" --days 7 --max 5 --debug

```