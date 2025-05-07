# Video Automation Script

A Python CLI tool that automates the end-to-end creation of narrated videos with AI-generated scripts, text-to-speech, images, and background music.

## Features

- Generate video narration using OpenAI GPT-4
- Text-to-speech using Azure Cognitive Services
- AI image generation via getimg.ai
- Subtitle (.srt) generation
- Background music selection from local library
- Automatic video composition with moviepy
- Cost and time report for each run
- Supports multiple languages (English, Spanish, Portuguese, German)
- Interactive CLI to select number of videos, language, channel, and title

## Prerequisites

- Python 3.8+
- FFmpeg installed and in PATH

## Installation

```bash
git clone https://github.com/Levilaell/script-video-ai.git
cd script-video-ai
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`:

   ```bash
   cp .env.example .env
   ```

2. Set your API keys in `.env`:

   ```
   OPENAI_KEY=your_openai_key_here
   GETIMG_KEY=your_getimg_key_here
   AZURE_KEY=your_azure_key_here
   SPEECH_REGION=your_speech_region_here
   ```

## Usage

Run the script:

```bash
python script-video.py
```

Follow the prompts:

1. **How many videos?**
2. **Choose language** (1=English, 2=Spanish, 3=Portuguese, 4=German)
3. **Choose channel** from the displayed list
4. **Enter or choose a title**

The script will:

- Generate the video script in two parts using GPT-4
- Synthesize speech with bookmarks using Azure TTS
- Generate images asynchronously via getimg.ai
- Compose video with subtitles and background music
- Output MP4 files in the `videos/` directory
- Display a cost and time report

## File Structure

```
.
├── script-video.py
├── .env.example
├── .env
├── .gitignore
├── soundtracks/     # Background music folders by channel
├── videos/          # Generated videos output
└── README.md
```

## Contributing

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/my-feature`)  
3. Commit changes (`git commit -m "Add feature"`)  
4. Push to branch (`git push origin feature/my-feature`)  
5. Open a pull request

---

## 📝 License

MIT License

---

## ✉️ Contact

Levi Lael • [linkedin.com/in/levilael](https://www.linkedin.com/in/levi-lael-939b4a1b9/)
