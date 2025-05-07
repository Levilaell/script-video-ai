import openai
import azure.cognitiveservices.speech as speechsdk # type: ignore
import time
import re
from nltk.tokenize.punkt import PunktSentenceTokenizer # type: ignore
import os
import sys
import json
import asyncio
import aiohttp
from functools import partial
from pydub import AudioSegment # type: ignore
import concurrent.futures
import nltk # type: ignore
import threading
from xml.sax.saxutils import escape
import traceback
from dotenv import load_dotenv
import websocket # type: ignore
import uuid
import requests
import json
import urllib.request
import urllib.parse
import os
from PIL import Image # type: ignore
import random
import io
import datetime
from datetime import datetime
from moviepy.audio.AudioClip import concatenate_audioclips, CompositeAudioClip # type: ignore
from moviepy.audio.io.AudioFileClip import AudioFileClip # type: ignore
from moviepy.audio.fx.AudioFadeIn import AudioFadeIn # type: ignore
from moviepy.audio.fx.AudioFadeOut import AudioFadeOut # type: ignore
from moviepy.audio.fx.MultiplyVolume import MultiplyVolume # type: ignore
from moviepy.video.VideoClip import ImageClip # type: ignore
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip # type: ignore
from moviepy.video.fx.Crop import Crop # type: ignore
from moviepy.video.fx.FadeIn import FadeIn # type: ignore
from moviepy.video.fx.FadeOut import FadeOut # type: ignore
from moviepy.video.fx.CrossFadeIn import CrossFadeIn # type: ignore
from moviepy.video.fx.CrossFadeOut import CrossFadeOut # type: ignore
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Use todos os núcleos disponíveis
executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
print(f"CPUs disponíveis: {multiprocessing.cpu_count()}")

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["NUMEXPR_MAX_THREADS"] = str(multiprocessing.cpu_count())

load_dotenv()
nltk.download('punkt', force=True)
punkt_tokenizer = PunktSentenceTokenizer()
lock = threading.Lock()

openai_cost_per_1k_input_tokens = 0.000150
openai_cost_per_1k_output_tokens = 0.000600
azure_cost_per_million_characters = 15.00
total_input_tokens = 0
total_output_tokens = 0
total_characters_synthesized = 0
cost_per_image = 0.002359296
total_images_generated = 0

start_time = time.time()

openai.api_key = os.getenv('OPENAI_KEY')
speech_config = speechsdk.SpeechConfig(
    subscription=os.getenv('AZURE_KEY'),
    region='eastus'
)
speech_config.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
)


def track_openai_tokens(response):
    """
    Soma a contagem de tokens de entrada e saída em variáveis globais.
    """
    global total_input_tokens, total_output_tokens
    input_tokens = response['usage']['prompt_tokens']
    output_tokens = response['usage']['completion_tokens']
    with lock:
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens


def calculate_gpt_cost():
    input_cost = (total_input_tokens / 1000) * openai_cost_per_1k_input_tokens
    output_cost = (total_output_tokens / 1000) * openai_cost_per_1k_output_tokens
    return input_cost, output_cost


def calculate_azure_tts_cost():
    total_cost = (total_characters_synthesized / 1_000_000) * azure_cost_per_million_characters
    return total_cost


def generate_report():
    global total_input_tokens, total_output_tokens, total_characters_synthesized, start_time, total_images_generated

    input_cost, output_cost = calculate_gpt_cost()
    openai_cost = input_cost + output_cost
    azure_tts_cost = calculate_azure_tts_cost()
    image_generation_cost = total_images_generated * cost_per_image
    total_cost = openai_cost + azure_tts_cost + image_generation_cost

    total_time = time.time() - start_time

    report = f"""
    ----- Cost and Time Report -----
    OpenAI Input Tokens Used: {total_input_tokens}
    OpenAI Output Tokens Used: {total_output_tokens}
    Estimated OpenAI Input Cost: ${input_cost:.4f}
    Estimated OpenAI Output Cost: ${output_cost:.4f}
    Total OpenAI Cost: ${openai_cost:.4f}

    Characters Synthesized by Azure TTS: {total_characters_synthesized}
    Estimated Azure TTS Cost: ${azure_tts_cost:.4f}

    Total Images Generated: {total_images_generated}
    Estimated Image Generation Cost: ${image_generation_cost:.4f}

    Total Cost: ${total_cost:.4f}
    Total Time Spent: {total_time:.2f} seconds
    ---------------------------------------
    """
    print(report)


def cleaned_phrases(text):
    phrases = punkt_tokenizer.tokenize(text)
    phrases = [phrase.strip() for phrase in phrases if phrase.strip()]
    return phrases


def split_sentences_into_chunks(sentences, max_characters=4500):
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_characters:
            chunks.append(current_chunk)
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def milliseconds_to_srt_time(ms):
    total_seconds = ms / 1000
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(ms % 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def text_to_audio_with_bookmarks(ssml_chunks, audio_filepath):
    global total_characters_synthesized
    combined_audio = AudioSegment.empty()
    all_bookmark_timings = []
    total_offset = 0

    for chunk_idx, ssml_text in enumerate(ssml_chunks):
        total_characters = len(re.sub(r'<[^>]+>', '', ssml_text))
        with lock:
            total_characters_synthesized += total_characters

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        bookmark_timings = []

        def on_bookmark_reached(evt):
            bookmark_timings.append({
                'bookmark': evt.text,
                'offset': evt.audio_offset / 10000
            })

        synthesizer.bookmark_reached.connect(on_bookmark_reached)

        result = synthesizer.speak_ssml_async(ssml_text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            audio_stream = io.BytesIO(audio_data)
            chunk_audio = AudioSegment.from_file(audio_stream, format="mp3")
            combined_audio += chunk_audio
            for bm in bookmark_timings:
                all_bookmark_timings.append({
                    'bookmark': bm['bookmark'],
                    'offset': bm['offset'] + total_offset
                })
            total_offset += len(chunk_audio)
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")

    combined_audio.export(audio_filepath, format="mp3")
    return all_bookmark_timings


def clean_prompt_text(prompt_text):
    prompt_text = re.sub(r'^\s*(\.?\s:|\d+\.\s+|\d+\)\s+|\*\s+)', '', prompt_text)
    prompt_text = prompt_text.strip()
    return prompt_text


def validate_image_prompts(prompts):
    prompts = [p.strip() for p in prompts if p.strip()]
    return len(prompts) >= 2 and len(set(prompts)) >= 2


def generate_image_prompts(title, phrases, channel_name, max_attempts=3):
    image_prompts = []
    total_phrases = len(phrases)
    index_phrases = list(enumerate(phrases))
    results = [None] * len(phrases)
    context_phrase = ''

    if channel_name == 'Historias de Crimen y Terror' or channel_name == 'Crime and Horror Stories':
        context_phrase = 'dark, sinister, '
    elif channel_name == 'La Biblia Explorada' or channel_name == 'The Bible Explored':
        context_phrase = 'biblical era, ancient, '
    else:
        context_phrase = ''

    def generate_prompt_for_phrase(idx, phrase, title):
        print(f'Generating prompt for phrase {idx + 1}/{total_phrases}...')
        prompt = f"""
Based on the title '{title}', create a simplified prompt for generating an image for the sentence:
"{phrase}"
Do not use proper names, as the image generator may not recognize them.
Specify the scenery of the era, culture, and clothing (if there is a person).
Do not use characteristics unless they are physical, nor adverbs of manner.
Do not include emotional contexts.
Do not use symbols, posters, texts, banners or messages.
Avoid abstract or philosophical descriptions. Respond in an objective and literal manner.
Obs: Add 'realistic' on the final
Respond in english.
"""
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                track_openai_tokens(response)
                response_content = response.choices[0].message['content'].strip()
                generated_prompts = response_content.split("\n")
                # Pega apenas a primeira linha
                generated_prompt = generated_prompts[0].strip()
                if generated_prompt:
                    clean_prompt = f"{context_phrase}{clean_prompt_text(generated_prompt)}"
                    results[idx] = (idx + 1, clean_prompt)
                    print(f"Prompt successfully generated for phrase {idx + 1}")
                    return
                else:
                    print(f"Attempt {attempts} failed: No prompt found. Retrying...")
            except Exception as e:
                print(f"Attempt {attempts} failed: {e}")
                traceback.print_exc()
            if attempts < max_attempts:
                time.sleep(1)
            else:
                print(f"Failed to generate prompt for phrase {idx + 1}")
                results[idx] = (idx + 1, f"Error generating prompt for phrase: {phrase}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_prompt_for_phrase, idx, phrase, title) for idx, phrase in index_phrases]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred while generating prompts: {e}")
                traceback.print_exc()

    for prompt in results:
        if prompt:
            image_prompts.append(prompt)

    return image_prompts


async def async_generate_image_with_getimg(args, semaphore):
    global total_images_generated
    idx, phrase_num, prompt_text, images_dir, image_index = args
    print(f"Processing image {image_index} for phrase {phrase_num}...")

    url = "https://api.getimg.ai/v1/flux-schnell/text-to-image"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {os.getenv('GETIMG_KEY')}"
    }

    payload = {
        "prompt": prompt_text,
        "width": 1280,
        "height": 768,
        "steps": 4,
        "response_format": "url",
        "output_format": "jpeg"
    }

    retry_delay = 2  # Starta com 2 segundos para rate limit
    max_retries = 7

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            image_url = result.get('url')
                            if image_url:
                                # Salva a imagem localmente
                                if not os.path.exists(images_dir):
                                    os.makedirs(images_dir)

                                filename = os.path.join(images_dir, f'image_{image_index}.jpeg')
                                async with session.get(image_url) as image_response:
                                    with open(filename, 'wb') as img_file:
                                        img_file.write(await image_response.read())

                                with Image.open(filename) as img:
                                    upscaled_img = img.resize((1920, 1080), Image.LANCZOS)
                                    upscaled_img.save(filename)

                                print(f"Image {image_index} saved successfully at {filename}.")
                                total_images_generated += 1
                                return filename
                            else:
                                print(f"No 'url' in response for image {image_index}.")
                                return None
                        elif response.status == 429:  # Rate limit
                            print(f"Rate limit hit for image {image_index}. Retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            print(f"Unexpected status code {response.status} for image {image_index}.")
                            return None
            except Exception as e:
                print(f"Error generating image {image_index}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Max retries reached for image {image_index}. Skipping...")
                    return None


async def async_generate_images(args_list):
    """Orquestra a geração assíncrona de imagens."""
    semaphore = asyncio.Semaphore(5)  # Limita requisições simultâneas
    tasks = [async_generate_image_with_getimg(args, semaphore) for args in args_list]
    results = await asyncio.gather(*tasks)
    return results


def generate_images_with_async(args_list):
    """Wrapper de execução do async_generate_images."""
    results = asyncio.run(async_generate_images(args_list))
    return results


async def process_images_async(args_list, images_dir):
    """Orquestra a geração assíncrona de imagens."""
    semaphore = asyncio.Semaphore(5)
    tasks = [
        async_generate_image_with_getimg(args, semaphore)
        for args in args_list
    ]
    results = await asyncio.gather(*tasks)
    return results


def save_to_file(filepath, content):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"Saved to file: {filepath}")


def sanitize_filename(filename):
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1F\'`]', '', filename)
    filename = filename.strip()
    return filename


def generate_ssml_with_bookmarks(sentences, voice_name, channel_name):
    style = 'serious'
    pitch = '-0.7st'
    rate = '-5%'
    volume = 'default'

    if channel_name == 'Crime and Horror Stories':
        style = 'serious'

    if channel_name == 'Historias de Crimen y Terror':
        style = 'serious'
        pitch = '-1.0st'

    if channel_name in ['A Bíblia Explorada', 'Mundo Descoberto']:
        style = 'calm'
        rate = '-13%'
        pitch = '-0.7st'
        

    ssml_parts = [
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="http://www.w3.org/2001/mstts" '
        'xmlns:emo="http://www.w3.org/2009/10/emotionml" '
        f'xml:lang="{voice_name.split("-")[0]}">',
        f'<voice name="{voice_name}">'
    ]
    for idx, sentence in enumerate(sentences):
        escaped_sentence = escape(sentence)
        ssml_parts.append(
            f'<bookmark mark="sentence_{idx+1}"/> '
            f'<mstts:express-as style="{style}">'
            f'<prosody rate="{rate}" pitch="{pitch}" volume="{volume}">'
            f'{escaped_sentence}'
            f'</prosody>'
            f'</mstts:express-as>'
        )
    ssml_parts.append('</voice></speak>')
    ssml_content = ''.join(ssml_parts)
    return ssml_content


def get_soundtrack_folder(channel_name):
    """
    Define a pasta de trilha sonora com base no canal.
    """
    # Mapeamento do canal para a pasta
    channel_to_folder = {
        'the bible explored': 'bible',
        'la biblia explorada': 'bible',
        'a bíblia explorada': 'bible',

        'crime and horror stories': 'horror',
        'historias de crimen y terror': 'horror',

        'secrets of money': 'money',
        'secretos del dinero': 'money',

        'explored universe': 'universe',
        'el universo explorado': 'universe',
        'erforschtes universum': 'universe',
        'o universo explorado': 'universe',

        'unveiled world': 'curiosities',
        'mundo desvelado': 'curiosities',
        'mundo descoberto': 'curiosities',

        'living history': 'historias',
        'historia viva': 'historias',
        'a história viva': 'historias',
        'lebendige geschichte': 'historias'

    }

    base_dir = os.path.dirname(os.path.abspath(__file__))

    folder = os.path.join(base_dir, 'soundtracks', channel_to_folder.get(channel_name.lower(), 'default'))
    print(f"Resolved soundtrack folder: {folder}")  # Debug
    return folder


def generate_video(sentence_timings, images_dir, audio_file, output_video_path, soundtrack_folder):
    clips = []
    current_time = 0
    fade_duration = 0.5
    video_width, video_height = 1920, 1080

    # Cria um clipe de imagem para cada sentença
    for idx, entry in enumerate(sentence_timings):
        start_time = entry["start_time"] / 1000.0
        end_time = entry["end_time"] / 1000.0
        duration = end_time - start_time

        image_index = idx + 1
        image_path = os.path.join(images_dir, f"image_{image_index}.jpeg")

        if not os.path.exists(image_path):
            print(f"Image {image_path} not found, skipping this sentence.")
            continue

        image_clip = ImageClip(image_path).with_duration(duration)

        w_img, h_img = image_clip.size
        zoom_start = 1.25
        zoom_end = 1.15

        scale_factor = max(video_width / w_img, video_height / h_img) * zoom_end
        image_clip = image_clip.resized(scale_factor / zoom_end)

        def scaling(t):
            return zoom_start - (zoom_start - zoom_end) * (t / image_clip.duration)

        # Aplica zoom suave
        image_clip = image_clip.resized(lambda t: scaling(t))

        # Recorta imagem no tamanho do vídeo
        image_clip = Crop(x1=0, y1=0, x2=video_width, y2=video_height).apply(image_clip)

        if idx == 0:
            image_clip = image_clip.with_start(current_time)
        else:
            image_clip = image_clip.with_start(current_time - fade_duration)

        # CrossFade
        if idx < len(sentence_timings) - 1:
            image_clip = CrossFadeOut(fade_duration).apply(image_clip)
        if idx > 0:
            image_clip = CrossFadeIn(fade_duration).apply(image_clip)

        current_time += duration
        clips.append(image_clip)

    final_clip = CompositeVideoClip(clips, size=(video_width, video_height))
    final_clip = final_clip.with_duration(current_time)

    # Áudio da narração
    narration_audio = AudioFileClip(audio_file)
    narration_duration = narration_audio.duration

    # Tenta adicionar trilha sonora
    if not os.path.exists(soundtrack_folder):
        print(f"Soundtrack folder {soundtrack_folder} does not exist. Proceeding without background music.")
        final_clip = final_clip.with_audio(narration_audio)
    else:
        music_files = [os.path.join(soundtrack_folder, f) for f in os.listdir(soundtrack_folder) if f.endswith(('.mp3', '.wav'))]
        if not music_files:
            print(f"No music files found in {soundtrack_folder}. Proceeding without background music.")
            final_clip = final_clip.with_audio(narration_audio)
        else:
            random.shuffle(music_files)
            segment_duration = 120
            fade_duration_music = 2.0
            music_clips = []
            current_time_music = 0.0
            music_file_index = 0
            while current_time_music < narration_duration:
                music_file = music_files[music_file_index % len(music_files)]
                music_file_index += 1
                music_clip = AudioFileClip(music_file)
                desired_duration = segment_duration + fade_duration_music

                if music_clip.duration > desired_duration:
                    music_clip = music_clip.subclipped(0, desired_duration)
                else:
                    loops = int(desired_duration // music_clip.duration) + 1
                    music_clip = concatenate_audioclips([music_clip] * loops)
                    music_clip = music_clip.subclipped(0, desired_duration)

                music_clip = music_clip.with_effects([AudioFadeIn(fade_duration_music), AudioFadeOut(fade_duration_music)])
                music_clip = music_clip.with_effects([MultiplyVolume(0.06)])
                start_time = max(0, current_time_music - fade_duration_music)
                music_clip = music_clip.with_start(start_time)
                music_clips.append(music_clip)
                current_time_music += segment_duration

            background_music = CompositeAudioClip(music_clips)
            background_music = background_music.subclipped(0, narration_duration)
            final_audio = CompositeAudioClip([
                narration_audio.with_effects([MultiplyVolume(1.5)]),
                background_music
            ])
            final_clip = final_clip.with_audio(final_audio)

    temp_audiofile = os.path.join(os.path.dirname(output_video_path), "temp_audio.m4a")
    final_clip.write_videofile(
        output_video_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        threads=multiprocessing.cpu_count(),  # Use o máximo de threads disponíveis
        temp_audiofile=temp_audiofile,
        remove_temp=True,
        ffmpeg_params=["-crf", "18"],
    )


def validate_list_response(response_text):
    lines = response_text.strip().split("\n")
    for line in lines:
        if re.match(r'^\d+[\.\)]\s', line) or ':' in line:
            return False
    return True


def clean_list_items(items):
    cleaned_items = [re.sub(r'^\-+\s*', '', item).strip() for item in items]
    return cleaned_items


def get_titles(title_idea, language, max_attempts=3):
    prompt = f"""
Based on the title idea '{title_idea}', generate a list of 5 concise and impactful video titles that maximize curiosity, engagement, and SEO potential. Each title must:

Directly align with the theme: Focus on delivering clarity while maintaining a strong connection to the topic.
Leverage high-impact keywords: Include terms likely to resonate with the target audience's search queries.
Evoke curiosity naturally: Use emotionally compelling language or unexpected phrases to intrigue viewers, avoiding unnecessary complexity.
Remain authentic: Avoid misleading claims or over-promising while keeping a strong click-worthy appeal.
Achieve a balance of intrigue and specificity: Hint at what the viewer will learn, ensuring the title sets clear expectations for the content.
If the idea is biblical, never suggest that the Bible is a myth. The Bible is 100% real.

Follow the examples:
'
How did Newton explain the motion of the planets?
5 Ways AI Is Changing the World
What Happens to Your Body During Fasting?
'

Output Guidelines:
Deliver exactly 5 titles.
Provide each title on a separate line without numbering, labels, or additional text.
Respond in {language}.
    """
    for attempt in range(max_attempts):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            track_openai_tokens(response)
            titles_text = response.choices[0].message['content']
            titles = [line.strip() for line in titles_text.split('\n') if line.strip()]
            titles = clean_list_items(titles)
            return titles
        except Exception as e:
            print(f"Attempt {attempt+1}: An exception occurred: {e}")
            if attempt < max_attempts - 1:
                time.sleep(1)
    print("Failed to obtain the titles in the desired format after multiple attempts.")
    return []


def generate_content_in_two_calls_with_old_prompt(title, language, channel_name, max_attempts=3):
    """
    Gera o conteúdo para o vídeo em duas solicitações, mas mantendo o mesmo prompt original.
    1) Solicitação para a primeira metade do texto (cerca de 1250 palavras).
    2) Solicitação para a segunda metade (mais ~1250), repassando o texto já gerado, 
       para ter ~2500 palavras no total.
    """
    total_words = 2800
    half_words = total_words // 2  # Primeira metade

    # --- Prompt antigo, sem alterações ---
    if channel_name in ['La Biblia Explorada', 'The Bible Explored', 'A Bíblia Explorada']:
        base_prompt = f'''
Write a narration of {total_words} words based on the topic '{title}', specifically crafted to align with a Christian audience and provide biblical insights that meet and exceed viewer expectations set by the title, while keeping them engaged and curious throughout the video.

Instructions:
1. Start with Impact

Open with a impactful hook that immediately connects with the audience's curiosity, addressing the core question or mystery posed by the title.
Make it clear in the first 7 seconds that the viewer should continue watching.

2. Consistently Deliver Biblical Value

Provide relevant scripture references, biblical insights, and spiritual revelations frequently, ideally every 50-100 words.
Never put the Bible to the test or in contradiction with the christian faith. The Bible is 100% real.
Alternate between factual biblical teachings, thought-provoking interpretations, and emotional reflections to maintain variety and engagement.

3. Maintain Curiosity with Mini-Hooks

Expose curiosities about the theme.
Use faith-based mini-hooks every 100-150 words to sustain interest:
Pose rhetorical questions that align with viewer curiosity.
Include teasers about upcoming insights or biblical revelations.
Use surprising interpretations of scripture to challenge traditional thinking in a respectful manner.

4. Build Toward a Spiritually Rewarding Conclusion

Gradually escalate the narrative or spiritual insights toward the most uplifting or profound revelation.
Ensure the conclusion ties all elements together, leaving viewers with a sense of title content delivered.

5. Stay Aligned with the Title

Ensure every part of the narration aligns with the promise of the title and a biblical perspective.
Avoid unnecessary tangents or overgeneralizations.
Fully satisfy the curiosity generated by the title, while keeping Christ-centered teachings at the forefront.

6. Use an Engaging and Inspirational Tone

Write in a conversational yet reverent tone that resonates with a Christian audience, ensuring it feels authentic, dynamic.
Make viewers feel like they're on a faith-filled journey of discovery, guided by biblical truth.

7. Reflect the Viewer's Faith and Curiosity

Pose rhetorical questions sparingly but strategically to reflect the audience's curiosity and spiritual hunger.
Use phrases like:
“Have you ever wondered what the Bible says about…?”
“Here's a passage that might surprise you…”
“What comes next will deepen your understanding of God's word…”

Language and Word Count
Write the text in {language}, ensuring it meets or exceeds the {total_words} word count requirement.
Write the text very humanly.
Output Guidelines:
Deliver the narration as a single, uninterrupted block of text.
Exclude titles, subtitles, formatting instructions, or meta-comments.
Ensure the narrative flows smoothly, engages consistently, and fully delivers on the title's promise through a biblical lens.
'''
    else:
        base_prompt = f'''
Write a narration of {total_words} words based on the topic '{title}', specifically crafted to meet and exceed viewer expectations set by the title while keeping them engaged and curious throughout the video.

Instructions:
1. Start with Impact

Open with a powerful hook that immediately connects emotionally with the audience, addressing the core question or mystery posed by the title.
Introduce the topic in a way that reassures the viewer they'll gain valuable insights or revelations by continuing to watch.
Make it clear in the first 10 seconds that the viewer should continue watching.

2. Consistently Deliver Value

Expose curiosities about the theme.
Provide relevant insights, surprising facts, or captivating revelations frequently, ideally every 50-100 words.
Alternate between factual revelations, thought-provoking insights, and emotional hooks to maintain variety and engagement.

3. Maintain Curiosity with Mini-Hooks

Use mini-hooks every 100-150 words to sustain interest, ensuring they feel naturally integrated into the narrative:
Pose rhetorical questions that align with viewer curiosity.
Include teasers about upcoming twists or surprises.
Use shocking facts or statements to challenge conventional beliefs.

4. Build Toward a Rewarding Conclusion

Gradually escalate the narrative or insights toward the most impactful or surprising revelation.
Ensure the conclusion ties all elements together, providing both emotional and intellectual closure, and leaving the viewer with a sense of wonder or clear takeaway.

5. Stay Aligned with the Title

Ensure every part of the narration aligns with the promise, tone, and style of the title.
Avoid unnecessary tangents or overgeneralizations that detract from the topic.
Fully satisfy the curiosity generated by the title.

6. Use an Engaging and Conversational Tone

Write in a conversational tone that matches the intended audience's preferences, ensuring it feels natural, dynamic, and compelling when spoken aloud.
Make the viewer feel like they're part of an exciting journey of discovery.

7. Reflect the Viewer's Curiosity

Pose rhetorical questions sparingly but strategically to reflect the viewer's curiosity.
Use phrases like:
“You might be wondering...”
“Here's something you didn't expect…”
“What comes next will leave you speechless…”

Language and Word Count

Write the text in {language}, ensuring it meets or exceeds the {total_words} word count requirement.
Write the text very humanly.
Output Guidelines:
Deliver the narration as a single, uninterrupted block of text. Exclude titles, subtitles, formatting instructions, or meta-comments. Ensure the narrative flows smoothly, engages consistently, and fully delivers on the title's promise.
Avoid addressing topics that promote advocacy for LGBT ideologies or mystical practices, ensuring the content aligns with the intended values.
'''
    # --- Fim do prompt antigo ---

    # Criamos agora dois prompts para gerar metade + metade:
    # 1) Primeira metade (~1250 palavras)
    # 2) Segunda metade, continuando o texto anterior

    # Prompt da primeira solicitação (metade do texto)
    part1_prompt = f"""
{base_prompt}

Now, produce only the first half (about {half_words} words).
Make it clear that you will continue in a second part, but do NOT finalize or conclude the text yet.
Stop at a suitable transition point, leaving the story/argumentation open for continuation.
"""

    # Primeira solicitação
    response_part1 = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": part1_prompt}],
    )
    track_openai_tokens(response_part1) 
    part1_text = response_part1.choices[0].message['content'].strip()

    # Prompt da segunda solicitação (continuação)
    part2_prompt = f"""
You have already written the first half of the text:

{part1_text}

Now continue and produce the second half (about {half_words} words) to complete the {total_words}-word narration.
Ensure the text flows smoothly from the first half, picking up right where it left off.
Provide a proper conclusion that aligns with the previous instructions.
Do not repeat the first half text, only continue.
"""

    # Segunda solicitação
    response_part2 = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": part2_prompt}],
    )
    track_openai_tokens(response_part2)
    part2_text = response_part2.choices[0].message['content'].strip()

    # Junta as duas metades
    final_text = part1_text + "\n\n" + part2_text
    return final_text


def generate_video_with_options(language, channel_name, title, script_content):

    if language == 'english':
        tts_voice = 'en-US-TonyNeural'
    elif language == 'portuguese':
        tts_voice = 'pt-BR-DonatoNeural'
    elif language == 'spanish':
        tts_voice = 'es-ES-AlvaroNeural'
    elif language == 'german':
        tts_voice = 'de-DE-ChristophNeural'
    else:
        tts_voice = 'en-US-TonyNeural'

    global speech_config
    speech_config.speech_synthesis_voice_name = tts_voice

    sanitized_title = sanitize_filename(title).upper()

    main_dir = "videos"

    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    existing_folders = [f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))]
    folder_count = len(existing_folders) + 1

    current_date = datetime.now().strftime("%Y-%m-%d")

    if channel_name in [
        'The Bible Explored', 'Crime and Horror Stories',
        'Secrets of Money', 'Constant Evolution',
        'Explored Universe', 'Unveiled World',
        'Living History', 'Natural Health',
        'Uncomplicated Psychology'
    ]:
        dir_language = 'US'


     
    elif channel_name in [
        'La Biblia Explorada', 'Historias de Crimen y Terror',
        'Secretos del Dinero', 'Evolución Constante',
        'El Universo Explorado', 'Mundo Desvelado',
        'Historia Viva', 'Salud Natural',
        'Psicología Descomplicada'
    ]:
        dir_language = 'ES'

    elif channel_name in ['A Bíblia Explorada', 'A História Viva', 'O Universo Explorado']:
        dir_language = 'PT'

    else:
        dir_language = 'DE'

    video_content_dir = f'{dir_language}_{channel_name}_{sanitized_title}'

    video_dir = os.path.join(main_dir, video_content_dir)
    mp4_dir = os.path.join(main_dir, "videos_mp4")

    texts_dir = os.path.join(video_dir, "texts")
    images_dir = os.path.join(video_dir, "images")
    audio_dir = os.path.join(video_dir, "audio")

    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(mp4_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    # Salva o título
    save_to_file(os.path.join(texts_dir, "title.txt"), title)
 
    # Quebra o texto final em sentenças
    sentences = punkt_tokenizer.tokenize(script_content)

    # Prepara em chunks (por causa do limite ~4500 caracteres no TTS).
    sentence_chunks = split_sentences_into_chunks(sentences, max_characters=4500)

    ssml_chunks = [generate_ssml_with_bookmarks(chunk, tts_voice, channel_name) for chunk in sentence_chunks]

    all_phrases_path = os.path.join(texts_dir, "all_phrases.txt")
    with open(all_phrases_path, "w", encoding="utf-8") as file:
        for sentence in sentences:
            file.write(sentence + "\n")
    print(f"All phrases saved")

    full_content_path = os.path.join(texts_dir, "full_content.txt")
    with open(full_content_path, "w", encoding="utf-8") as file:
        file.write(script_content)
    print(f"Full content saved")

    ssml_text = generate_ssml_with_bookmarks(sentences, tts_voice, channel_name)

    audio_filename = os.path.join(audio_dir, "full_audio_with_bookmarks.mp3")
    bookmark_timings = text_to_audio_with_bookmarks(ssml_chunks, audio_filename)

    # Adiciona 2s de silêncio no início
    final_audio = AudioSegment.silent(duration=200) + AudioSegment.from_mp3(audio_filename)
    final_audio.export(audio_filename, format="mp3")

    # Calcula intervalos de cada sentença
    sentence_timings = []
    for idx, bookmark in enumerate(bookmark_timings):
        start_time = bookmark['offset']
        if idx + 1 < len(bookmark_timings):
            end_time = bookmark_timings[idx + 1]['offset']
        else:
            audio = AudioSegment.from_mp3(audio_filename)
            end_time = len(audio)

        sentence_index = int(bookmark['bookmark'].split('_')[1]) - 1

        if sentence_index >= len(sentences) or sentence_index < 0:
            print(f"Warning: Bookmark {bookmark['bookmark']} points to invalid index {sentence_index}. Skipping.")
            continue

        sentence_timings.append({
            'sentence': sentences[sentence_index],
            'start_time': start_time + 1000,
            'end_time': end_time + 1000
        })

    # Gera SRT
    srt_lines = []
    for idx, entry in enumerate(sentence_timings):
        start_srt_time = milliseconds_to_srt_time(entry['start_time'])
        end_srt_time = milliseconds_to_srt_time(entry['end_time'])
        srt_lines.append(f"""{idx+1}
{start_srt_time} --> {end_srt_time}
{entry['sentence']}
""")

    srt_content = ''.join(srt_lines)
    srt_file_path = os.path.join(texts_dir, "subtitles.srt")
    if not srt_content.strip():
        print("Error: No content generated for .srt file.")
    else:
        with open(srt_file_path, "w", encoding="utf-8") as file:
            file.write(srt_content)
        print(f"Subtitles saved to {srt_file_path}")

    # Gera prompts para imagens
    phrases = sentences
    image_prompts = generate_image_prompts(title, phrases, channel_name)

    with open(os.path.join(texts_dir, "all_prompts.txt"), "w", encoding="utf-8") as f:
        for idx, (phrase_num, prompt_text) in enumerate(image_prompts):
            f.write(f"Phrase {phrase_num}:\n{prompt_text}\n\n")

    args_list = []
    for idx, (phrase_num, prompt_text) in enumerate(image_prompts):
        image_index = phrase_num
        args_list.append((idx, phrase_num, prompt_text, images_dir, image_index))

    print("Starting asynchronous image generation...")
    asyncio.run(process_images_async(args_list, images_dir))
    print("Asynchronous image generation complete.")

    # Gera o vídeo
    video_output_path = os.path.join(mp4_dir, f"{dir_language}_{channel_name}_{sanitized_title}.mp4")
    soundtrack_folder = get_soundtrack_folder(channel_name)
    generate_video(sentence_timings, images_dir, audio_filename, video_output_path, soundtrack_folder)
    print(f"Video saved at {video_output_path}")

    print("All tasks completed successfully.")


def main():
    number_of_videos = int(input("How many videos? "))
    video_tasks = []
    for video_index in range(1, number_of_videos + 1):
        print(f"\nVideo {video_index}")
        language_options = {'1': 'english', '2': 'spanish', '3': 'portuguese', '4': 'german'}
        print("Choose language:")
        for key, lang in language_options.items():
            print(f"{key}. {lang.capitalize()}")
        language_choice = input("Enter the number of your choice: ").strip()
        language = language_options.get(language_choice, 'english')

        if language == 'english':
            channels = [
                'The Bible Explored', 
                'Unveiled World',
                'Explored Universe', 
                'Living History',
                'Crime and Horror Stories'
            ]
        elif language == 'spanish':
            channels = [
                'La Biblia Explorada', 
                'Mundo Desvelado',
                'El Universo Explorado', 
                'Historia Viva',
            ]
        elif language == 'portuguese':
            channels = [
                'A Bíblia Explorada',
                'O Universo Explorado',
                'A História Viva',
            ]
        elif language == 'german':
            channels = [
                'Lebendige Geschichte',
                'Erforschtes Universum'
            ]
        else:
            channels = [

            ]

        print("Choose a channel:")
        for idx, channel in enumerate(channels, start=1):
            print(f"{idx}- {channel}")
        channel_choice = int(input("Choose a channel by number: "))
        channel_name = channels[channel_choice - 1]

        #title_idea = input('Title idea: ')
        #titles = get_titles(title_idea, language=language, max_attempts=3)

        #print("Choose a title:")
        #for idx, title in enumerate(titles, start=1):
        #    print(f"{idx}. {title}")
        #title_choice = int(input("Choose a title by number: "))
        #title = titles[title_choice - 1]
        title = input('Choose a title: ')

        video_tasks.append({
            'language': language,
            'channel_name': channel_name,
            'title': title,
        })

    print("\n--- Planning Completed ---")
    print("All videos will now be generated.")

    for task in video_tasks:
        print(f"\n--- Generating Video: {task['title']} ---")
        # Gera o roteiro em duas chamadas
        full_content = generate_content_in_two_calls_with_old_prompt(
            title=task['title'],
            language=task['language'],
            channel_name=task['channel_name']
        )

        # Faz todo o pipeline de geração de vídeo
        generate_video_with_options(
            language=task['language'],
            channel_name=task['channel_name'],
            title=task['title'],
            script_content=full_content,
        )

    # Gera relatório de custos e tempo
    generate_report()
    print("All videos generated successfully!")


if __name__ == '__main__':
    main()