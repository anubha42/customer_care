from flask import Flask, request, send_file, jsonify
from flask_socketio import SocketIO
import os
import torch
import torchaudio
from transformers import VitsModel, AutoTokenizer
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
import unicodedata
import aiohttp
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize models and move to GPU
@torch.no_grad()
def load_models():
    global model, tokenizer
    print("Loading models...")
    model = VitsModel.from_pretrained("facebook/mms-tts-hin").to(device)
    model.eval()  # Set model to evaluation mode
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
    print("Models loaded successfully!")

# Call the model loading function
load_models()

# OpenAI API settings
HEADERS = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
BASE_URL = "https://api.openai.com/v1"

# System prompt
SYSTEM_PROMPT = (
    "You are a casual Hindi speaking customer calling customer care for a telecom service. "
    "Your complaint is that your connection has been cut even after you have paid the monthly bill. "
    "You speak Hindi with a polite but slightly frustrated tone. "
    "Keep your responses concise and stay in character as a customer."
    "If asked numbers, write the number as a word."
)

# Conversation messages
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# Create a thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=2)

async def transcribe_audio(audio_data):
    """Transcribe audio using Whisper API."""
    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field("model", "whisper-1")
        form.add_field("response_format", "text")
        form.add_field("file", audio_data, filename="audio.wav")
        
        async with session.post(f"{BASE_URL}/audio/transcriptions", headers=HEADERS, data=form) as resp:
            if resp.status != 200:
                error_message = await resp.text()
                raise Exception(f"Transcription error: {error_message}")
            transcription = await resp.text()
            print(f"Transcription: {transcription}")  # Debug: Print transcription
            return transcription

async def query_llm(messages):
    """Query GPT-4 for response."""
    async with aiohttp.ClientSession() as session:
        data = {"model": "gpt-4", "messages": messages}
        async with session.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=data) as resp:
            response = await resp.json()
            customer_text = response['choices'][0]['message']['content']
            print(f"Generated LLM Response (before transliteration): {customer_text}")  # Debug: Print raw LLM response
            tr_customer_text = UnicodeIndicTransliterator.transliterate(customer_text, "en", "hi")
            normalized_text = unicodedata.normalize('NFC', tr_customer_text)
            print(f"Generated LLM Response : {tr_customer_text}")  # Debug: Print transliterated response
            return tr_customer_text

# Generate TTS audio
@torch.no_grad()  # Disable gradient computation for inference
def tts(text, filename="customer_reply.wav"):
    """Convert text to speech using GPU acceleration."""
    if not text.strip():
        raise ValueError("Empty input text")
    
    try:
        # Tokenize text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["input_ids"] = inputs["input_ids"].long()
        
        # Generate audio
        with torch.cuda.amp.autocast():  # Enable automatic mixed precision
            waveform = model(**inputs).waveform
        
        # Move waveform back to CPU for saving
        waveform = waveform.cpu()
        
        # Save audio file explicitly setting the format to 'wav'
        torchaudio.save(filename, waveform, 16000, format='wav')  # Specify format explicitly
        print(f"TTS audio saved to: {filename}")  # Debug: Print TTS output file path
        return filename
    
    except Exception as e:
        print(f"Error in TTS generation: {str(e)}")
        raise
    
# Cache directory for temporary files
CACHE_DIR = "audio_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

@app.route('/process_audio', methods=['POST'])
async def process_audio():
    """Handle audio processing request with GPU acceleration."""
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['file']
    
    try:
        # Generate unique filename for this request
        cache_filename = f"{CACHE_DIR}/response_{np.random.randint(1, 100000)}.wav"
        
        # Transcribe audio
        transcription = await transcribe_audio(audio_file.read())
        messages.append({"role": "user", "content": transcription})
        
        # Get LLM response
        customer_response = await query_llm(messages)
        messages.append({"role": "assistant", "content": customer_response})
        
        # Generate TTS using GPU
        loop = asyncio.get_event_loop()
        output_file = await loop.run_in_executor(thread_pool, tts, customer_response, cache_filename)
        
        return send_file(output_file, mimetype='audio/wav')
    
    except Exception as e:
        print(f"Error in process_audio: {str(e)}")  # Debug: Print error
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Cleanup temporary files
        try:
            if os.path.exists(cache_filename):
                os.remove(cache_filename)
        except:
            pass

# GPU memory management
def cleanup_gpu_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# Register cleanup function
@app.before_request
def before_request():
    cleanup_gpu_memory()

if __name__ == '__main__':
    # Set GPU memory growth
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use up to 80% of GPU memory

    socketio.run(app, host='0.0.0.0', port=3500, debug=True)