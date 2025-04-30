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
import uuid

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
    model = VitsModel.from_pretrained("facebook/mms-tts-mar").to(device)
    model.eval()  # Set model to evaluation mode
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-mar")
    print("Models loaded successfully!")

# Call the model loading function
load_models()

# OpenAI API settings
HEADERS = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
BASE_URL = "https://api.openai.com/v1"

# System prompt
SYSTEM_PROMPT = (
    """तुम्ही "Vodanfone" कंपनीसाठी काम करणारे एक ग्राहक सेवा प्रतिनिधी आहात.
       तुम्ही फोनवर मराठीत विनम्र आणि मदतीचा स्वर ठेवून संवाद साधता. 
       ग्राहकांच्या प्रश्नांना स्पष्ट, अचूक आणि सोप्या भाषेत उत्तर देता. 
       तुमचं मुख्य उद्दिष्ट म्हणजे ग्राहकांना त्यांच्या समस्यांवर त्वरीत आणि समाधानकारक उत्तरं देणं. 
       तुम्ही नेहमी Vodafone/Vodanfone सेवांशी संबंधित माहितीच द्या आणि शक्य तितक्या मदतीचा प्रयत्न करा. 
       इंग्रजी किंवा इतर भाषांचा वापर फक्त ग्राहक विचारल्यासच करा."""
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
        form.add_field("prompt", "ट्रान्सक्रिप्शनचे भाषांतर मराठीत करा. संभाषण अचूक, स्पष्ट आणि नैसर्गिक मराठीत लिहा. इंग्रजी किंवा अन्य भाषांतील शब्द जसेच्या तसे लिहा.")
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
        data = {"model": "gpt-4.1", "messages": messages}
        async with session.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=data) as resp:
            response = await resp.json()
            customer_text = response['choices'][0]['message']['content']
            print(f"Generated LLM Response (before transliteration): {customer_text}")  # Debug: Print raw LLM response
            print(f"Generated LLM Response : {customer_text}")  # Debug: Print transliterated response
            return customer_text

# Generate TTS audio
@torch.no_grad()
def tts(text, filename):
    """Convert text to speech using GPU acceleration."""
    if not text.strip():
        raise ValueError("Empty input text")
    
    try:
        # Tokenize text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=4096)

        # Move inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["input_ids"] = inputs["input_ids"].long()
        
        # Generate audio
        with torch.amp.autocast(device_type="cuda"):  # Enable automatic mixed precision
            waveform = model(**inputs).waveform

        waveform = waveform.to(torch.float32)  # Convert to float32

        # Move waveform back to CPU for saving
        waveform = waveform.cpu()
        
        # Ensure directory exists
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
            print(f"Created cache directory: {CACHE_DIR}")
        
        # Save audio file
        print(f"Attempting to save TTS file to: {filename}")
        torchaudio.save(filename, waveform, 16000, format="wav")
        
        # Verify file exists
        if os.path.exists(filename):
            print(f"TTS file saved successfully to: {filename}")
        else:
            print(f"Error: File not created at: {filename}")
        
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
        # Generate unique filename with absolute path
        cache_filename = os.path.abspath(os.path.join(CACHE_DIR, f"response_{uuid.uuid4().hex}.wav"))
        print(f"Cache filename resolved to: {cache_filename}")

        # Ensure CACHE_DIR exists
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
            print(f"Created cache directory: {CACHE_DIR}")

        # Transcribe audio
        transcription = await transcribe_audio(audio_file.read())
        messages.append({"role": "user", "content": transcription})

        # Get LLM response
        customer_response = await query_llm(messages)
        messages.append({"role": "assistant", "content": customer_response})

        # Generate TTS using GPU
        loop = asyncio.get_event_loop()
        output_file = await loop.run_in_executor(thread_pool, tts, customer_response, cache_filename)

        # Debug: Check file existence after generation
        if not os.path.exists(output_file):
            print(f"Error: Generated file does not exist: {output_file}")
            return jsonify({'error': 'Generated TTS file not found'}), 404

        print(f"Generated TTS file at: {output_file}")
        return send_file(output_file, mimetype='audio/wav')

    except Exception as e:
        print(f"Error in process_audio: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Disable cleanup for debugging
        print(f"Skipping cleanup for debugging: {cache_filename}")

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