from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from elevenlabs.client import ElevenLabs
import os
from dotenv import load_dotenv
import tempfile
import base64

# Load environment variables
load_dotenv()

# Initialize OpenAI and ElevenLabs clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

def text_to_speech(text: str) -> bytes:
    """Convert text to speech using ElevenLabs"""
    try:
        audio_gen = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id="cjVigY5qzO86Huf0OWal",  # You can change this voice ID as needed
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        
        audio_bytes = b''
        for chunk in audio_gen:
            audio_bytes += chunk
        
        return audio_bytes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text to speech error: {str(e)}")

@app.post("/api/chat")
async def chat(message: Message):
    try:
        # Get GPT response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": message.text}
            ]
        )
        response_text = response.choices[0].message.content
        
        # Convert response to speech
        audio_bytes = text_to_speech(response_text)
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        return {
            "response": response_text,
            "audio": audio_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        # Read the file content
        content = await audio.read()
        
        # Create a temporary file with the correct extension
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Transcribe using OpenAI Whisper API
            with open(temp_path, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            # Get GPT response for the transcribed text
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": transcript.text}
                ]
            )
            response_text = response.choices[0].message.content
            
            # Convert response to speech
            audio_bytes = text_to_speech(response_text)
            audio_base64 = base64.b64encode(audio_bytes).decode()

            return {
                "text": transcript.text,
                "response": response_text,
                "audio": audio_base64
            }
        finally:
            # Clean up temp file
            os.unlink(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)