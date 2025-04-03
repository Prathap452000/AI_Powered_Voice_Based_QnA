import os
import re
import tempfile
import threading
from datetime import datetime
from dotenv import load_dotenv

import google.generativeai as genai
import pygame
import speech_recognition as sr
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from gtts import gTTS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-here")

# Initialize SocketIO with eventlet
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Initialize Pygame mixer
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# Global conversation history
conversation_history = []

def clean_text_for_speech(text):
    """Clean text for natural sounding speech output"""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italics
    text = re.sub(r'```[\s\S]*?```', '', text)    # Remove code blocks
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Remove markdown links
    text = re.sub(r'\n{2,}', '. ', text)          # Replace multiple newlines
    text = re.sub(r'\n', ' ', text)               # Replace single newlines
    return text.strip()

def speak_response(text):
    """Convert text to speech and play using Pygame"""
    cleaned_text = clean_text_for_speech(text)
    temp_file_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts = gTTS(text=cleaned_text, lang='en')
            tts.save(temp_audio_file.name)
            temp_file_path = temp_audio_file.name
        
        pygame.mixer.music.load(temp_file_path)
        pygame.mixer.music.play()

        # Wait for playback to finish
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(30)
            
    except Exception as e:
        print(f"Error playing audio: {e}")
        emit('error', {'message': f'Audio playback error: {str(e)}'})
    finally:
        # Add a small delay before deleting the file to ensure it's no longer in use
        pygame.time.wait(100)  # Wait 100ms
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except PermissionError:
                # If file is still in use, we'll just leave it
                # The OS will clean up temporary files eventually
                pass

def generate_intelligent_response(user_input):
    """Generate thoughtful, intelligent responses (2-3 lines)"""
    global conversation_history
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    conversation_history.append({'role': 'user', 'text': user_input, 'timestamp': timestamp})
    
    # Create context-aware prompt for intelligent responses
    context = "\n".join([f"{msg['role']}: {msg['text']}" for msg in conversation_history[-3:]])
    
    prompt = f"""You are an intelligent AI assistant. Provide thoughtful, concise responses in 2 or less lines.
    Consider the context and provide accurate, insightful answers.
    Be professional yet approachable in your tone.
    Before the answer use filler words that are natural to a human and contextually appropriate.
    
    Conversation context:
    {context}
    
    User's question/statement: "{user_input}"
    
    Your intelligent response (2-3 lines maximum):"""
    
    response = model.generate_content(prompt)
    ai_text = response.text
    
    # Ensure response is concise (2-3 lines max)
    ai_text = "\n".join(ai_text.split("\n")[:3]).strip()
    
    conversation_history.append({'role': 'assistant', 'text': ai_text, 'timestamp': timestamp})
    return ai_text, timestamp

@app.route("/")
def home():
    return render_template("index.html")

@socketio.on('start_listening')
def handle_listening():
    """Handle voice input via WebSocket"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            emit('listening_status', {'status': 'active'})
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # Convert speech to text
            user_input = recognizer.recognize_google(audio)
            emit('user_input', {'text': user_input, 'timestamp': datetime.now().strftime("%H:%M:%S")})
            
            # Get AI response
            ai_response, response_timestamp = generate_intelligent_response(user_input)
            emit('ai_response', {'text': ai_response, 'timestamp': response_timestamp})
            
            # Play audio in background thread
            threading.Thread(
                target=speak_response,
                args=(ai_response,),
                daemon=True
            ).start()
            
    except sr.WaitTimeoutError:
        emit('error', {'message': 'No speech detected'})
    except sr.UnknownValueError:
        emit('error', {'message': 'Could not understand audio'})
    except Exception as e:
        emit('error', {'message': f'Error: {str(e)}'})
    finally:
        emit('listening_status', {'status': 'inactive'})

if __name__ == "__main__":
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)