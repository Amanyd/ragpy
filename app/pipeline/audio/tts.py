import io
import soundfile as sf
import numpy as np
from kokoro import KPipeline

tts_pipeline = None

def get_tts_pipeline():
    global tts_pipeline
    if tts_pipeline is None:
        try:
            tts_pipeline = KPipeline(lang_code='h') # Using Hindi as requested earlier, or 'a' for American 
        except Exception as e:
            print(f"Failed to load Kokoro TTS pipeline: {e}")
            raise RuntimeError(f"TTS Init Error: {e}")
    return tts_pipeline

def generate_speech(text: str, voice: str = 'hm_omega', speed: float = 1.0) -> bytes:
    pipeline = get_tts_pipeline()
    if not pipeline:
        raise RuntimeError("Kokoro TTS pipeline is not initialized.")
        
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
    audio_chunks = []
    
    for i, (gs, ps, audio) in enumerate(generator):
        if audio is not None:
            audio_chunks.append(audio)
            
    if not audio_chunks:
        return b""
        
    final_audio = np.concatenate(audio_chunks)
    out_io = io.BytesIO()
    sf.write(out_io, final_audio, 24000, format='WAV')
    
    return out_io.getvalue()
