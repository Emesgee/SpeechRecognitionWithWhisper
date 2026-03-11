import os
import requests
import subprocess
import librosa
import soundfile as sf
import sounddevice as sd
from scipy.io.wavfile import write
import noisereduce as nr
from faster_whisper import WhisperModel


# ---------------------------------------------------------
# 1. UNIVERSAL AUDIO LOADER (local file, remote URL, mic)
# ---------------------------------------------------------
def load_audio_source(source):
    """
    Accepts:
    - Local file path
    - Remote audio URL
    - Keyword 'mic' to record from microphone
    Returns a guaranteed valid filename.
    """

    # --- Microphone recording ---
    if source.lower() == "mic":
        print("Recording from microphone (5 seconds)...")
        fs = 16000
        duration = 5
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write("mic.wav", fs, recording)
        return "mic.wav"

    # --- Remote audio URL ---
    if source.startswith("http"):
        print("Downloading remote audio file...")
        r = requests.get(source, allow_redirects=True)

        if r.status_code != 200:
            raise ValueError(f"Failed to download remote audio. HTTP status: {r.status_code}")

        content_type = r.headers.get("Content-Type", "")

        # Validate that the URL actually returned audio
        if "audio" not in content_type:
            raise ValueError(f"URL did not return audio data. Content-Type was: {content_type}")

        ext = content_type.split("/")[-1]
        filename = f"remote_audio.{ext}"

        with open(filename, "wb") as f:
            f.write(r.content)

        print(f"Downloaded remote audio: {filename}")
        return filename

    # --- Local file ---
    if os.path.exists(source):
        print("Using local audio file...")
        return source

    raise ValueError(f"Invalid audio source: {source}")


# ---------------------------------------------------------
# 2. CONVERT ANY FILE TO WAV (FFmpeg)
# ---------------------------------------------------------
def convert_to_wav(path):
    """
    Converts any audio/video file to WAV (16 kHz mono).
    Returns the new WAV filename.
    """
    print("Converting to WAV...")

    base = os.path.splitext(path)[0]
    output = f"{base}_converted.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", path,
        "-ac", "1",
        "-ar", "16000",
        output
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print("FFmpeg error:")
        print(result.stderr.decode())
        raise RuntimeError("FFmpeg failed to convert the file.")

    if not os.path.exists(output):
        raise FileNotFoundError("FFmpeg did not produce the WAV file.")

    print(f"Converted to WAV: {output}")
    return output


# ---------------------------------------------------------
# 3. CLEAN AUDIO
# ---------------------------------------------------------
def clean_audio(path):
    print("Cleaning audio...")

    y, sr = librosa.load(path, sr=16000)

    y_reduced = nr.reduce_noise(y=y, sr=sr)
    y_norm = librosa.util.normalize(y_reduced)

    output = "cleaned.wav"
    sf.write(output, y_norm, sr)

    print("Cleaned audio saved as cleaned.wav")
    return output


# ---------------------------------------------------------
# 4. TRANSCRIBE WITH FASTER-WHISPER
# ---------------------------------------------------------
def transcribe_audio(path):
    print("Transcribing with faster-whisper...")
    model = WhisperModel("base", device="cpu")

    segments, info = model.transcribe(path)
    text = " ".join([seg.text for seg in segments])

    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("Transcript saved to transcript.txt")
    return text


# ---------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------
def run_pipeline():
    source = input("Enter audio source (file path, remote URL, or 'mic'): ")

    audio_path = load_audio_source(source)
    wav_path = convert_to_wav(audio_path)
    cleaned = clean_audio(wav_path)
    text = transcribe_audio(cleaned)

    print("\n--- TRANSCRIPTION COMPLETE ---\n")
    print(text)


# Run the pipeline
if __name__ == "__main__":
    run_pipeline()

