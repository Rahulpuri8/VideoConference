import os
import cv2
import numpy as np
import subprocess
from config import AUDIO_SAMPLE_RATE, VAD_FRAME_MS, VIDEO_FPS, CODEC_PRIORITY

def extract_audio(video_path, output_path="temp_audio/audio.wav"):
    """Robust audio extraction with proper format for VAD"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', video_path,
        '-ac', '1',
        '-ar', str(AUDIO_SAMPLE_RATE),
        '-acodec', 'pcm_s16le',
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path

def prepare_audio_frames(audio, sample_rate):
    """Convert audio to VAD-compatible frames"""
    frame_size = int(sample_rate * VAD_FRAME_MS / 1000)
    audio = (audio * 32767).astype(np.int16)
    
    frames = []
    for i in range(0, len(audio), frame_size):
        frame = audio[i:i+frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        frames.append(frame)
    return frames

def calculate_frames_per_audio_frame(video_fps, audio_frame_ms):
    """Calculate synchronization ratio"""
    audio_frame_duration = audio_frame_ms / 1000
    return max(1, int(video_fps * audio_frame_duration))

def create_video_writer(output_path, fps, width, height):
    """Try multiple codecs until finding a working one"""
    for codec in CODEC_PRIORITY:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
        if out.isOpened():
            print(f"Using codec: {codec}")
            return out
    raise RuntimeError(f"Failed to initialize VideoWriter with codecs: {CODEC_PRIORITY}")

def zoom_face(frame, face_box, zoom=1.8):
    """Smooth zoom effect with boundary checks"""
    x, y, w, h = face_box
    cx, cy = x + w//2, y + h//2
    new_size = int(max(w, h) * zoom)
    
    x1 = max(0, cx - new_size//2)
    y1 = max(0, cy - new_size//2)
    x2 = min(frame.shape[1], cx + new_size//2)
    y2 = min(frame.shape[0], cy + new_size//2)
    
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (frame.shape[1], frame.shape[0]), 
                     interpolation=cv2.INTER_LINEAR)

def verify_video_file(path):
    """Check if video file is valid and playable"""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 1024:
        return False
    
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return False
    
    ret, _ = cap.read()
    cap.release()
    return ret
