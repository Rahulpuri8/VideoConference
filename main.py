import os
import cv2
import numpy as np
import librosa
import webrtcvad
from tqdm import tqdm
import mediapipe as mp
from utils import *
from config import *

def initialize_models():
    """Initialize face detection and VAD"""
    mp_face = mp.solutions.face_detection
    return (
        mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7  # Increased for better accuracy
        ),
        webrtcvad.Vad(3)  # Aggressive mode
    )

def process_video(input_path, output_path, zoom_factor=1.8):
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Initialize models
    face_detector, vad = initialize_models()
    
    # Audio processing
    audio_path = extract_audio(input_path)
    audio, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)
    audio_frames = prepare_audio_frames(audio, AUDIO_SAMPLE_RATE)
    
    # Video processing setup
    cap = cv2.VideoCapture(input_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = actual_fps if actual_fps > 0 else VIDEO_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    out = create_video_writer(output_path, fps, width, height)
    
    # Calculate frame synchronization
    frames_per_audio_frame = calculate_frames_per_audio_frame(fps, VAD_FRAME_MS)
    
    # Processing loop
    prev_speaker = None
    try:
        with tqdm(total=total_frames, desc="Processing") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Ensure frame is in BGR format
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Get corresponding audio frame
                audio_frame_idx = frame_idx // frames_per_audio_frame
                is_speech = False
                if audio_frame_idx < len(audio_frames):
                    try:
                        is_speech = vad.is_speech(
                            audio_frames[audio_frame_idx].tobytes(),
                            AUDIO_SAMPLE_RATE
                        )
                    except Exception as e:
                        print(f"VAD error: {e}")
                
                # Face detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detector.process(rgb_frame)
                
                active_speaker = None
                if results.detections and is_speech:
                    # Find most central face
                    min_distance = float('inf')
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        cx = bbox.xmin + bbox.width/2 - 0.5
                        cy = bbox.ymin + bbox.height/2 - 0.5
                        distance = cx**2 + cy**2
                        
                        if distance < min_distance:
                            min_distance = distance
                            active_speaker = (
                                int(bbox.xmin * width),
                                int(bbox.ymin * height),
                                int(bbox.width * width),
                                int(bbox.height * height)
                            )
                
                # Apply zoom
                if active_speaker:
                    frame = zoom_face(frame, active_speaker, zoom_factor)
                    prev_speaker = active_speaker
                elif prev_speaker:
                    frame = zoom_face(frame, prev_speaker, zoom_factor)
                
                out.write(frame)
                pbar.update(1)
    finally:
        # Ensure resources are released
        out.release()
        cap.release()
        cv2.destroyAllWindows()
    
    # Verify output
    if not verify_video_file(output_path):
        raise RuntimeError("Failed to create valid output video")
    print(f"Successfully created output: {output_path}")

if __name__ == "__main__":
    input_video = "input_videos/zoom_sample (1).mp4"
    output_video = "output_videos/output.mp4"
    
    # Clear previous output if exists
    if os.path.exists(output_video):
        os.remove(output_video)
    
    process_video(input_video, output_video)
