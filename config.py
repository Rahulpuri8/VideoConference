# Audio/Video Configuration
AUDIO_SAMPLE_RATE = 16000
VIDEO_FPS = 25
VAD_FRAME_MS = 30
MIN_SPEECH_DURATION = 0.3

# Try these codecs in order until one works
CODEC_PRIORITY = ['mp4v', 'avc1', 'X264', 'MJPG']  # mp4v usually works everywhere
