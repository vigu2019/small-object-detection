import os

# VIDEO SETTINGS
VIDEO_PATH = 0
OUTPUT_PATH = "video_data/outputs/detected_video_alert.mp4"

# DETECTION SETTINGS
MODEL_PATH = "yolov8s.pt"
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection
TARGET_CLASSES = ["person"]  # Classes to trigger alerts (e.g., ["person", "car"])

# ZONE CONFIGURATION (as percentage of frame: 0.0 to 1.0)
ZONE_X1 = 0.25  # Left boundary (25% from left)
ZONE_Y1 = 0.25  # Top boundary (25% from top)
ZONE_X2 = 0.75  # Right boundary (75% from left)
ZONE_Y2 = 0.75  # Bottom boundary (75% from top)

# Zone visualization
ZONE_COLOR = (0, 0, 255)  # Red in BGR
ZONE_THICKNESS = 3

# ALERT TRIGGER SETTINGS
PERSISTENCE_FRAMES = 5  # Object must stay in zone for N consecutive frames
COOLDOWN_SECONDS = 10   # Minimum seconds between alerts

# ALERT MECHANISMS
ENABLE_SCREENSHOT = True
ENABLE_AUDIO_BEEP = True
ENABLE_EMAIL = True  
ENABLE_CSV_LOG = True


# FILE PATHS
SCREENSHOT_DIR = "alerts/screenshots"
CSV_LOG_PATH = "alerts/intrusion_log.csv"


# AUDIO SETTINGS (Windows)
BEEP_FREQUENCY = 1000  # Hz
BEEP_DURATION = 500    # milliseconds

# EMAIL CONFIGURATION
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "campusease25@gmail.com"
SENDER_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD")
RECIPIENT_EMAILS = ["campusease25@gmail.com","nehazzrajesh@gmail.com"]

# Email content
EMAIL_SUBJECT = "Intrusion Alert - Restricted Zone Breach"
EMAIL_BODY_TEMPLATE = """
<html>
<body>
    <h2 style="color: #d32f2f;">Intrusion Alert</h2>
    <p><strong>Timestamp:</strong> {timestamp}</p>
    <p><strong>Detected Class:</strong> {class_name}</p>
    <p><strong>Confidence:</strong> {confidence:.2%}</p>
    <p><strong>Zone:</strong> ({zone_x1}, {zone_y1}) to ({zone_x2}, {zone_y2})</p>
    <p><strong>Persistence:</strong> Object remained in zone for {frames} consecutive frames</p>
    <hr>
    <p style="color: #666;">This is an automated alert from your intrusion detection system.</p>
</body>
</html>
"""
