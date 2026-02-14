"""
Zone-Based Intrusion Detection System
Extends YOLOv8 video inference with intelligent alert system
"""

import cv2
import csv
import os
import time
import winsound
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


import config_alert as config


def ensure_directories():
    """Create necessary directories if they don't exist"""
    Path(config.SCREENSHOT_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.CSV_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(config.CSV_LOG_PATH):
        with open(config.CSV_LOG_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Class', 'Confidence', 
                'Zone_Coords', 'Screenshot_Path', 'Alert_Type'
            ])


def get_box_center(box):
    """Calculate center point of bounding box"""
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return int(center_x), int(center_y)


def is_point_in_zone(point, zone):
    """Check if point is inside zone rectangle"""
    px, py = point
    zx1, zy1, zx2, zy2 = zone
    return zx1 <= px <= zx2 and zy1 <= py <= zy2


def save_screenshot(frame, class_name, confidence):
    """Save current frame as screenshot with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{class_name}_{confidence:.2f}.jpg"
    filepath = os.path.join(config.SCREENSHOT_DIR, filename)
    cv2.imwrite(filepath, frame)
    print(f"📸 Screenshot saved: {filepath}")
    return filepath


def play_alert_beep():
    """Play Windows beep sound"""
    try:
        winsound.Beep(config.BEEP_FREQUENCY, config.BEEP_DURATION)
        print("🔔 Alert beep played")
    except Exception as e:
        print(f"⚠️ Could not play beep: {e}")


def send_email_alert(class_name, confidence, zone_coords, screenshot_path, frames):
    """Send email notification with screenshot attachment"""
    if not config.ENABLE_EMAIL:
        return
    
    if not config.SENDER_EMAIL or not config.RECIPIENT_EMAILS:
        print("⚠️ Email not configured. Skipping email alert.")
        return
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = config.SENDER_EMAIL
        msg['To'] = ', '.join(config.RECIPIENT_EMAILS)
        msg['Subject'] = config.EMAIL_SUBJECT
        
        # Email body
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = config.EMAIL_BODY_TEMPLATE.format(
            timestamp=timestamp,
            class_name=class_name,
            confidence=confidence,
            zone_x1=zone_coords[0],
            zone_y1=zone_coords[1],
            zone_x2=zone_coords[2],
            zone_y2=zone_coords[3],
            frames=frames
        )
        msg.attach(MIMEText(body, 'html'))
        
        # Attach screenshot
        if os.path.exists(screenshot_path):
            with open(screenshot_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', 
                             filename=os.path.basename(screenshot_path))
                msg.attach(img)
        
        # Send email
        with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SENDER_EMAIL, config.SENDER_PASSWORD)
            server.send_message(msg)
        
        print(f"📧 Email sent to {', '.join(config.RECIPIENT_EMAILS)}")
    
    except Exception as e:
        print(f"⚠️ Failed to send email: {e}")


def log_to_csv(class_name, confidence, zone_coords, screenshot_path):
    """Log intrusion event to CSV file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    zone_str = f"({zone_coords[0]},{zone_coords[1]})-({zone_coords[2]},{zone_coords[3]})"
    
    with open(config.CSV_LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, class_name, f"{confidence:.3f}",
            zone_str, screenshot_path, "Zone_Intrusion"
        ])
    
    print(f"📝 Event logged to CSV: {config.CSV_LOG_PATH}")


def trigger_alert(frame, class_name, confidence, zone_coords, frames_in_zone):
    """Execute all alert mechanisms"""
    print("\n" + "="*60)
    print(f"ALERT TRIGGERED - {class_name.upper()} DETECTED IN RESTRICTED ZONE")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Persistence: {frames_in_zone} consecutive frames")
    print("="*60)
    
    screenshot_path = ""
    
    # Save screenshot
    if config.ENABLE_SCREENSHOT:
        screenshot_path = save_screenshot(frame, class_name, confidence)
    
    # Play audio beep
    if config.ENABLE_AUDIO_BEEP:
        play_alert_beep()
    
    # Send email
    if config.ENABLE_EMAIL:
        send_email_alert(class_name, confidence, zone_coords, screenshot_path, frames_in_zone)
    
    # Log to CSV
    if config.ENABLE_CSV_LOG:
        log_to_csv(class_name, confidence, zone_coords, screenshot_path)
    
    print()


def draw_zone(frame, zone, color, thickness):
    """Draw zone rectangle on frame"""
    x1, y1, x2, y2 = zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Add zone label
    label = "RESTRICTED ZONE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    
    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, font_thickness
    )
    
    # Draw background rectangle for text
    cv2.rectangle(
        frame,
        (x1, y1 - text_height - 10),
        (x1 + text_width + 10, y1),
        color,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        label,
        (x1 + 5, y1 - 5),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness
    )



def main():
    """Main video processing loop with intrusion detection"""
    
    # Setup
    ensure_directories()
    print("Starting Zone-Based Intrusion Detection System")
    print(f"Video: {config.VIDEO_PATH}")
    print(f"Target classes: {config.TARGET_CLASSES}")
    print(f"Persistence required: {config.PERSISTENCE_FRAMES} frames")
    print(f"Cooldown period: {config.COOLDOWN_SECONDS} seconds")
    print()
    
    # Load model
    model = YOLO(config.MODEL_PATH)
    
    # Open video
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {config.VIDEO_PATH}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate zone coordinates in pixels
    zone_x1 = int(config.ZONE_X1 * frame_width)
    zone_y1 = int(config.ZONE_Y1 * frame_height)
    zone_x2 = int(config.ZONE_X2 * frame_width)
    zone_y2 = int(config.ZONE_Y2 * frame_height)
    zone_coords = (zone_x1, zone_y1, zone_x2, zone_y2)
    
    print(f"📐 Frame size: {frame_width}x{frame_height}")
    print(f"🔲 Zone coordinates: ({zone_x1}, {zone_y1}) to ({zone_x2}, {zone_y2})")
    print()
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(config.OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    
    # Tracking variables
    object_tracker = {}  # {object_id: frames_in_zone}
    last_alert_time = 0
    frame_count = 0
    next_object_id = 0
    
    # Processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Draw zone on frame
        draw_zone(frame, zone_coords, config.ZONE_COLOR, config.ZONE_THICKNESS)
        
        # Run detection
        results = model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
        
        current_objects = []
        
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            
            for i, box in enumerate(boxes):
                # Get detection info
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Check if class is in target list
                if class_name not in config.TARGET_CLASSES:
                    continue
                
                # Calculate center point
                center = get_box_center((x1, y1, x2, y2))
                
                # Check if center is in zone
                in_zone = is_point_in_zone(center, zone_coords)
                
                # Draw bounding box
                box_color = (0, 0, 255) if in_zone else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Draw center point
                cv2.circle(frame, center, 5, box_color, -1)
                
                # Draw label
                label = f"{class_name} {confidence:.2f}"
                if in_zone:
                    label += " [IN ZONE]"
                
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )
                
                # Track object if in zone
                if in_zone:
                    # Simple tracking: assign ID based on proximity to previous detections
                    # For production, consider using proper tracking algorithms
                    obj_id = f"{class_name}_{i}"
                    current_objects.append(obj_id)
                    
                    # Update frame counter
                    if obj_id in object_tracker:
                        object_tracker[obj_id] += 1
                    else:
                        object_tracker[obj_id] = 1
                    
                    frames_in_zone = object_tracker[obj_id]
                    
                    # Check if alert should be triggered
                    current_time = time.time()
                    cooldown_elapsed = (current_time - last_alert_time) >= config.COOLDOWN_SECONDS
                    
                    if frames_in_zone >= config.PERSISTENCE_FRAMES and cooldown_elapsed:
                        trigger_alert(frame, class_name, confidence, zone_coords, frames_in_zone)
                        last_alert_time = current_time
                        object_tracker[obj_id] = 0  # Reset counter after alert
        
        # Remove objects that are no longer in zone
        objects_to_remove = [obj_id for obj_id in object_tracker if obj_id not in current_objects]
        for obj_id in objects_to_remove:
            del object_tracker[obj_id]
        
        # Add frame info
        info_text = f"Frame: {frame_count} | Objects in zone: {len(current_objects)}"
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Write frame and display
        out.write(frame)
        cv2.imshow("Zone-Based Intrusion Detection", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("✅ Video processing completed")
    print(f"📹 Output saved to: {config.OUTPUT_PATH}")
    print(f"📊 Total frames processed: {frame_count}")
    if config.ENABLE_CSV_LOG and os.path.exists(config.CSV_LOG_PATH):
        with open(config.CSV_LOG_PATH, 'r') as f:
            alert_count = sum(1 for line in f) - 1  # Subtract header
        print(f"🚨 Total alerts triggered: {alert_count}")
    print("="*60)


if __name__ == "__main__":
    main()
