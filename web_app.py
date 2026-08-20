"""
Web interface for Drowsiness Detection System
Beautiful HTML/CSS interface with real-time video streaming
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import base64
from flask import Flask, Response, render_template, jsonify, request
import threading
import time
import webbrowser
import sys

# Try to import winsound for Windows beep
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

from eye_tracker_mobilenet import DrowsinessDetector

app = Flask(__name__)

class _OpenCVCapture:
    def __init__(self, cap):
        self._cap = cap

    def isOpened(self):
        return self._cap.isOpened()

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()


class _PiCamera2Capture:
    def __init__(self, picam2):
        self._picam2 = picam2

    def isOpened(self):
        return True

    def read(self):
        frame = self._picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame

    def release(self):
        try:
            self._picam2.stop()
        except Exception:
            pass


def _create_video_capture():
    backend = os.getenv("CAMERA_BACKEND", "auto").strip().lower()
    if backend in ("auto", "picamera2", "pi", "libcamera") and sys.platform.startswith("linux"):
        try:
            from picamera2 import Picamera2

            size_str = os.getenv("CAMERA_SIZE", "640x480").lower()
            try:
                w_str, h_str = size_str.split("x", 1)
                size = (int(w_str), int(h_str))
            except Exception:
                size = (640, 480)
            fps = int(os.getenv("CAMERA_FPS", "30"))

            picam2 = Picamera2()
            config = picam2.create_video_configuration(
                main={"format": "RGB888", "size": size},
                controls={"FrameRate": fps},
            )
            picam2.configure(config)
            picam2.start()
            return _PiCamera2Capture(picam2)
        except Exception:
            if backend != "auto":
                raise

    index = int(os.getenv("CAMERA_INDEX", "0"))
    cap = cv2.VideoCapture(index)
    width = os.getenv("CAMERA_WIDTH")
    height = os.getenv("CAMERA_HEIGHT")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return _OpenCVCapture(cap)


# Global variables for video capture and detection
detector = None
video_capture = None
frame_lock = threading.Lock()
current_frame = None
no_face_alarm_active = False
no_face_alarm_thread = None
current_data = {
    'is_drowsy': False,
    'eye_state': 'No Face',
    'ear': 0.0,
    'perclos': 0.0,
    'risk_score': 0.0,
    'risk_category': 'Awake',
    'alert_level': 0
}

def play_no_face_alarm():
    """Play continuous alarm sound when no face is detected"""
    global no_face_alarm_active
    while no_face_alarm_active:
        try:
            # Play beep sound (Windows)
            if sys.platform == 'win32' and WINSOUND_AVAILABLE:
                winsound.Beep(800, 300)  # 800 Hz, 300ms duration
            else:
                # Linux/Mac fallback - system bell
                print('\a', end='', flush=True)
            time.sleep(0.5)  # Repeat every 0.5 seconds for continuous alarm
        except:
            pass

def generate_frames():
    """Generator function for video streaming"""
    global current_frame, current_data
    
    while True:
        with frame_lock:
            if current_frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', current_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    # Yield frame in multipart format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Small delay to control frame rate
        time.sleep(0.033)  # ~30 FPS

def video_capture_thread():
    """Thread function for continuous video capture and detection"""
    global detector, video_capture, current_frame, current_data
    
    detector = DrowsinessDetector(use_mobilenet=True)
    video_capture = _create_video_capture()
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully")
    
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Run detection
            is_drowsy, eye_state, ear, perclos, landmarks, risk_score, risk_category, alert_level = detector.detect(frame)
            
            # Handle no face detected - play alarm
            global no_face_alarm_active, no_face_alarm_thread
            if eye_state == "No Face":
                # Start alarm if not already playing
                if not no_face_alarm_active:
                    no_face_alarm_active = True
                    if no_face_alarm_thread is None or not no_face_alarm_thread.is_alive():
                        no_face_alarm_thread = threading.Thread(target=play_no_face_alarm, daemon=True)
                        no_face_alarm_thread.start()
                
                # Draw camera error overlay
                h, w = frame.shape[:2]
                
                # Create dark overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
                frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
                
                # Error box design with border
                box_padding = 30
                box_width = min(500, w - 40)
                box_height = 180
                box_x = (w - box_width) // 2
                box_y = (h - box_height) // 2
                
                # Draw error box background
                cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (30, 30, 30), -1)
                cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 100, 255), 3)
                
                # Error icon (simple circle with X)
                icon_center = (box_x + 50, box_y + 50)
                cv2.circle(frame, icon_center, 25, (0, 100, 255), 3)
                cv2.line(frame, (icon_center[0] - 15, icon_center[1] - 15), 
                        (icon_center[0] + 15, icon_center[1] + 15), (0, 100, 255), 3)
                cv2.line(frame, (icon_center[0] + 15, icon_center[1] - 15), 
                        (icon_center[0] - 15, icon_center[1] + 15), (0, 100, 255), 3)
                
                # Error text - calculate font scale to fit
                font = cv2.FONT_HERSHEY_SIMPLEX
                main_text = "Face Not Detected"
                sub_text = "Check camera position"
                hint_text = "Ensure proper lighting"
                
                # Calculate appropriate font scale based on box width
                main_scale = 0.8
                sub_scale = 0.5
                hint_scale = 0.45
                
                # Get text sizes
                (main_w, main_h), _ = cv2.getTextSize(main_text, font, main_scale, 2)
                (sub_w, sub_h), _ = cv2.getTextSize(sub_text, font, sub_scale, 1)
                (hint_w, hint_h), _ = cv2.getTextSize(hint_text, font, hint_scale, 1)
                
                # Adjust if text is too wide
                max_text_width = box_width - 100
                if main_w > max_text_width:
                    main_scale = (max_text_width / main_w) * main_scale
                    (main_w, main_h), _ = cv2.getTextSize(main_text, font, main_scale, 2)
                
                # Draw texts centered in box
                text_start_x = box_x + (box_width - main_w) // 2
                main_y = box_y + 50
                sub_y = box_y + 90
                hint_y = box_y + 125
                
                cv2.putText(frame, main_text, (text_start_x, main_y), font, main_scale, (0, 150, 255), 2)
                cv2.putText(frame, sub_text, (box_x + (box_width - sub_w) // 2, sub_y), font, sub_scale, (180, 180, 180), 1)
                cv2.putText(frame, hint_text, (box_x + (box_width - hint_w) // 2, hint_y), font, hint_scale, (150, 150, 150), 1)
            else:
                # Stop alarm if face is detected
                if no_face_alarm_active:
                    no_face_alarm_active = False
                
                # Draw landmarks if face detected
                if landmarks is not None:
                    frame = detector.draw_landmarks(frame, landmarks)
            
            # Update risk category if no face detected
            if eye_state == "No Face":
                risk_category = "ERROR"
            
            # Update current frame and data
            with frame_lock:
                current_frame = frame.copy()
                current_data = {
                    'is_drowsy': is_drowsy,
                    'eye_state': eye_state,
                    'ear': round(ear, 3),
                    'perclos': round(perclos * 100, 1),  # Convert to percentage
                    'risk_score': round(risk_score, 1),
                    'risk_category': risk_category,
                    'alert_level': alert_level
                }
            
            time.sleep(0.033)  # ~30 FPS
    
    except Exception as e:
        print(f"Error in video capture thread: {e}")
    finally:
        if video_capture is not None:
            video_capture.release()
        if detector is not None:
            detector.close()

def get_detector():
    global detector
    if detector is None:
        detector = DrowsinessDetector(use_mobilenet=True)
    return detector

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route for local camera"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def get_data():
    """API endpoint for getting current local detection data"""
    with frame_lock:
        return jsonify(current_data)

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """API endpoint to process a base64 encoded frame sent from client browser camera"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        img_str = data['image']
        if ',' in img_str:
            img_str = img_str.split(',', 1)[1]
        
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400

        det = get_detector()
        is_drowsy, eye_state, ear, perclos, landmarks, risk_score, risk_category, alert_level = det.detect(frame)

        if eye_state != "No Face" and landmarks is not None:
            frame = det.draw_landmarks(frame, landmarks)

        if eye_state == "No Face":
            risk_category = "ERROR"

        # Encode annotated image back to base64
        ret, buffer = cv2.imencode('.jpg', frame)
        annotated_base64 = ""
        if ret:
            annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'is_drowsy': bool(is_drowsy),
            'eye_state': str(eye_state),
            'ear': round(float(ear), 3),
            'perclos': round(float(perclos) * 100, 1),
            'risk_score': round(float(risk_score), 1),
            'risk_category': str(risk_category),
            'alert_level': int(alert_level),
            'annotated_image': annotated_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize detector on startup for cloud deployment
try:
    get_detector()
except Exception as err:
    print(f"Detector initialization notice: {err}")

if __name__ == '__main__':
    enable_server_cam = os.getenv("ENABLE_SERVER_CAM", "true").lower() == "true"
    if enable_server_cam:
        # Start video capture thread for server webcam if enabled
        video_thread = threading.Thread(target=video_capture_thread, daemon=True)
        video_thread.start()
        time.sleep(1)
    
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*60)
    print("Drowsiness Detection Web Interface")
    print("="*60)
    print(f"Starting Server on port {port}...")
    print("="*60 + "\n")
    
    if port == 5000:
        def open_browser():
            time.sleep(2.5)
            url = f'http://localhost:{port}'
            webbrowser.open(url)
            print(f"\n✓ Browser opened automatically at {url}")
            print(f"  Or go to: {url}")
            print("  Press Ctrl+C to stop the server\n")
        
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\nShutting down server...")

