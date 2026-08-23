"""
Real-time Eye Tracker with MobileNetV2 + EAR + PERCLOS
Combines deep learning (MobileNetV2) with physiological signals (EAR/PERCLOS)
"""

# Fix for protobuf compatibility issue with MediaPipe
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# Suppress TensorFlow warnings and fix compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

import cv2
import numpy as np
import time
import pickle
import json
import os
import sys
from collections import deque
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Sound alerts will not work.")
    print("Install pygame: pip install pygame")

# Import MediaPipe after setting environment variable
try:
    import mediapipe as mp
except AttributeError as e:
    if 'getprototype' in str(e):
        import sys
        if 'google.protobuf.pyext._message' in sys.modules:
            del sys.modules['google.protobuf.pyext._message']
        import mediapipe as mp
    else:
        raise

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

# Constants
EAR_THRESHOLD = 0.18  # Eye Aspect Ratio threshold for closed eyes (optimized for Asian & hooded eyes)
EAR_CONSECUTIVE_FRAMES = 3  # Number of consecutive frames for eye closure
PERCLOS_THRESHOLD = 0.20  # PERCLOS threshold (20% eye closure over time)
FRAME_WINDOW = 30  # Number of frames to calculate PERCLOS over
IMG_SIZE = 224  # MobileNetV2 input size
MIN_EAR_FOR_OPEN = 0.05  # Minimum EAR to consider eye as open
BLINK_DURATION_THRESHOLD = 0.4  # Minimum eye closure duration (seconds) to be considered drowsy (blinks are < 0.4s)

# MediaPipe face mesh indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Full detailed contours for rich visualization
FULL_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
FULL_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]
LEFT_IRIS = 468
RIGHT_IRIS = 473

class DrowsinessDetector:
    def __init__(self, use_mobilenet=True):
        """
        Initialize detector with MobileNetV2 + EAR/PERCLOS
        """
        self.use_mobilenet = use_mobilenet
        
        # Initialize MediaPipe with robust fallback
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
        except (AttributeError, Exception):
            try:
                import mediapipe.python.solutions.face_mesh as mp_face_mesh
                self.mp_face_mesh = mp_face_mesh
            except Exception as e:
                raise RuntimeError(
                    f"MediaPipe Face Mesh solution is unavailable. Ensure mediapipe<=0.10.14 is installed. Details: {e}"
                )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        try:
            self.mp_face_detection = mp.solutions.face_detection
        except (AttributeError, Exception):
            try:
                import mediapipe.python.solutions.face_detection as mp_face_detection
                self.mp_face_detection = mp_face_detection
            except Exception as e:
                raise RuntimeError(
                    f"MediaPipe Face Detection solution is unavailable. Ensure mediapipe<=0.10.14 is installed. Details: {e}"
                )

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.3
        )
        
        # Base directory of this script to ensure relative paths work from any working directory
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Load MobileNetV2 models if available
        self.eye_model = None
        self.scaler_ear = None
        self.model_accuracy = None  # Store actual model accuracy
        
        if use_mobilenet:
            h5_path = os.path.join(base_dir, 'eye_model_mobilenet.h5')
            part1_path = os.path.join(base_dir, 'eye_model_mobilenet.h5.part1')
            part2_path = os.path.join(base_dir, 'eye_model_mobilenet.h5.part2')

            # Recombine split files if main .h5 is missing
            if not os.path.exists(h5_path) and os.path.exists(part1_path) and os.path.exists(part2_path):
                try:
                    with open(h5_path, 'wb') as fout:
                        with open(part1_path, 'rb') as f1:
                            fout.write(f1.read())
                        with open(part2_path, 'rb') as f2:
                            fout.write(f2.read())
                    print("[OK] Recombined split model files into eye_model_mobilenet.h5!")
                except Exception as ex:
                    print(f"[WARNING] Could not recombine model split files: {ex}")

            # Try loading H5 format first, then SavedModel format
            try:
                self.eye_model = tf.keras.models.load_model(
                    h5_path if os.path.exists(h5_path) else 'eye_model_mobilenet.h5',
                    compile=False
                )
                print("[OK] Loaded MobileNetV2 eye state model (H5 format)!")
            except:
                try:
                    # Try SavedModel format
                    saved_model_path = os.path.join(base_dir, 'eye_model_mobilenet')
                    self.eye_model = tf.keras.models.load_model(
                        saved_model_path if os.path.exists(saved_model_path) else 'eye_model_mobilenet',
                        compile=False
                    )
                    print("[OK] Loaded MobileNetV2 eye state model (SavedModel format)!")
                except Exception as e:
                    print(f"[WARNING] MobileNetV2 eye model not found or error: {e}")
                    print("   Using threshold-based detection.")
            
            # Recompile eye model if loaded
            if self.eye_model is not None:
                try:
                    self.eye_model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                except:
                    pass  # Model might already be compiled
            
            scaler_path = os.path.join(base_dir, 'scaler_ear_mobilenet.pkl')
            try:
                with open(scaler_path if os.path.exists(scaler_path) else 'scaler_ear_mobilenet.pkl', 'rb') as f:
                    self.scaler_ear = pickle.load(f)
                print("[OK] Loaded EAR feature scaler!")
            except:
                print("[WARNING] Scaler not found. Will not scale EAR features.")
            
            # Load actual model accuracy from training
            accuracy_path = os.path.join(base_dir, 'model_accuracy.json')
            if not os.path.exists(accuracy_path):
                accuracy_path = 'model_accuracy.json'
            try:
                if os.path.exists(accuracy_path):
                    with open(accuracy_path, 'r') as f:
                        accuracy_info = json.load(f)
                        self.model_accuracy = accuracy_info.get('test_accuracy', None)
                        if self.model_accuracy is not None:
                            print(f"[INFO] Model Test Accuracy: {self.model_accuracy:.2%}")
            except Exception as e:
                print(f"[WARNING] Could not load model accuracy: {e}")
        
        # Initialize pygame for sound alerts
        self.alert_sound = None
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                sound_candidates = [
                    os.path.join(base_dir, 'drowsy.wav'),
                    os.path.join(base_dir, 'critical.wav'),
                    os.path.join(base_dir, 'alert.wav'),
                    os.path.join(base_dir, 'static', 'drowsy.wav'),
                    os.path.join(base_dir, 'static', 'critical.wav'),
                    'drowsy.wav',
                    'critical.wav',
                    'alert.wav'
                ]
                for candidate in sound_candidates:
                    if os.path.exists(candidate):
                        self.alert_sound = pygame.mixer.Sound(candidate)
                        print(f"[OK] Loaded sound alert: {os.path.basename(candidate)}")
                        break
                if self.alert_sound is None:
                    print("Warning: alert sound file not found. System beep will be used.")
            except Exception as e:
                print(f"Warning: Could not initialize sound alert ({e}). Using system beep.")
        
        # Personal calibration & adaptive thresholding
        self.ear_threshold = EAR_THRESHOLD  # Baseline threshold (default: 0.18)
        self._calibrating = True
        self._calib_ear_samples = []
        self._calib_duration = 2.0  # 2 seconds calibration on startup
        self._calib_start_time = None
        self._smoothed_ear = None
        
        # State tracking
        self.ear_counter = 0
        self.ear_history = deque(maxlen=FRAME_WINDOW)
        self.last_alert_time = 0
        self.alert_cooldown = 2.0
        self.is_alert_playing = False
        self.alert_channel = None  # For continuous playback
        self.drowsiness_start_time = None  # Track when drowsiness started
        self.DROWSINESS_DELAY = 3.0  # 3 seconds delay before alerting
        self.was_drowsy = False  # Track previous state
        
        # Statistics
        self.total_frames = 0
        self.closed_frames = 0
        self.correct_predictions = 0  # For accuracy tracking
        self.total_predictions = 0  # For accuracy tracking
        self.accuracy_history = deque(maxlen=100)  # Track last 100 frames accuracy
        
        # Risk Scoring System
        self.driving_start_time = time.time()  # Time when detection started
        self.blink_count = 0  # Count blinks in window
        self.blink_window_start = time.time()  # Start of blink frequency window
        self.blink_window_duration = 10.0  # 10 seconds window for blink frequency
        self.last_eye_state = "Open"  # Track eye state transitions for blink detection
        self.head_pose_history = deque(maxlen=30)  # Track head pose for deviation
        self.eye_closure_start_time = None  # Track when eyes closed
        self.max_eye_closure_duration = 0.0  # Maximum eye closure duration
        self.alert_level = 0  # 0=No alert, 1=Visual, 2=Voice, 3=Loud+Vibration
        self.last_alert_escalation = 0  # Time of last alert escalation
    
    def calculate_ear(self, eye):
        """
        Calculate Eye Aspect Ratio (EAR) with normalization for different eye sizes
        Handles small eyes (e.g., Asian eyes), large eyebags, and glasses
        """
        if len(eye) < 6:
            return 0.0
        
        # Standard EAR calculation
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        
        if C == 0:
            return 0.0
        
        ear = (A + B) / (2.0 * C)
        
        # Additional validation: check if eye width vs height ratio makes sense
        # This helps filter out false detections from eyebags and glasses reflections
        eye_width = C
        eye_height_avg = (A + B) / 2.0
        
        # Validation for glasses and eyebags:
        # 1. If height is too large relative to width, might be eyebag/glasses interference
        # 2. Check if distances are reasonable (glasses can cause outliers)
        height_to_width_ratio = eye_height_avg / eye_width if eye_width > 0 else 0
        
        if height_to_width_ratio > 0.8:
            # Suspicious - might be detecting eyebag or glasses reflection
            # Adjust EAR slightly upward to avoid false closure
            ear = ear * 1.1
        
        # Check for glasses frame interference: if A or B is abnormally large/small
        # Glasses can cause one side to be detected incorrectly
        distances = [A, B]
        if max(distances) / min(distances) > 2.0 and min(distances) > 0:
            # One distance is much larger - might be glasses frame interference
            # Use the smaller distance (more reliable)
            ear = min(distances) / C
        
        # Clamp EAR to reasonable range (glasses might cause extreme values)
        ear = max(0.0, min(ear, 0.5))  # EAR typically ranges from 0.1 to 0.4
        
        return ear
    
    def extract_face_roi(self, frame, landmarks=None):
        """Extract face ROI for MobileNetV2 using landmarks or face detection"""
        h, w, _ = frame.shape
        
        # Method 1: Extract directly from landmarks (fast & 100% reliable)
        if landmarks is not None:
            try:
                xs = [landmarks.landmark[i].x * w for i in range(len(landmarks.landmark))]
                ys = [landmarks.landmark[i].y * h for i in range(len(landmarks.landmark))]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                pad = int(max(x_max - x_min, y_max - y_min) * 0.1)
                x1 = max(0, x_min - pad)
                y1 = max(0, y_min - pad)
                x2 = min(w, x_max + pad)
                y2 = min(h, y_max + pad)
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0:
                    return cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            except Exception:
                pass
        
        # Method 2: Fallback to face_detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if not results.detections:
            return None
        
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        width = min(w - x, width + 2 * padding)
        height = min(h - y, height + 2 * padding)
        
        face_roi = frame[y:y+height, x:x+width]
        if face_roi.size == 0:
            return None
        
        face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        return face_roi
    
    def extract_ear_features(self, landmarks, h, w):
        """Extract EAR features for eye state detection"""
        def get_landmark_coords(indices):
            coords = []
            for idx in indices:
                try:
                    coords.append((landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h))
                except (IndexError, AttributeError):
                    continue
            return np.array(coords) if coords else np.array([])
        
        left_eye = get_landmark_coords(LEFT_EYE_INDICES)
        right_eye = get_landmark_coords(RIGHT_EYE_INDICES)
        
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        # Handle glasses: if one eye has very low EAR but the other is normal,
        # it might be glasses frame interference, so use the better value
        # This helps when glasses partially occlude one eye
        if left_ear > 0 and right_ear > 0:
            # Both eyes detected - use average
            avg_ear = (left_ear + right_ear) / 2.0
            # If one is much lower, might be glasses occlusion - trust the higher one more
            if abs(left_ear - right_ear) > 0.1:  # Significant difference
                # Use weighted average favoring the higher (more reliable) value
                max_ear = max(left_ear, right_ear)
                min_ear = min(left_ear, right_ear)
                avg_ear = max_ear * 0.7 + min_ear * 0.3
        elif left_ear > 0:
            avg_ear = left_ear
        elif right_ear > 0:
            avg_ear = right_ear
        else:
            avg_ear = 0.0
        
        # Calculate eye dimensions with error handling
        left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3]) if len(left_eye) >= 4 else 0
        left_eye_height = 0
        if len(left_eye) >= 6:
            left_eye_height = (np.linalg.norm(left_eye[1] - left_eye[5]) + 
                              np.linalg.norm(left_eye[2] - left_eye[4])) / 2.0
        
        right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3]) if len(right_eye) >= 4 else 0
        right_eye_height = 0
        if len(right_eye) >= 6:
            right_eye_height = (np.linalg.norm(right_eye[1] - right_eye[5]) + 
                               np.linalg.norm(right_eye[2] - right_eye[4])) / 2.0
        
        # Calculate eye statistics with safety checks
        eye_aspect_variance = np.var([left_ear, right_ear]) if left_ear > 0 and right_ear > 0 else 0
        eye_symmetry = 1.0 - abs(left_ear - right_ear) / (avg_ear + 1e-6) if avg_ear > 0 else 1.0
        
        features = [
            avg_ear, left_ear, right_ear,
            left_eye_width, left_eye_height,
            right_eye_width, right_eye_height,
            eye_aspect_variance, eye_symmetry
        ]
        
        return features, avg_ear
    
    def calculate_perclos(self):
        """Calculate PERCLOS (Percentage of Eye Closure)"""
        if len(self.ear_history) == 0:
            return 0.0
        thr = getattr(self, 'ear_threshold', EAR_THRESHOLD)
        closed_count = sum(1 for ear in self.ear_history if ear < thr)
        return closed_count / len(self.ear_history)
    
    def calculate_head_pose(self, landmarks, h, w):
        """Calculate head pose (pitch, yaw, roll) from face landmarks"""
        try:
            # Key facial points for head pose estimation
            # Nose tip
            nose_tip = np.array([
                landmarks.landmark[1].x * w,
                landmarks.landmark[1].y * h
            ])
            # Chin
            chin = np.array([
                landmarks.landmark[175].x * w,
                landmarks.landmark[175].y * h
            ])
            # Left eye corner
            left_eye = np.array([
                landmarks.landmark[33].x * w,
                landmarks.landmark[33].y * h
            ])
            # Right eye corner
            right_eye = np.array([
                landmarks.landmark[263].x * w,
                landmarks.landmark[263].y * h
            ])
            # Forehead center
            forehead = np.array([
                landmarks.landmark[10].x * w,
                landmarks.landmark[10].y * h
            ])
            
            # Calculate angles
            # Yaw (left-right rotation) - based on eye horizontal position
            eye_center = (left_eye + right_eye) / 2
            face_center_x = nose_tip[0]
            eye_center_x = eye_center[0]
            yaw = abs(face_center_x - eye_center_x) / max(w, h) * 100  # Normalized
            
            # Pitch (up-down rotation) - based on vertical position of nose vs eyes
            nose_eye_y_diff = nose_tip[1] - eye_center[1]
            pitch = abs(nose_eye_y_diff) / max(w, h) * 100  # Normalized
            
            # Roll (tilt) - angle between eyes horizontal line
            eye_vec = right_eye - left_eye
            roll = np.degrees(np.arctan2(eye_vec[1], eye_vec[0]))
            
            return {
                'pitch': pitch,
                'yaw': yaw,
                'roll': abs(roll)
            }
        except:
            return {'pitch': 0, 'yaw': 0, 'roll': 0}
    
    def calculate_gaze_deviation(self, head_pose):
        """Calculate gaze deviation score (0-100)"""
        # Combine pitch and yaw to get total deviation
        deviation = (head_pose['pitch'] + head_pose['yaw']) / 2.0
        # Normalize to 0-100 (threshold: 15% deviation is dangerous)
        deviation_score = min(100, (deviation / 15.0) * 100)
        return deviation_score
    
    def track_blink_frequency(self, eye_state):
        """Track blink frequency (blinks per minute)"""
        current_time = time.time()
        
        # Detect blink: transition from Open to Closed
        if self.last_eye_state == "Open" and eye_state == "Closed":
            self.blink_count += 1
        
        self.last_eye_state = eye_state
        
        # Reset window if time elapsed
        if current_time - self.blink_window_start >= self.blink_window_duration:
            # Calculate blinks per minute
            elapsed = current_time - self.blink_window_start
            if elapsed > 0:
                blinks_per_minute = (self.blink_count / elapsed) * 60
            else:
                blinks_per_minute = 0
            
            # Reset
            self.blink_count = 0
            self.blink_window_start = current_time
            
            return blinks_per_minute
        
        # Return None if window not complete yet
        if current_time - self.blink_window_start < 2.0:  # Need at least 2 seconds
            return None
        
        # Calculate current rate
        elapsed = current_time - self.blink_window_start
        if elapsed > 0:
            return (self.blink_count / elapsed) * 60
        return 0
    
    def calculate_risk_score(self, eye_state, avg_ear, perclos, head_pose, 
                           eye_closure_duration, blink_frequency, driving_duration):
        """
        Calculate Driver Risk Index (0-100)
        Based on multiple factors:
        - Eye closure duration (0-30 points)
        - Blink frequency (0-20 points)
        - Gaze deviation (0-20 points)
        - Head pose (0-15 points)
        - Time of continuous driving (0-15 points)
        """
        risk_score = 0.0
        
        # 1. Eye closure duration (0-30 points)
        # Only count closures longer than blink threshold (ignore normal blinks < 0.4s)
        if eye_closure_duration > BLINK_DURATION_THRESHOLD:
            # Calculate risk based on duration beyond blink threshold
            # 3+ seconds beyond blink threshold = max risk (30 points)
            effective_duration = eye_closure_duration - BLINK_DURATION_THRESHOLD
            closure_risk = min(30, (effective_duration / 3.0) * 30)
            risk_score += closure_risk
        
        # 2. PERCLOS factor (overlaps with closure duration, but adds risk)
        if perclos > 0.3:  # 30% closure
            risk_score += min(20, perclos * 50)
        
        # 3. Blink frequency (0-20 points)
        if blink_frequency is not None:
            # Normal: 15-20 blinks/min
            # Too low (<10): drowsy (high risk)
            # Too high (>30): stressed or irritated (moderate risk)
            if blink_frequency < 10:
                blink_risk = 20 - (blink_frequency / 10.0) * 20  # Inverse: lower = higher risk
            elif blink_frequency > 30:
                blink_risk = min(15, (blink_frequency - 30) / 30.0 * 15)
            else:
                blink_risk = 0  # Normal range
            risk_score += blink_risk
        
        # 4. Gaze deviation (0-20 points)
        gaze_deviation = self.calculate_gaze_deviation(head_pose)
        risk_score += min(20, gaze_deviation)
        
        # 5. Head pose deviation (0-15 points)
        # Roll (tilt) is particularly dangerous
        roll_risk = min(10, (head_pose['roll'] / 45.0) * 10)  # 45 degrees = max risk
        yaw_pitch_risk = min(5, ((head_pose['yaw'] + head_pose['pitch']) / 30.0) * 5)
        risk_score += roll_risk + yaw_pitch_risk
        
        # 6. Time of continuous driving (0-15 points)
        # After 2 hours, risk increases
        if driving_duration > 7200:  # 2 hours = 7200 seconds
            driving_risk = min(15, ((driving_duration - 7200) / 3600.0) * 15)  # +15 per hour after 2h
            risk_score += driving_risk
        
        # Cap at 100
        return min(100, risk_score)
    
    def get_alert_level_from_risk(self, risk_score, previous_alert_level):
        """
        Determine alert level based on risk score and escalation logic
        0 = No alert
        1 = Visual warning (icon)
        2 = Voice alert
        3 = Loud sound + vibration (Critical)
        """
        current_time = time.time()
        escalation_cooldown = 3.0  # Wait 3 seconds before escalating
        
        if risk_score < 20:
            return 0  # Safe
        elif risk_score < 50:
            return 1  # Visual warning
        elif risk_score < 70:
            # Voice alert - escalate from visual if not already voice
            if previous_alert_level < 2:
                if current_time - self.last_alert_escalation >= escalation_cooldown:
                    self.last_alert_escalation = current_time
                    return 2
            return max(previous_alert_level, 2)
        else:
            # High risk (Drowsy) - immediately trigger critical alert with voice
            # Don't wait for cooldown if jumping to critical
            if previous_alert_level < 3:
                self.last_alert_escalation = current_time
                return 3
            return 3
    
    def play_alert(self, continuous=True):
        """Play sound alert - continuous if drowsy, stop if not"""
        if continuous:
            if PYGAME_AVAILABLE and self.alert_sound:
                if not self.is_alert_playing:
                    # Start playing in a loop
                    self.alert_channel = self.alert_sound.play(loops=-1)  # -1 means infinite loop
                    self.is_alert_playing = True
            else:
                # Fallback beep if sound file or channel is unavailable
                try:
                    if sys.platform == 'win32':
                        import winsound
                        winsound.Beep(1000, 200)
                    else:
                        print('\a', end='', flush=True)
                except Exception:
                    pass
        else:
            # Stop the alert
            self.stop_alert()
    
    def stop_alert(self):
        """Stop the continuous alert"""
        if PYGAME_AVAILABLE and self.is_alert_playing and self.alert_channel:
            self.alert_channel.stop()
            self.is_alert_playing = False
            self.alert_channel = None
    
    def detect(self, frame):
        """Detect drowsiness using MobileNetV2 + EAR + PERCLOS"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return (False, "No Face", 0.0, 0.0, None, 0.0, "ERROR", 0)
        
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        
        # Extract EAR/PERCLOS features
        ear_features, avg_ear = self.extract_ear_features(landmarks, h, w)
        
        # Temporal smoothing for EAR to eliminate webcam noise & compression jitter
        if self._smoothed_ear is None:
            self._smoothed_ear = avg_ear
        else:
            self._smoothed_ear = 0.65 * avg_ear + 0.35 * self._smoothed_ear
        current_ear = self._smoothed_ear
        
        # Calibration phase (first 2 seconds of seeing face)
        current_time = time.time()
        if self._calibrating:
            if self._calib_start_time is None:
                self._calib_start_time = current_time
            if current_ear > 0.05:
                self._calib_ear_samples.append(current_ear)
            if current_time - self._calib_start_time >= self._calib_duration and len(self._calib_ear_samples) >= 5:
                baseline = float(np.median(self._calib_ear_samples))
                self.ear_threshold = max(0.13, min(0.23, round(baseline * 0.72, 3)))
                self._calibrating = False
                print(f"[Auto-Calibration] Baseline EAR: {baseline:.3f} | Dynamic Threshold: {self.ear_threshold:.3f}")
        
        # Update EAR history for PERCLOS
        self.ear_history.append(current_ear)
        perclos = self.calculate_perclos()
        
        # Determine eye state using Intelligent Multi-Modal Fusion (EAR + MobileNet)
        active_threshold = getattr(self, 'ear_threshold', EAR_THRESHOLD)
        left_eye_width = ear_features[3] if len(ear_features) > 3 else 50
        right_eye_width = ear_features[5] if len(ear_features) > 5 else 50
        avg_eye_width = (left_eye_width + right_eye_width) / 2.0 if (left_eye_width + right_eye_width) > 0 else 50
        if avg_eye_width < 35:
            active_threshold = active_threshold * 0.88
            
        ear_closed = current_ear < active_threshold
        ear_ambiguous = (active_threshold <= current_ear <= active_threshold * 1.30)
        
        model_state = None
        # In ambiguous zone, query MobileNetV2 for feature fusion confirmation
        if self.use_mobilenet and self.eye_model is not None and ear_ambiguous:
            try:
                face_roi = self.extract_face_roi(frame, landmarks)
                if face_roi is not None:
                    face_processed = preprocess_input(face_roi.astype('float32'))
                    face_processed = np.expand_dims(face_processed, axis=0)
                    if self.scaler_ear is not None:
                        ear_features_scaled = self.scaler_ear.transform([ear_features])
                    else:
                        ear_features_scaled = np.array([ear_features])
                    eye_pred = self.eye_model.predict([face_processed, ear_features_scaled], verbose=0)
                    # Class 0: Closed, Class 1: Open
                    prob_closed = float(eye_pred[0][0])
                    prob_open = float(eye_pred[0][1])
                    if prob_closed > 0.55:
                        model_state = "Closed"
                    elif prob_open > 0.55:
                        model_state = "Open"
            except Exception as e:
                pass
        
        # Decision Fusion:
        if ear_closed:
            eye_state = "Closed"
        elif ear_ambiguous and model_state is not None:
            eye_state = model_state
        else:
            eye_state = "Open"

        # Update closure counters for drowsiness tracking
        if eye_state == "Closed":
            self.ear_counter += 1
            if self.ear_counter >= EAR_CONSECUTIVE_FRAMES:
                self.closed_frames += 1
        else:
            self.ear_counter = 0
        
        # Track drowsiness duration - only alert after 3 seconds
        current_time = time.time()
        is_drowsy = False  # Initialize
        
        # Track eye closure duration for risk scoring (calculate BEFORE checking drowsiness)
        if eye_state == "Closed":
            if self.eye_closure_start_time is None:
                self.eye_closure_start_time = current_time
            eye_closure_duration = current_time - self.eye_closure_start_time
            self.max_eye_closure_duration = max(self.max_eye_closure_duration, eye_closure_duration)
        else:
            if self.eye_closure_start_time is not None:
                # Eyes just opened, reset
                self.eye_closure_start_time = None
            eye_closure_duration = 0.0
        
        # Determine if drowsy (but don't alert immediately)
        # Drowsy if: eyes closed for longer than blink threshold OR high PERCLOS
        # Normal blinks (< 0.4s) should NOT trigger drowsiness
        is_drowsy_condition = False
        if eye_state == "Closed":
            # Only consider it drowsy if closure duration exceeds blink threshold (filter out normal blinks)
            if eye_closure_duration > BLINK_DURATION_THRESHOLD:
                is_drowsy_condition = True
            # If eyes just closed (< 0.4s), it's likely a blink - don't trigger drowsiness
        elif perclos > PERCLOS_THRESHOLD:
            # PERCLOS already filters out short blinks over a window, so this is reliable
            is_drowsy_condition = True
        
        if is_drowsy_condition:
            if self.drowsiness_start_time is None:
                # First frame of drowsiness - start timer
                self.drowsiness_start_time = current_time
                is_drowsy = False  # Not drowsy yet, still in delay
            else:
                # Check if 3 seconds have passed
                drowsiness_duration = current_time - self.drowsiness_start_time
                if drowsiness_duration >= self.DROWSINESS_DELAY:
                    # 3 seconds passed - now trigger alert
                    is_drowsy = True
                    self.play_alert(continuous=True)
                else:
                    # Still in delay period - don't alert yet
                    is_drowsy = False
        else:
            # Not drowsy - reset timer and stop alert
            self.drowsiness_start_time = None
            is_drowsy = False
            self.play_alert(continuous=False)  # This will stop the alert
        
        # Calculate head pose and gaze deviation
        head_pose = self.calculate_head_pose(landmarks, h, w)
        self.head_pose_history.append(head_pose)
        
        # Track blink frequency
        blink_frequency = self.track_blink_frequency(eye_state)
        
        # Calculate driving duration
        driving_duration = current_time - self.driving_start_time
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(
            eye_state, avg_ear, perclos, head_pose,
            eye_closure_duration, blink_frequency, driving_duration
        )
        
        # Determine alert level based on risk score
        self.alert_level = self.get_alert_level_from_risk(risk_score, self.alert_level)
        
        # Determine risk category based on risk_score (0-100)
        # 0-39: Awake, 40-69: Drowsy, 70-100: Critical
        if risk_score <= 39:
            risk_category = "Awake"
        elif risk_score <= 69:
            risk_category = "Drowsy"
        else:
            risk_category = "Critical"
        
        self.total_frames += 1
        
        return (is_drowsy, eye_state, avg_ear, perclos, landmarks, risk_score, risk_category, self.alert_level)
    
    def draw_landmarks(self, frame, landmarks, eye_state="Open", avg_ear=None):
        """Draw prominent facial and eye landmarks with DMS HUD styling"""
        if landmarks is None or frame is None:
            return frame

        h, w, _ = frame.shape
        num_landmarks = len(landmarks.landmark)

        # Dynamic color based on eye state
        is_closed = (eye_state == "Closed")
        primary_color = (0, 70, 255) if is_closed else (0, 255, 80)        # Red if closed, Vibrant Green if open (BGR)
        keypoint_color = (0, 230, 255) if not is_closed else (0, 165, 255) # Yellow/Cyan dots
        bracket_color = (0, 140, 255) if is_closed else (0, 220, 255)      # HUD Corner brackets

        def draw_contour(indices, color, thickness=2, is_closed_loop=True):
            try:
                pts = np.array([
                    (int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h))
                    for i in indices if i < num_landmarks
                ], np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], is_closed_loop, color, thickness, cv2.LINE_AA)
                    for pt in pts:
                        cv2.circle(frame, tuple(pt), 2, keypoint_color, -1, cv2.LINE_AA)
                return pts
            except Exception:
                return None

        # 1. Draw Full Detailed Eye Contours
        pts_left = draw_contour(FULL_LEFT_EYE, primary_color, thickness=2, is_closed_loop=True)
        pts_right = draw_contour(FULL_RIGHT_EYE, primary_color, thickness=2, is_closed_loop=True)

        # 2. Draw Eyebrows (subtle tech overlay)
        draw_contour(LEFT_EYEBROW, (160, 160, 160), thickness=1, is_closed_loop=False)
        draw_contour(RIGHT_EYEBROW, (160, 160, 160), thickness=1, is_closed_loop=False)

        # 3. Draw Iris / Pupil center points
        if num_landmarks > 473:
            for iris_idx in (LEFT_IRIS, RIGHT_IRIS):
                ix = int(landmarks.landmark[iris_idx].x * w)
                iy = int(landmarks.landmark[iris_idx].y * h)
                cv2.circle(frame, (ix, iy), 3, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (ix, iy), 6, (0, 255, 255), 1, cv2.LINE_AA)

        # 4. Draw Corner HUD Brackets around each eye
        for pts, label in ((pts_left, "L"), (pts_right, "R")):
            if pts is not None and len(pts) > 0:
                x_min, y_min = np.min(pts, axis=0)
                x_max, y_max = np.max(pts, axis=0)
                pad = 8
                x1, y1 = max(0, x_min - pad), max(0, y_min - pad)
                x2, y2 = min(w - 1, x_max + pad), min(h - 1, y_max + pad)
                
                blen = max(6, int((x2 - x1) * 0.25))
                # Top-left
                cv2.line(frame, (x1, y1), (x1 + blen, y1), bracket_color, 2, cv2.LINE_AA)
                cv2.line(frame, (x1, y1), (x1, y1 + blen), bracket_color, 2, cv2.LINE_AA)
                # Top-right
                cv2.line(frame, (x2, y1), (x2 - blen, y1), bracket_color, 2, cv2.LINE_AA)
                cv2.line(frame, (x2, y1), (x2, y1 + blen), bracket_color, 2, cv2.LINE_AA)
                # Bottom-left
                cv2.line(frame, (x1, y2), (x1 + blen, y2), bracket_color, 2, cv2.LINE_AA)
                cv2.line(frame, (x1, y2), (x1, y2 - blen), bracket_color, 2, cv2.LINE_AA)
                # Bottom-right
                cv2.line(frame, (x2, y2), (x2 - blen, y2), bracket_color, 2, cv2.LINE_AA)
                cv2.line(frame, (x2, y2), (x2, y2 - blen), bracket_color, 2, cv2.LINE_AA)

        return frame
    
    def close(self):
        """Close MediaPipe resources and stop any alerts"""
        self.stop_alert()  # Stop any playing alerts
        self.face_mesh.close()
        self.face_detection.close()

def main():
    """Main function for real-time drowsiness detection"""
    detector = DrowsinessDetector(use_mobilenet=True)
    
    cap = _create_video_capture()
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\n" + "="*60)
    print("MobileNetV2 + EAR + PERCLOS Drowsiness Detection")
    print("="*60)
    print("Press 'f' to toggle mirror view, 'q' to quit\n")
    
    mirror_mode = True
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if mirror_mode:
                frame = cv2.flip(frame, 1)
            
            is_drowsy, eye_state, ear, perclos, landmarks, risk_score, risk_category, alert_level = detector.detect(frame)
            
            if landmarks is not None:
                frame = detector.draw_landmarks(frame, landmarks, eye_state=eye_state, avg_ear=ear)
            
            status_color = (0, 0, 255) if is_drowsy else (0, 255, 0)
            status_text = "DROWSY!" if is_drowsy else "AWAKE"
            
            cv2.putText(frame, f"Status: {status_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"Eye: {eye_state}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"PERCLOS: {perclos:.1%}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Model: MobileNetV2+EAR+PERCLOS", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            if is_drowsy:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), 
                             (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                cv2.putText(frame, "WARNING: DROWSINESS DETECTED!", 
                             (50, frame.shape[0] // 2),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            cv2.imshow("MobileNetV2 + EAR + PERCLOS Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                mirror_mode = not mirror_mode
                print(f"[Camera] Mirror View: {'ON (Selfie)' if mirror_mode else 'OFF (Observer)'}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        
        if detector.total_frames > 0:
            print("\n" + "="*60)
            print("SESSION STATISTICS")
            print("="*60)
            print(f"Total frames processed: {detector.total_frames}")
            print(f"Closed frames: {detector.closed_frames}")
            print(f"Eye closure rate: {detector.closed_frames/detector.total_frames:.1%}")
            
            # Only show accuracy if it's from actual training (not estimates)
            if detector.eye_model is not None and detector.model_accuracy is not None:
                final_accuracy = detector.model_accuracy * 100  # Convert to percentage
                print("\n" + "="*60)
                print("MODEL ACCURACY (TRAINED ON TEST SET)")
                print("="*60)
                print(f"[OK] Model Type: MobileNetV2 + EAR + PERCLOS")
                print(f"[OK] Test Accuracy: {final_accuracy:.2f}%")
                print(f"[OK] This is the ACTUAL accuracy from model training!")
                print("="*60)

if __name__ == "__main__":
    main()
