"""
Train MobileNetV2 + EAR Feature Model for Eye State Detection
Combines MobileNetV2 (face image) with EAR features for drowsiness detection
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

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

# Constants
IMG_SIZE = 224
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye):
    """Calculate Eye Aspect Ratio"""
    if len(eye) < 6:
        return 0.0
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def extract_features(image_path, face_mesh):
    """Extract MobileNetV2 features and EAR features from image"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        # Resize and preprocess for MobileNetV2
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
        image_preprocessed = preprocess_input(image_resized.astype('float32'))
        
        # Extract EAR features using MediaPipe
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None, None
        
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        
        def get_landmark_coords(indices):
            coords = []
            for idx in indices:
                try:
                    coords.append((landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h))
                except:
                    continue
            return np.array(coords) if coords else np.array([])
        
        left_eye = get_landmark_coords(LEFT_EYE_INDICES)
        right_eye = get_landmark_coords(RIGHT_EYE_INDICES)
        
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0 if (left_ear > 0 and right_ear > 0) else (left_ear if left_ear > 0 else right_ear)
        
        # Calculate additional EAR features
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
        
        eye_aspect_variance = np.var([left_ear, right_ear]) if left_ear > 0 and right_ear > 0 else 0
        eye_symmetry = 1.0 - abs(left_ear - right_ear) / (avg_ear + 1e-6) if avg_ear > 0 else 1.0
        
        ear_features = [
            avg_ear, left_ear, right_ear,
            left_eye_width, left_eye_height,
            right_eye_width, right_eye_height,
            eye_aspect_variance, eye_symmetry
        ]
        
        return image_preprocessed, ear_features
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def load_dataset(dataset_dir):
    """Load dataset from directory structure"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    images = []
    ear_features_list = []
    labels = []
    
    # Open = 1, Closed = 0
    class_mapping = {
        'Open': 1,
        'Closed': 0
    }
    
    for class_name, class_label in class_mapping.items():
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping...")
            continue
        
        print(f"Loading {class_name} images...")
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for idx, img_file in enumerate(image_files):
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} {class_name} images")
            
            img_path = os.path.join(class_dir, img_file)
            image_preprocessed, ear_features = extract_features(img_path, face_mesh)
            
            if image_preprocessed is not None and ear_features is not None:
                images.append(image_preprocessed)
                ear_features_list.append(ear_features)
                labels.append(class_label)
    
    face_mesh.close()
    
    print(f"\nLoaded {len(images)} images total")
    print(f"  Open: {labels.count(1)}")
    print(f"  Closed: {labels.count(0)}")
    
    return np.array(images), np.array(ear_features_list), np.array(labels)

def build_model(ear_feature_dim=9):
    """Build MobileNetV2 + EAR feature fusion model"""
    # MobileNetV2 input (face image)
    image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    mobilenet_base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                 include_top=False,
                                 weights='imagenet',
                                 alpha=1.0)
    mobilenet_base.trainable = True  # Fine-tune the base model
    
    # Extract features from MobileNetV2
    mobilenet_features = mobilenet_base(image_input)
    mobilenet_features = tf.keras.layers.GlobalAveragePooling2D()(mobilenet_features)
    mobilenet_features = Dropout(0.5)(mobilenet_features)
    
    # EAR features input
    ear_input = Input(shape=(ear_feature_dim,), name='ear_input')
    ear_dense = Dense(64, activation='relu')(ear_input)
    ear_dense = Dropout(0.3)(ear_dense)
    ear_dense = Dense(32, activation='relu')(ear_dense)
    
    # Concatenate features
    combined = Concatenate()([mobilenet_features, ear_dense])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    
    # Output layer (2 classes: Closed=0, Open=1)
    output = Dense(2, activation='softmax', name='output')(combined)
    
    model = Model(inputs=[image_input, ear_input], outputs=output)
    return model

def main():
    print("=" * 60)
    print("MobileNetV2 + EAR Feature Model Training")
    print("=" * 60)
    
    # Load dataset
    dataset_dir = 'Datasets'
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        return
    
    print("\n[1/5] Loading dataset...")
    images, ear_features, labels = load_dataset(dataset_dir)
    
    if len(images) == 0:
        print("Error: No images loaded!")
        return
    
    # Split dataset
    print("\n[2/5] Splitting dataset...")
    X_img_train, X_img_test, X_ear_train, X_ear_test, y_train, y_test = train_test_split(
        images, ear_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training: {len(X_img_train)} images")
    print(f"Testing: {len(X_img_test)} images")
    
    # Scale EAR features
    print("\n[3/5] Scaling EAR features...")
    scaler = StandardScaler()
    X_ear_train_scaled = scaler.fit_transform(X_ear_train)
    X_ear_test_scaled = scaler.transform(X_ear_test)
    
    # Save scaler
    with open('scaler_ear_mobilenet.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved scaler to: scaler_ear_mobilenet.pkl")
    
    # Build model
    print("\n[4/5] Building model...")
    model = build_model(ear_feature_dim=X_ear_train.shape[1])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint('eye_model_mobilenet.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Train model
    print("\n[5/5] Training model...")
    print("This may take a while...")
    
    history = model.fit(
        [X_img_train, X_ear_train_scaled], y_train,
        validation_data=([X_img_test, X_ear_test_scaled], y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    test_loss, test_accuracy = model.evaluate(
        [X_img_test, X_ear_test_scaled], y_test, verbose=0
    )
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"\nModel saved to: eye_model_mobilenet.h5")
    print(f"Scaler saved to: scaler_ear_mobilenet.pkl")
    
    # Save accuracy to file for eye tracker to display
    accuracy_info = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'model_type': 'MobileNetV2 + EAR + PERCLOS'
    }
    with open('model_accuracy.json', 'w') as f:
        json.dump(accuracy_info, f, indent=2)
    print("Accuracy saved to: model_accuracy.json")
    print("\nYou can now use the trained model with eye_tracker_mobilenet.py")

if __name__ == '__main__':
    main()













