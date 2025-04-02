import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Captura de video
cap = cv2.VideoCapture(0)

# Parámetros de detección
EAR_THRESHOLD = 0.25  # Umbral para detección de parpadeo
MOVEMENT_THRESHOLD = 15  # Umbral de movimiento en píxeles
CONSECUTIVE_FRAMES = 3   # Mínimo de frames para considerar parpadeo
BLINK_RATIO = 0.2        # Proporción de ancho/alto para detección de foto

# Índices de landmarks para ojos y nariz
LEFT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
NOSE_TIP = 4

# Variables de seguimiento
blink_counter = 0
movement_history = deque(maxlen=10)
blink_history = deque(maxlen=30)
head_position = None

def distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye_points, landmarks, frame_shape):
    """Calcula la relación de aspecto del ojo (EAR)"""
    points = [(landmarks.landmark[i].x * frame_shape[1], 
               landmarks.landmark[i].y * frame_shape[0]) 
              for i in eye_points]
    
    # Calcular distancias verticales
    vert1 = distancia(points[1], points[5])
    vert2 = distancia(points[2], points[4])
    
    # Calcular distancia horizontal
    horiz = distancia(points[0], points[3])
    
    ear = (vert1 + vert2) / (2.0 * horiz)
    return ear

def detect_blink(face_landmarks, frame_shape):
    """Detecta parpadeo en ambos ojos"""
    left_ear = eye_aspect_ratio(LEFT_EYE_POINTS, face_landmarks, frame_shape)
    right_ear = eye_aspect_ratio(RIGHT_EYE_POINTS, face_landmarks, frame_shape)
    return (left_ear + right_ear) / 2.0

def detect_photo(face_landmarks, frame_shape):
    """Detección de foto basada en relación de aspecto facial"""
    # Puntos para relación de aspecto
    horizontal = [33, 263]  # Esquinas de ojos
    vertical = [10, 152]    # Frente y barbilla
    
    h_dist = distancia(
        (face_landmarks.landmark[horizontal[0]].x * frame_shape[1],
         face_landmarks.landmark[horizontal[0]].y * frame_shape[0]),
        (face_landmarks.landmark[horizontal[1]].x * frame_shape[1],
         face_landmarks.landmark[horizontal[1]].y * frame_shape[0]))
    
    v_dist = distancia(
        (face_landmarks.landmark[vertical[0]].x * frame_shape[1],
         face_landmarks.landmark[vertical[0]].y * frame_shape[0]),
        (face_landmarks.landmark[vertical[1]].x * frame_shape[1],
         face_landmarks.landmark[vertical[1]].y * frame_shape[0]))
    
    return (h_dist / v_dist) < BLINK_RATIO

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    is_real = False
    current_movement = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Detección de parpadeo
            ear = detect_blink(face_landmarks, frame.shape)
            blink_history.append(ear < EAR_THRESHOLD)
            
            # Detección de movimiento
            nose_point = (int(face_landmarks.landmark[NOSE_TIP].x * frame.shape[1]),
                          int(face_landmarks.landmark[NOSE_TIP].y * frame.shape[0]))
            
            if head_position:
                current_movement = distancia(head_position, nose_point)
                movement_history.append(current_movement)
            
            head_position = nose_point
            
            # Detección de foto
            photo_detected = detect_photo(face_landmarks, frame.shape)
            
            # Lógica de detección
            if photo_detected:
                is_real = False
            else:
                # Verificar parpadeos recientes (al menos 1 en últimos 30 frames)
                if sum(blink_history) >= 1:
                    is_real = True
                # Verificar movimiento reciente
                if len(movement_history) > 5 and sum(movement_history) > MOVEMENT_THRESHOLD * 5:
                    is_real = True

            # Dibujar información
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Movement: {current_movement:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, "Status: REAL" if is_real else "Status: PHOTO", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 255, 0) if is_real else (0, 0, 255), 2)

    cv2.imshow('Liveness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()