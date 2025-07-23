import cv2
from fastai.vision.all import load_learner, PILImage

# 1. Load your trained learner from the data folder
THRESHOLD = 0.5
learn = load_learner('data/asl_classifier.pkl', cpu=True)

# 2. Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 3. Wrap into a PILImage for FastAI
    img = PILImage.create(frame)

    # 4. Predict
    pred, idx, probs = learn.predict(img)

    # 5. Only overlay if confidence > THRESHOLD
    if probs[idx] > THRESHOLD:
        label = f'{pred} ({probs[idx]*100:.1f}%)'
        cv2.putText(frame, label, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # 6. Show and quit on “q”
    cv2.imshow('ASL Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
