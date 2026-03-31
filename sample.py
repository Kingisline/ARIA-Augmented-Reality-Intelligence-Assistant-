import cv2
import easyocr
import numpy as np
import threading
import time
import re
import os
from gtts import gTTS
import pygame

# ── Config ────────────────────────────────────────────
IP            = "192.168.1.204"
STREAM_URL    = f"http://{IP}:81/stream"
CONFIDENCE    = 0.4
PADDING       = 10
PROCESS_EVERY = 5
LANGUAGES     = ['en']
SHOW_LABEL    = True
AUDIO_FILE    = "tts_output.mp3"
COOLDOWN      = 8.0   # seconds before repeating same text

# ── Pygame audio init ─────────────────────────────────
pygame.mixer.init()
print("[OK] Audio engine ready")

# ── TTS ───────────────────────────────────────────────
tts_lock      = threading.Lock()
last_text     = ""
last_speak_time = 0

def speak(text):
    with tts_lock:
        try:
            print(f"[TTS] Speaking: {text}")

            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            time.sleep(0.2)

            filename = f"tts_{int(time.time())}.mp3"
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filename)
            time.sleep(0.2)

            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            pygame.mixer.music.unload()
            os.remove(filename)
            print("[TTS] Done")

        except Exception as e:
            print(f"[ERROR] TTS failed: {e}")

# ── Helper functions ──────────────────────────────────
def expand_box(box, pad, w, h):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x1 = max(int(min(xs)) - pad, 0)
    y1 = max(int(min(ys)) - pad, 0)
    x2 = min(int(max(xs)) + pad, w)
    y2 = min(int(max(ys)) + pad, h)
    return x1, y1, x2, y2

def draw_label(frame, text, x1, y1):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    lx, ly = x1, max(y1 - 6, th + 4)
    cv2.rectangle(frame, (lx, ly - th - 4),
                  (lx + tw + 4, ly + baseline), (0, 200, 80), -1)
    cv2.putText(frame, text, (lx + 2, ly),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

def build_text_only_frame(frame, detections):
    
    h, w  = frame.shape[:2]
    mask  = np.zeros((h, w), dtype=np.uint8)
    for (box, text, conf) in detections:
        
        x1, y1, x2, y2 = expand_box(box, PADDING, w, h)
        mask[y1:y2, x1:x2] = 255
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask     = cv2.dilate(mask, kernel, iterations=2)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(frame, mask_3ch)

def connect_camera(url, retries=5, delay=3):
    for attempt in range(1, retries + 1):
        print(f"[INFO] Connecting … attempt {attempt}/{retries}")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print(f"[OK] Connected: {url}")
            return cap
        print(f"[WARN] Failed. Retrying in {delay}s …")
        time.sleep(delay)
    return None

# ── EasyOCR background thread ─────────────────────────
class TextDetector:
    def __init__(self):
        print("[INFO] Loading EasyOCR model ...")
        self.reader  = easyocr.Reader(LANGUAGES, gpu=False)
        self.results = []
        self.lock    = threading.Lock()
        self.busy    = False
        print("[INFO] Model ready")

    def submit(self, frame):
        if self.busy:
            return
        self.busy = True
        threading.Thread(
            target=self._run,
            args=(frame.copy(),),
            daemon=True
        ).start()

    def _run(self, frame):
        raw = self.reader.readtext(frame)
        with self.lock:
            self.results = [r for r in raw if r[2] >= CONFIDENCE]
        self.busy = False

    def get(self):
        with self.lock:
            return list(self.results)

# ── Combine all detected words into one clean sentence ─
def build_sentence(detections):
    words = []
    for (box, text, conf) in detections:
        clean = text.strip()
        clean = ''.join(c for c in clean if ord(c) < 128 and c.isprintable())
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', clean).strip()
        if len(clean) >= 2:
            words.append(clean)
    return ' '.join(words)

# ── Main ──────────────────────────────────────────────
def main():
    global last_text, last_speak_time

    print(f"[INFO] Stream: {STREAM_URL}")
    cap = connect_camera(STREAM_URL)

    if cap is None:
        print("[ERROR] Could not connect to camera")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector      = TextDetector()
    frame_count   = 0
    show_original = False
    detections    = []

    print("\n[CONTROLS]")
    print("  q  —  quit")
    print("  o  —  toggle original / text-only view")
    print("  s  —  save current frame\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[WARN] Frame lost. Reconnecting ...")
            cap.release()
            time.sleep(2)
            cap = connect_camera(STREAM_URL, retries=3, delay=2)
            if cap is None:
                print("[ERROR] Reconnect failed")
                break
            continue

        # Fix inverted camera
        frame = cv2.flip(frame, -1)

        frame_count += 1

        # Submit frame to EasyOCR every N frames
        if frame_count % PROCESS_EVERY == 0:
            detector.submit(frame)

        detections = detector.get()

        # ── Speak detected text ───────────────────────
        # ── Speak detected text ───────────────────────
        if detections:
            sentence = build_sentence(detections)
            now      = time.time()

            if sentence and len(sentence) > 3:
                is_new      = sentence != last_text
                is_cooldown = (now - last_speak_time) >= COOLDOWN

                if is_new or is_cooldown:
                    print(f"[DETECTED] {sentence}")
                    last_text       = sentence
                    last_speak_time = now
                    threading.Thread(
                        target=speak,
                        args=(sentence,),
                        daemon=True
                    ).start()
        else:
            # No text found — stop audio and reset
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                print("[INFO] No text detected — audio stopped")
            last_text       = ""
            last_speak_time = 0
            sentence = build_sentence(detections)
            now      = time.time()

            if sentence and len(sentence) > 3:
                is_new      = sentence != last_text
                is_cooldown = (now - last_speak_time) >= COOLDOWN

                if is_new or is_cooldown:
                    print(f"[DETECTED] {sentence}")
                    last_text       = sentence
                    last_speak_time = now
                    threading.Thread(
                        target=speak,
                        args=(sentence,),
                        daemon=True
                    ).start()

        # ── Display ───────────────────────────────────
        if show_original:
            display = frame.copy()
            for (box, text, conf) in detections:
                pts = np.array(
                    [[int(p[0]), int(p[1])] for p in box],
                    dtype=np.int32)
                cv2.polylines(display, [pts], True, (0, 255, 80), 2)
                if SHOW_LABEL:
                    x1, y1, _, _ = expand_box(
                        box, 0, frame.shape[1], frame.shape[0])
                    draw_label(display,
                               f"{text} ({conf:.0%})", x1, y1)
            mode_tag = "ORIGINAL + BOXES"
        else:
            display = build_text_only_frame(frame, detections)
            if SHOW_LABEL:
                h, w = frame.shape[:2]
                for (box, text, conf) in detections:
                    x1, y1, _, _ = expand_box(box, PADDING, w, h)
                    draw_label(display,
                               f"{text} ({conf:.0%})", x1, y1)
            mode_tag = "TEXT ONLY"

        # HUD
        cv2.rectangle(display, (0, 0), (420, 28), (0, 0, 0), -1)
        hud = (f"[{mode_tag}]  found: {len(detections)}"
               f"  | O=toggle  S=save  Q=quit")
        cv2.putText(display, hud, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 220, 100), 1, cv2.LINE_AA)

        cv2.imshow("ESP32-CAM — Text Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            show_original = not show_original
        elif key == ord('s'):
            fname = f"capture_{int(time.time())}.png"
            cv2.imwrite(fname, display)
            print(f"[SAVED] {fname}")

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("[INFO] Done")

if __name__ == "__main__":
    main()