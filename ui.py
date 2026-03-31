"""
ESP32-CAM  ·  Assistive Text Reader  ·  v2.0  (J.A.R.V.I.S. Edition)
══════════════════════════════════════════════════════════════════
Visual theme: Advanced Holographic HUD
  ✦ Stark Tech Palette (Cyan, Ice Blue, Gold, Warning Red)
  ✦ Rotating circular central HUD elements
  ✦ Advanced targeting reticles with joint-nodes
  ✦ Clean, technical font faces
  ✦ Blueprint-style data side panel
"""

import cv2
import easyocr
import numpy as np
import threading
import time
import re
import os
from gtts import gTTS
import pygame

# ── Config ─────────────────────────────────────────────────────────────────────
IP            = "192.168.1.204"
STREAM_URL    = f"http://{IP}:81/stream"
CONFIDENCE    = 0.4
PADDING       = 15
PROCESS_EVERY = 5
LANGUAGES     = ['en']
SHOW_LABEL    = True
COOLDOWN      = 8.0

DANGER_WORDS  = frozenset({
    'exit','danger','warning','hot','caution','stop','fire',
    'emergency','poison','toxic','wet floor','high voltage',
    'restricted','biohazard','flammable','do not enter',
})

# ── J.A.R.V.I.S. Palette (BGR Format) ──────────────────────────────────────────
C_MAIN      = (255, 230, 100)  # Ice Blue
C_ACCENT    = (255, 255, 200)  # Bright Cyan
C_DANGER    = (40, 40, 255)    # High-Alert Red
C_DIM       = (100, 80, 20)    # Faded Blue/Grey for backgrounds
C_WHITE     = (240, 250, 250)
C_PANEL     = (25, 15, 5)      # Deep holographic blue-black
C_SCAN      = (255, 200, 50)   # Cyan scan line
C_GOLD      = (0, 180, 255)    # Stark Gold
ALPHA_PANEL = 0.65             # Slightly more transparent
ALPHA_BOX   = 0.15
SIDE_W      = 300              # Slightly wider for blueprint look
HUD_FONT    = cv2.FONT_HERSHEY_COMPLEX_SMALL # Cleaner tech font

# ── Audio ──────────────────────────────────────────────────────────────────────
pygame.mixer.init()
tts_lock        = threading.Lock()
last_text       = ""
last_speak_time = 0

def speak(text):
    with tts_lock:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            time.sleep(0.15)
            fname = f"tts_{int(time.time())}.mp3"
            gTTS(text=text, lang='en', slow=False).save(fname)
            time.sleep(0.15)
            pygame.mixer.music.load(fname)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
            os.remove(fname)
        except Exception as e:
            print(f"[ERROR] TTS: {e}")

# ── Helpers ────────────────────────────────────────────────────────────────────
def expand_box(box, pad, w, h):
    xs = [p[0] for p in box]; ys = [p[1] for p in box]
    return (max(int(min(xs))-pad,0), max(int(min(ys))-pad,0),
            min(int(max(xs))+pad,w), min(int(max(ys))+pad,h))

def is_danger(text):
    return any(dw in text.lower() for dw in DANGER_WORDS)

def build_sentence(detections):
    words = []
    for (_, text, _) in detections:
        clean = ''.join(c for c in text.strip() if ord(c)<128 and c.isprintable())
        clean = re.sub(r'[^a-zA-Z0-9\s]','',clean).strip()
        if len(clean) >= 2:
            words.append(clean)
    return ' '.join(words)

def connect_camera(url, retries=5, delay=3):
    for i in range(1, retries+1):
        print(f"[SYSTEM] Establishing link ... {i}/{retries}")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print(f"[ONLINE] Link established: {url}")
            return cap
        time.sleep(delay)
    return None

# ── EasyOCR ────────────────────────────────────────────────────────────────────
class TextDetector:
    def __init__(self):
        print("[SYSTEM] Initializing Optical Recognition ...")
        self.reader  = easyocr.Reader(LANGUAGES, gpu=False)
        self.results = []
        self.lock    = threading.Lock()
        self.busy    = False
        print("[ONLINE] Optics ready.")

    def submit(self, frame):
        if self.busy: return
        self.busy = True
        threading.Thread(target=self._run, args=(frame.copy(),), daemon=True).start()

    def _run(self, frame):
        raw = self.reader.readtext(frame)
        with self.lock:
            self.results = [r for r in raw if r[2] >= CONFIDENCE]
        self.busy = False

    def get(self):
        with self.lock:
            return list(self.results)


# ══════════════════════════════════════════════════════════════════════════════
#  HUD RENDER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def blend_rect(canvas, x1, y1, x2, y2, color, alpha):
    sub  = canvas[y1:y2, x1:x2]
    rect = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1-alpha, 0, sub)
    canvas[y1:y2, x1:x2] = sub

def draw_target_reticle(canvas, x1, y1, x2, y2, color, thickness=1, length=20):
    """Draws high-tech targeting brackets with joint nodes."""
    pts = [
        ((x1, y1+length),(x1, y1),(x1+length, y1)),
        ((x2-length, y1),(x2, y1),(x2, y1+length)),
        ((x1, y2-length),(x1, y2),(x1+length, y2)),
        ((x2-length, y2),(x2, y2),(x2, y2-length)),
    ]
    for group in pts:
        for i in range(len(group)-1):
            cv2.line(canvas, group[i], group[i+1], color, thickness, cv2.LINE_AA)
        # Add a node at the corner
        cv2.circle(canvas, group[1], 2, color, -1, cv2.LINE_AA)

def draw_jarvis_ring(canvas, tick, h, w):
    """Draws a rotating holographic circle in the center of the camera feed."""
    cx, cy = (w - SIDE_W) // 2, h // 2
    radius = min(cx, cy) - 60
    
    # Outer static dashed ring
    cv2.circle(canvas, (cx, cy), radius + 20, C_DIM, 1, cv2.LINE_AA)
    
    # Inner rotating arcs
    angle1 = (tick * 2) % 360
    angle2 = (-tick * 1.5) % 360
    
    cv2.ellipse(canvas, (cx, cy), (radius, radius), angle1, 0, 120, C_MAIN, 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (radius, radius), angle1, 180, 300, C_MAIN, 2, cv2.LINE_AA)
    
    cv2.ellipse(canvas, (cx, cy), (radius-10, radius-10), angle2, 0, 90, C_GOLD, 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (radius-10, radius-10), angle2, 180, 270, C_GOLD, 1, cv2.LINE_AA)

    # Center crosshair
    cv2.line(canvas, (cx - 15, cy), (cx - 5, cy), C_ACCENT, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx + 5, cy), (cx + 15, cy), C_ACCENT, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - 15), (cx, cy - 5), C_ACCENT, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy + 5), (cx, cy + 15), C_ACCENT, 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 1, C_ACCENT, -1, cv2.LINE_AA)

def draw_confidence_bar(canvas, x, y, w, conf, color):
    bar_w = int(w * conf)
    cv2.rectangle(canvas, (x, y), (x+w,  y+2), C_DIM, -1)
    cv2.rectangle(canvas, (x, y), (x+bar_w, y+2), color, -1)

def draw_label_chip(canvas, text, x1, y1, color):
    fs     = 0.5
    th     = 1
    (tw, fh), bl = cv2.getTextSize(text, HUD_FONT, fs, th)
    px, py = 6, 4
    lx = x1
    ly = max(y1 - 8, fh + py*2)
    
    # Blueprint style fill
    blend_rect(canvas, lx, ly-fh-py, lx+tw+px*2, ly+bl+py, C_PANEL, 0.8)
    cv2.line(canvas, (lx, ly+bl+py), (lx+tw+px*2, ly+bl+py), color, 1, cv2.LINE_AA) # Bottom underline
    cv2.putText(canvas, text, (lx+px, ly), HUD_FONT, fs, color, th, cv2.LINE_AA)

def draw_top_bar(canvas, mode_tag, n_found, fps, w):
    blend_rect(canvas, 0, 0, w, 32, C_PANEL, ALPHA_PANEL)
    cv2.line(canvas, (0,32), (w,32), C_MAIN, 1)
    
    cv2.putText(canvas, f"SYS.MODE: {mode_tag}", (15, 21), HUD_FONT, 0.6, C_MAIN, 1, cv2.LINE_AA)
    
    title = "MARK II // ASSISTIVE OPTICS"
    (tw,_),_ = cv2.getTextSize(title, HUD_FONT, 0.6, 1)
    cv2.putText(canvas, title, (w//2 - tw//2, 21), HUD_FONT, 0.6, C_GOLD, 1, cv2.LINE_AA)
    
    right = f"FPS: {fps:4.1f} | TGT: {n_found:02d}"
    (rw,_),_ = cv2.getTextSize(right, HUD_FONT, 0.6, 1)
    cv2.putText(canvas, right, (w - rw - 15, 21), HUD_FONT, 0.6, C_ACCENT, 1, cv2.LINE_AA)

def draw_side_panel(canvas, sentence, detections, h, w, n_spoken):
    px = w - SIDE_W
    blend_rect(canvas, px, 32, w, h-28, C_PANEL, ALPHA_PANEL)
    cv2.line(canvas, (px, 32), (px, h-28), C_MAIN, 1)

    y = 60
    def txt(s, color=C_MAIN, scale=0.55, bold=False):
        nonlocal y
        cv2.putText(canvas, s, (px+15, y), HUD_FONT, scale, color, 2 if bold else 1, cv2.LINE_AA)
        y += int(scale * 45 + 8)

    # Live Analysis
    txt("[ ANALYSIS ]", C_GOLD, 0.6)
    if sentence:
        danger = is_danger(sentence)
        color  = C_DANGER if danger else C_ACCENT
        words  = sentence.split()
        line   = ""
        for word in words:
            if len(line) + len(word) + 1 > 26:
                txt(line, color, 0.6, bold=True)
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            txt(line, color, 0.6, bold=True)
        if danger:
            txt(">> CRITICAL ALERT <<", C_DANGER, 0.55)
    else:
        txt("AWAITING INPUT...", C_DIM, 0.55)

    y += 10
    cv2.line(canvas, (px+15, y), (w-15, y), C_DIM, 1); y += 15

    # Targets
    txt("[ ACTIVE TARGETS ]", C_GOLD, 0.6)
    for i, (box, text, conf) in enumerate(detections[:6]):
        col  = C_DANGER if is_danger(text) else C_MAIN
        tag  = f"TGT_{i:02d} : {text[:18]}"
        txt(tag, col, 0.5)
        draw_confidence_bar(canvas, px+15, y-4, SIDE_W-30, conf, col)
        y += 8

    y += 10
    cv2.line(canvas, (px+15, y), (w-15, y), C_DIM, 1); y += 15

    # Diagnostics
    txt("[ DIAGNOSTICS ]", C_GOLD, 0.6)
    txt(f"AUDIO CYCLES : {n_spoken}", C_WHITE, 0.5)
    txt(f"DATA STREAMS : {len(detections)}", C_WHITE, 0.5)

    # Footer Controls
    y = h - 90
    cv2.line(canvas, (px+15, y), (w-15, y), C_DIM, 1); y += 15
    for line in ["[Q] DISCONNECT", "[O] TOGGLE OPTICS", "[S] EXPORT DATA"]:
        txt(line, C_DIM, 0.5)

def draw_bottom_bar(canvas, sentence, h, w):
    blend_rect(canvas, 0, h-28, w - SIDE_W, h, C_PANEL, ALPHA_PANEL)
    cv2.line(canvas, (0, h-28), (w-SIDE_W, h-28), C_MAIN, 1)
    
    if sentence:
        danger = is_danger(sentence)
        color  = C_DANGER if danger else C_ACCENT
    else:
        sentence = "SCANNING ENVIRONMENT..."
        color    = C_DIM
        
    short = f"> {sentence[:(w - SIDE_W) // 10]}" 
    cv2.putText(canvas, short, (15, h-10), HUD_FONT, 0.6, color, 1, cv2.LINE_AA)

def add_vignette_fast(canvas, h, w):
    region_w = w - SIDE_W
    Y, X = np.mgrid[0:h, 0:region_w].astype(np.float32)
    cx, cy = region_w / 2, h / 2
    dist = np.sqrt(((X - cx)/cx)**2 + ((Y - cy)/cy)**2)
    fade = np.clip(1.0 - dist * 0.4, 0.2, 1.0)
    canvas[:, :region_w] = (canvas[:, :region_w] * fade[:, :, np.newaxis]).astype(np.uint8)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global last_text, last_speak_time

    print(f"[SYSTEM] Initializing uplink to {STREAM_URL}")
    cap = connect_camera(STREAM_URL)
    if cap is None:
        print("[ERROR] Uplink failed."); return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector      = TextDetector()
    frame_count   = 0
    fps           = 0.0
    fps_ts        = time.time()
    fps_counter   = 0
    show_original = True
    detections    = []
    sentence      = ""
    tick          = 0
    n_spoken      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Signal lost — attempting reconnection...")
            cap.release(); time.sleep(2)
            cap = connect_camera(STREAM_URL, retries=3, delay=2)
            if cap is None: break
            continue

        frame = cv2.flip(frame, -1)
        h, w  = frame.shape[:2]
        total_w = w + SIDE_W 

        # FPS
        frame_count  += 1
        fps_counter  += 1
        tick         += 1
        elapsed = time.time() - fps_ts
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_ts = time.time(); fps_counter = 0

        # OCR
        if frame_count % PROCESS_EVERY == 0:
            detector.submit(frame)
        detections = detector.get()

        # TTS
        if detections:
            sentence = build_sentence(detections)
            now      = time.time()
            if len(sentence) > 3:
                is_new      = sentence != last_text
                is_cooldown = (now - last_speak_time) >= COOLDOWN
                if is_new or is_cooldown:
                    last_text       = sentence
                    last_speak_time = now
                    n_spoken       += 1
                    threading.Thread(target=speak, args=(sentence,), daemon=True).start()
        else:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            last_text = ""; last_speak_time = 0; sentence = ""

        # Canvas Setup
        canvas = np.zeros((h, total_w, 3), dtype=np.uint8)

        if show_original:
            canvas[:, :w] = frame
            mode_tag = "OPTICAL"
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            for (box, _, _) in detections:
                x1,y1,x2,y2 = expand_box(box, PADDING, w, h)
                mask[y1:y2, x1:x2] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
            mask   = cv2.dilate(mask, kernel, iterations=2)
            canvas[:, :w] = cv2.bitwise_and(frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            mode_tag = "ISOLATION"

        # Apply Vignette & Central HUD
        add_vignette_fast(canvas, h, total_w)
        draw_jarvis_ring(canvas, tick, h, total_w)

        # Draw Annotations
        if show_original:
            for (box, text, conf) in detections:
                danger = is_danger(text)
                col    = C_DANGER if danger else C_MAIN
                x1,y1,x2,y2 = expand_box(box, PADDING, w, h)

                blend_rect(canvas, x1, y1, x2, y2, (0,50,50) if not danger else (0,0,80), ALPHA_BOX)
                draw_target_reticle(canvas, x1, y1, x2, y2, col, 1, 15)
                
                if SHOW_LABEL:
                    draw_label_chip(canvas, f"{text} [{conf:.0%}]", x1, y1, col)

        # Draw UI Panels
        draw_top_bar(canvas, mode_tag, len(detections), fps, total_w)
        draw_side_panel(canvas, sentence, detections, h, total_w, n_spoken)
        draw_bottom_bar(canvas, sentence, h, total_w)

        cv2.imshow("MARK II // OPTICS", canvas)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('o'): show_original = not show_original
        elif key == ord('s'):
            fname = f"capture_{int(time.time())}.png"
            cv2.imwrite(fname, canvas)
            print(f"[SYSTEM] Data exported to {fname}")

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("[SYSTEM] Powering down.")

if __name__ == "__main__":
    main()