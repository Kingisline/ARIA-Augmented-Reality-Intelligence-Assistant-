"""
ESP32-CAM  ·  Assistive Text Reader  ·  v1.1  (Aesthetic Edition)
══════════════════════════════════════════════════════════════════
Same core logic as v1.0 — only the rendering layer has been
redesigned.  ESP32 firmware is untouched.

Visual theme: Cyberpunk terminal
  ✦ Dark translucent top / bottom / side panels  (alpha blend)
  ✦ Corner-bracket detection markers  (no full rect clutter)
  ✦ Animated horizontal scan-line
  ✦ Neon-green  (#00FF9C) normal  /  red-orange  (#FF4545) DANGER
  ✦ Structured side panel  — live sentence + stats + controls
  ✦ Confidence heat-bar per detection
  ✦ CRT-style vignette
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
PADDING       = 10
PROCESS_EVERY = 5
LANGUAGES     = ['en']
SHOW_LABEL    = True
COOLDOWN      = 8.0

DANGER_WORDS  = frozenset({
    'exit','danger','warning','hot','caution','stop','fire',
    'emergency','poison','toxic','wet floor','high voltage',
    'restricted','biohazard','flammable','do not enter',
})

# ── Palette ────────────────────────────────────────────────────────────────────
C_NEON      = (0,   255, 156)   # #00FF9C  — normal text
C_DANGER    = (69,   69, 255)   # #FF4545  — danger text  (BGR)
C_DIM       = (0,   160,  90)   # dimmed green
C_WHITE     = (220, 220, 220)
C_PANEL     = (10,  12,  10)    # near-black panel bg
C_SCAN      = (0,   255, 120)   # scan-line tint
C_GOLD      = (30,  190, 255)   # accent gold  (BGR)
ALPHA_PANEL = 0.72
ALPHA_BOX   = 0.18
SIDE_W      = 280               # width of right-side info panel

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
        print(f"[INFO] Connecting … {i}/{retries}")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print(f"[OK] Connected: {url}")
            return cap
        time.sleep(delay)
    return None

# ── EasyOCR background thread ──────────────────────────────────────────────────
class TextDetector:
    def __init__(self):
        print("[INFO] Loading EasyOCR …")
        self.reader  = easyocr.Reader(LANGUAGES, gpu=False)
        self.results = []
        self.lock    = threading.Lock()
        self.busy    = False
        print("[OK]  EasyOCR ready")

    def submit(self, frame):
        if self.busy: return
        self.busy = True
        threading.Thread(target=self._run, args=(frame.copy(),),
                         daemon=True).start()

    def _run(self, frame):
        raw = self.reader.readtext(frame)
        with self.lock:
            self.results = [r for r in raw if r[2] >= CONFIDENCE]
        self.busy = False

    def get(self):
        with self.lock:
            return list(self.results)


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def blend_rect(canvas, x1, y1, x2, y2, color, alpha):
    """Filled rectangle with alpha transparency."""
    sub  = canvas[y1:y2, x1:x2]
    rect = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1-alpha, 0, sub)
    canvas[y1:y2, x1:x2] = sub


def corner_brackets(canvas, x1, y1, x2, y2, color, thickness=2, length=18):
    """Draw only the four corner brackets of a rectangle."""
    pts = [
        # top-left
        ((x1, y1+length),(x1, y1),(x1+length, y1)),
        # top-right
        ((x2-length, y1),(x2, y1),(x2, y1+length)),
        # bottom-left
        ((x1, y2-length),(x1, y2),(x1+length, y2)),
        # bottom-right
        ((x2-length, y2),(x2, y2),(x2, y2-length)),
    ]
    for group in pts:
        for i in range(len(group)-1):
            cv2.line(canvas, group[i], group[i+1], color, thickness,
                     cv2.LINE_AA)


def draw_confidence_bar(canvas, x, y, w, conf, color):
    """Small horizontal confidence heat-bar."""
    bar_w = int(w * conf)
    cv2.rectangle(canvas, (x, y), (x+w,  y+3), (40,40,40), -1)
    cv2.rectangle(canvas, (x, y), (x+bar_w, y+3), color, -1)


def draw_label_chip(canvas, text, x1, y1, color):
    """Slim pill-shaped label above bounding box."""
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs     = 0.42
    th     = 1
    (tw, fh), bl = cv2.getTextSize(text, font, fs, th)
    px, py = 5, 3
    lx = x1
    ly = max(y1 - 6, fh + py*2)
    blend_rect(canvas, lx, ly-fh-py, lx+tw+px*2, ly+bl+py,
               C_PANEL, 0.82)
    cv2.rectangle(canvas, (lx, ly-fh-py), (lx+tw+px*2, ly+bl+py),
                  color, 1, cv2.LINE_AA)
    cv2.putText(canvas, text, (lx+px, ly),
                font, fs, color, th, cv2.LINE_AA)


def draw_top_bar(canvas, mode_tag, n_found, fps, w):
    """Slim translucent top status bar."""
    blend_rect(canvas, 0, 0, w, 32, C_PANEL, ALPHA_PANEL)
    cv2.line(canvas, (0,32), (w,32), C_DIM, 1)
    # Left — mode
    cv2.putText(canvas, f"▶  {mode_tag}", (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_NEON, 1, cv2.LINE_AA)
    # Centre — title
    title = "ESP32-CAM  //  ASSISTIVE READER"
    (tw,_),_ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.putText(canvas, title, (w//2 - tw//2, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_GOLD, 1, cv2.LINE_AA)
    # Right — fps + count
    right = f"FPS {fps:4.1f}   TEXT {n_found:02d}"
    (rw,_),_ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(canvas, right, (w - rw - 10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)


def draw_side_panel(canvas, sentence, detections, h, w, n_spoken):
    """Right-side translucent info panel."""
    px = w - SIDE_W
    blend_rect(canvas, px, 32, w, h-28, C_PANEL, ALPHA_PANEL)
    cv2.line(canvas, (px, 32), (px, h-28), C_DIM, 1)

    y = 56
    def txt(s, color=C_DIM, scale=0.38, bold=False):
        nonlocal y
        cv2.putText(canvas, s, (px+10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    2 if bold else 1, cv2.LINE_AA)
        y += int(scale * 42 + 6)

    # ── Section: live sentence ─────────────────────────────────────────────
    txt("── DETECTED TEXT ──", C_GOLD, 0.38)
    if sentence:
        danger = is_danger(sentence)
        color  = C_DANGER if danger else C_NEON
        # Word-wrap at ~32 chars
        words  = sentence.split()
        line   = ""
        for word in words:
            if len(line) + len(word) + 1 > 28:
                txt(line, color, 0.40, bold=True)
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            txt(line, color, 0.40, bold=True)
        if danger:
            txt("⚠  DANGER WORD", C_DANGER, 0.37)
    else:
        txt("  —  no text  —", C_DIM, 0.37)

    y += 6
    cv2.line(canvas, (px+10, y), (w-10, y), C_DIM, 1); y += 10

    # ── Section: per-detection list ────────────────────────────────────────
    txt("── DETECTIONS ──", C_GOLD, 0.38)
    for i, (box, text, conf) in enumerate(detections[:8]):
        col  = C_DANGER if is_danger(text) else C_NEON
        tag  = f"{i+1}. {text[:22]}"
        txt(tag, col, 0.37)
        draw_confidence_bar(canvas, px+12, y-2, SIDE_W-28, conf, col)
        y += 5

    y += 6
    cv2.line(canvas, (px+10, y), (w-10, y), C_DIM, 1); y += 10

    # ── Section: stats ─────────────────────────────────────────────────────
    txt("── SESSION ──", C_GOLD, 0.38)
    txt(f"Utterances : {n_spoken}", C_WHITE, 0.37)
    txt(f"Active dets: {len(detections)}", C_WHITE, 0.37)

    # ── Controls footer ────────────────────────────────────────────────────
    y = h - 100
    cv2.line(canvas, (px+10, y), (w-10, y), C_DIM, 1); y += 8
    for line in ["Q  quit", "O  toggle view", "S  save frame"]:
        txt(line, C_DIM, 0.34)


def draw_bottom_bar(canvas, sentence, h, w):
    """Slim scrolling sentence at the very bottom."""
    blend_rect(canvas, 0, h-28, w - SIDE_W, h, C_PANEL, ALPHA_PANEL)
    cv2.line(canvas, (0, h-28), (w-SIDE_W, h-28), C_DIM, 1)
    if sentence:
        danger = is_danger(sentence)
        color  = C_DANGER if danger else C_NEON
    else:
        sentence = "awaiting text …"
        color    = C_DIM
    short = sentence[:(w - SIDE_W) // 8]  # crude char limit
    cv2.putText(canvas, short, (10, h-9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)


def draw_scan_line(canvas, tick, h, w):
    """Animated horizontal glowing scan-line."""
    y = int((tick * 1.8) % h)
    cv2.line(canvas, (0, y), (w - SIDE_W, y), C_SCAN, 1, cv2.LINE_AA)
    # soft glow above/below
    for off, alpha in [(1,0.15),(2,0.07),(3,0.03)]:
        for yo in (y-off, y+off):
            if 0 <= yo < h:
                sub = canvas[yo, :w-SIDE_W]
                glow = np.clip(sub.astype(np.int16) + np.array(
                    C_SCAN, dtype=np.int16) * alpha, 0, 255).astype(np.uint8)
                canvas[yo, :w-SIDE_W] = glow


def draw_vignette(canvas, h, w):
    """Dark CRT-style vignette on the camera region."""
    region_w = w - SIDE_W
    vig = np.zeros((h, region_w, 1), dtype=np.float32)
    cx, cy = region_w // 2, h // 2
    for y in range(h):
        for x in range(0, region_w, 4):   # stride for speed
            dx = (x - cx) / cx
            dy = (y - cy) / cy
            vig[y, x] = min(1.0, (dx*dx + dy*dy) * 0.55)
    # apply only to border region (optimised: use radial mask)
    mask = np.zeros((h, region_w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (cx, cy), 0, 0, 360, 1.0, -1)
    vig_mask = 1.0 - mask * 0.0 + (1.0 - mask) * 0.55
    vig_mask = np.clip(vig_mask, 0, 1)
    canvas[:, :region_w] = (
        canvas[:, :region_w].astype(np.float32) *
        vig_mask[:, :, np.newaxis]
    ).astype(np.uint8)


def add_vignette_fast(canvas, h, w):
    """Fast radial vignette using a precomputed mask."""
    region_w = w - SIDE_W
    Y, X = np.mgrid[0:h, 0:region_w].astype(np.float32)
    cx, cy = region_w / 2, h / 2
    dist = np.sqrt(((X - cx)/cx)**2 + ((Y - cy)/cy)**2)
    fade = np.clip(1.0 - dist * 0.5, 0.3, 1.0)
    canvas[:, :region_w] = (
        canvas[:, :region_w] * fade[:, :, np.newaxis]
    ).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global last_text, last_speak_time

    print(f"[INFO] Stream: {STREAM_URL}")
    cap = connect_camera(STREAM_URL)
    if cap is None:
        print("[ERROR] Could not connect"); return

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
    vignette_mask = None         # lazily computed on first frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame lost — reconnecting …")
            cap.release(); time.sleep(2)
            cap = connect_camera(STREAM_URL, retries=3, delay=2)
            if cap is None: break
            continue

        frame = cv2.flip(frame, -1)
        h, w  = frame.shape[:2]
        total_w = w + SIDE_W     # expand canvas for side panel

        # ── FPS ───────────────────────────────────────────────────────────────
        frame_count  += 1
        fps_counter  += 1
        tick         += 1
        elapsed = time.time() - fps_ts
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_ts = time.time(); fps_counter = 0

        # ── OCR ───────────────────────────────────────────────────────────────
        if frame_count % PROCESS_EVERY == 0:
            detector.submit(frame)
        detections = detector.get()

        # ── TTS ───────────────────────────────────────────────────────────────
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
                    threading.Thread(target=speak, args=(sentence,),
                                     daemon=True).start()
        else:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            last_text = ""; last_speak_time = 0; sentence = ""

        # ── Build canvas  (frame | side panel) ───────────────────────────────
        canvas = np.zeros((h, total_w, 3), dtype=np.uint8)

        if show_original:
            canvas[:, :w] = frame
            mode_tag = "LIVE  +  ANNOTATIONS"
        else:
            # text-mask view
            mask = np.zeros((h, w), dtype=np.uint8)
            for (box, _, _) in detections:
                x1,y1,x2,y2 = expand_box(box, PADDING, w, h)
                mask[y1:y2, x1:x2] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
            mask   = cv2.dilate(mask, kernel, iterations=2)
            canvas[:, :w] = cv2.bitwise_and(
                frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            mode_tag = "TEXT  ISOLATION"

        # ── Vignette (lazy init) ──────────────────────────────────────────────
        if vignette_mask is None or vignette_mask.shape[:2] != (h, w):
            Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
            dist  = np.sqrt(((X - w/2)/(w/2))**2 + ((Y - h/2)/(h/2))**2)
            vignette_mask = np.clip(1.0 - dist * 0.50, 0.28, 1.0)

        canvas[:, :w] = (canvas[:, :w] *
                         vignette_mask[:, :, np.newaxis]).astype(np.uint8)

        # ── Scan-line ─────────────────────────────────────────────────────────
        draw_scan_line(canvas, tick, h, total_w - SIDE_W)

        # ── OCR annotations ───────────────────────────────────────────────────
        if show_original:
            for (box, text, conf) in detections:
                danger = is_danger(text)
                col    = C_DANGER if danger else C_NEON
                x1,y1,x2,y2 = expand_box(box, PADDING, w, h)

                # Tinted fill
                blend_rect(canvas, x1, y1, x2, y2,
                           (0,80,40) if not danger else (0,30,100),
                           ALPHA_BOX)
                # Corner brackets
                corner_brackets(canvas, x1, y1, x2, y2, col, 2, 16)
                if SHOW_LABEL:
                    draw_label_chip(canvas,
                                    f"{text}  {conf:.0%}", x1, y1, col)
                    draw_confidence_bar(canvas, x1, y2+4,
                                        max(x2-x1, 20), conf, col)

        # ── Panels & HUD ──────────────────────────────────────────────────────
        draw_top_bar(canvas, mode_tag, len(detections), fps, total_w)
        draw_side_panel(canvas, sentence, detections,
                        h, total_w, n_spoken)
        draw_bottom_bar(canvas, sentence, h, total_w)

        # ── Vertical divider glow ─────────────────────────────────────────────
        cv2.line(canvas, (w, 32), (w, h-28), C_DIM, 1)

        cv2.imshow("ESP32-CAM  //  ASSISTIVE READER", canvas)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('o'): show_original = not show_original
        elif key == ord('s'):
            fname = f"capture_{int(time.time())}.png"
            cv2.imwrite(fname, canvas)
            print(f"[SAVED] {fname}")

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("[INFO] Session ended.")

if __name__ == "__main__":
    main()