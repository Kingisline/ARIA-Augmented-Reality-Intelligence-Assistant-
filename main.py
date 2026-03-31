"""
ESP32-CAM Assistive Text & Object Reader  —  v2.0
══════════════════════════════════════════════════
Upgrades over v1:
  ✦ Motion-triggered OCR  (frame-diff, saves CPU)
  ✦ Multi-frame confidence-weighted text merging
  ✦ Spell correction before TTS  (pyspellchecker)
  ✦ Named Entity Recognition  (spaCy en_core_web_sm)
  ✦ Priority TTS queue  (DANGER words preempt audio)
  ✦ Directional stereo audio  (pydub pan by text position)
  ✦ YOLOv8 object detection with OCR context
  ✦ SQLite logging of every detection
  ✦ Live FPS / motion / object HUD overlay
  ✦ 'r' key prints DB summary in terminal
"""

import cv2
import easyocr
import numpy as np
import threading
import time
import re
import os
import sqlite3
import queue
from datetime import datetime
from gtts import gTTS
import pygame
from pydub import AudioSegment
from spellchecker import SpellChecker
import spacy
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────────
IP               = "192.168.1.204"
STREAM_URL       = f"http://{IP}:81/stream"
OCR_CONFIDENCE   = 0.40
PADDING          = 10
LANGUAGES        = ['en']
SHOW_LABEL       = True
COOLDOWN         = 8.0           # seconds before repeating same sentence
MOTION_THRESHOLD = 4000          # pixel-diff sum to trigger OCR
HISTORY_DEPTH    = 5             # frames kept for text merging
DANGER_WORDS     = frozenset({
    'exit', 'danger', 'warning', 'hot', 'caution', 'stop', 'fire',
    'emergency', 'poison', 'toxic', 'wet floor', 'high voltage',
    'do not enter', 'restricted', 'biohazard', 'flammable',
})
DB_PATH          = "detections.db"
YOLO_MODEL       = "yolov8n.pt"   # auto-downloaded on first run (~6 MB)
YOLO_CONFIDENCE  = 0.45
YOLO_EVERY       = 10             # run YOLO every N frames


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE LOGGER
# ══════════════════════════════════════════════════════════════════════════════
class Logger:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_tables()
        print(f"[OK] Database ready → {db_path}")

    def _init_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS text_detections (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    raw_text    TEXT    NOT NULL,
                    corrected   TEXT,
                    entities    TEXT,
                    priority    TEXT,
                    avg_conf    REAL
                )""")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS object_detections (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    label       TEXT    NOT NULL,
                    confidence  REAL,
                    bbox        TEXT
                )""")

    def log_text(self, raw, corrected, entities, priority, avg_conf):
        with self._lock:
            self.conn.execute(
                "INSERT INTO text_detections VALUES (NULL,?,?,?,?,?,?)",
                (datetime.now().isoformat(), raw, corrected,
                 str(entities), priority, round(avg_conf, 4)))
            self.conn.commit()

    def log_objects(self, detections: list):
        ts = datetime.now().isoformat()
        with self._lock:
            self.conn.executemany(
                "INSERT INTO object_detections VALUES (NULL,?,?,?,?)",
                [(ts, lbl, round(conf, 4), str(bbox))
                 for (lbl, conf, bbox) in detections])
            self.conn.commit()

    def summary(self) -> tuple:
        with self._lock:
            t = self.conn.execute(
                "SELECT COUNT(*) FROM text_detections").fetchone()[0]
            o = self.conn.execute(
                "SELECT COUNT(*) FROM object_detections").fetchone()[0]
        return t, o

    def recent_texts(self, n=10) -> list:
        with self._lock:
            rows = self.conn.execute(
                "SELECT timestamp, corrected, priority FROM text_detections "
                "ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return rows


# ══════════════════════════════════════════════════════════════════════════════
#  NLP PIPELINE  (spell-check + NER + priority classifier)
# ══════════════════════════════════════════════════════════════════════════════
class NLPPipeline:
    def __init__(self):
        self.spell = SpellChecker()

        print("[INFO] Loading spaCy model …")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("[OK] spaCy ready")
        except OSError:
            print("[WARN] spaCy model missing. Run:\n"
                  "       python -m spacy download en_core_web_sm")
            self.nlp = None

    # ── Spell correction ──────────────────────────────────────────────────────
    def correct(self, text: str) -> str:
        words     = text.split()
        corrected = []
        for w in words:
            fix = self.spell.correction(w.lower())
            corrected.append(fix if fix else w)
        return ' '.join(corrected)

    # ── Named entity recognition ──────────────────────────────────────────────
    def entities(self, text: str) -> list:
        if self.nlp is None:
            return []
        return [(e.text, e.label_) for e in self.nlp(text).ents]

    # ── Danger classifier ─────────────────────────────────────────────────────
    def priority(self, text: str) -> str:
        lower = text.lower()
        return 'DANGER' if any(dw in lower for dw in DANGER_WORDS) else 'NORMAL'

    # ── Build a richer TTS sentence ───────────────────────────────────────────
    @staticmethod
    def enrich(corrected: str, ents: list, objects: list) -> str:
        _LABEL_MAP = {
            'PERSON': 'person named', 'ORG': 'organisation called',
            'GPE': 'place called',   'DATE': 'date',
            'TIME': 'time',          'MONEY': 'amount',
            'PRODUCT': 'product called',
        }
        parts = []
        for txt, lbl in ents[:2]:
            spoken = _LABEL_MAP.get(lbl)
            if spoken:
                parts.append(f"{spoken} {txt}")

        if objects:
            unique = list(dict.fromkeys(o[0] for o in objects))[:3]
            parts.append("near " + ', '.join(unique))

        return corrected + ('. ' + '. '.join(parts) if parts else '')


# ══════════════════════════════════════════════════════════════════════════════
#  PRIORITY TTS ENGINE  with directional stereo panning
# ══════════════════════════════════════════════════════════════════════════════
class TTSEngine:
    """
    Queued TTS.  Priority 0 = DANGER (preempts playback).
    pan:  -1.0 full-left … 0.0 centre … +1.0 full-right
    """
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self._q       = queue.PriorityQueue()
        self._counter = 0
        self._lock    = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
        print("[OK] TTS engine ready")

    def speak(self, text: str, pan: float = 0.0, priority: str = 'NORMAL'):
        p = 0 if priority == 'DANGER' else 1
        with self._lock:
            self._counter += 1
            self._q.put((p, self._counter, text, pan))
        if priority == 'DANGER':
            pygame.mixer.music.stop()   # immediate interrupt

    def _loop(self):
        while True:
            try:
                _, _, text, pan = self._q.get(timeout=1)
                self._play(text, pan)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] TTS loop: {e}")

    def _play(self, text: str, pan: float):
        ts     = int(time.time() * 1000)
        raw    = f"_raw_{ts}.mp3"
        panned = f"_pan_{ts}.mp3"
        try:
            print(f"[TTS] pan={pan:+.2f}  '{text[:60]}'")
            gTTS(text=text, lang='en', slow=False).save(raw)
            AudioSegment.from_mp3(raw).pan(pan).export(panned, format="mp3")

            pygame.mixer.music.stop()
            pygame.mixer.music.load(panned)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.unload()
        except Exception as e:
            print(f"[ERROR] TTS play: {e}")
        finally:
            for f in (raw, panned):
                try: os.remove(f)
                except: pass


# ══════════════════════════════════════════════════════════════════════════════
#  MOTION DETECTOR  (frame differencing)
# ══════════════════════════════════════════════════════════════════════════════
class MotionDetector:
    def __init__(self, threshold: int = MOTION_THRESHOLD):
        self.threshold = threshold
        self._prev     = None

    def check(self, frame) -> bool:
        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 0)
        if self._prev is None:
            self._prev = gray
            return True
        score      = int(np.sum(cv2.absdiff(self._prev, gray)))
        self._prev = gray
        return score > self.threshold


# ══════════════════════════════════════════════════════════════════════════════
#  EASYOCR TEXT DETECTOR  (background thread + multi-frame merge)
# ══════════════════════════════════════════════════════════════════════════════
class TextDetector:
    def __init__(self):
        print("[INFO] Loading EasyOCR model …")
        self.reader   = easyocr.Reader(LANGUAGES, gpu=False)
        self._lock    = threading.Lock()
        self._results = []
        self._history = []          # rolling window of raw results
        self.busy     = False
        print("[OK] EasyOCR ready")

    def submit(self, frame):
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._run, args=(frame.copy(),),
                         daemon=True).start()

    def _run(self, frame):
        raw = self.reader.readtext(frame)
        new = [r for r in raw if r[2] >= OCR_CONFIDENCE]
        with self._lock:
            self._results = new
            self._history.append(new)
            if len(self._history) > HISTORY_DEPTH:
                self._history.pop(0)
        self.busy = False

    def get_raw(self) -> list:
        with self._lock:
            return list(self._results)

    def get_merged(self) -> list:
        """
        Confidence-weighted merge across the rolling history window.
        Returns list of (box, text, conf) keeping best conf per unique word.
        """
        with self._lock:
            seen: dict = {}
            for frame_res in self._history:
                for (box, text, conf) in frame_res:
                    key = text.strip().lower()
                    if key not in seen or seen[key][2] < conf:
                        seen[key] = (box, text, conf)
            return list(seen.values())


# ══════════════════════════════════════════════════════════════════════════════
#  YOLO OBJECT DETECTOR  (background thread)
# ══════════════════════════════════════════════════════════════════════════════
class ObjectDetector:
    def __init__(self):
        print("[INFO] Loading YOLOv8 model …")
        self.model    = YOLO(YOLO_MODEL)
        self._lock    = threading.Lock()
        self._results = []
        self.busy     = False
        print("[OK] YOLO ready")

    def submit(self, frame):
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._run, args=(frame.copy(),),
                         daemon=True).start()

    def _run(self, frame):
        res  = self.model(frame, conf=YOLO_CONFIDENCE, verbose=False)[0]
        dets = [
            (self.model.names[int(b.cls[0])], float(b.conf[0]),
             b.xyxy[0].tolist())
            for b in res.boxes
        ]
        with self._lock:
            self._results = dets
        self.busy = False

    def get(self) -> list:
        with self._lock:
            return list(self._results)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def expand_box(box, pad, w, h):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (max(int(min(xs)) - pad, 0), max(int(min(ys)) - pad, 0),
            min(int(max(xs)) + pad, w), min(int(max(ys)) + pad, h))


def compute_pan(box, frame_width: int) -> float:
    """Map horizontal centre of a text box to a pan value in [-1, 1]."""
    xs     = [p[0] for p in box]
    centre = (min(xs) + max(xs)) / 2.0
    pan    = (centre / frame_width) * 2.0 - 1.0
    return float(np.clip(pan, -1.0, 1.0))


def build_raw_sentence(detections: list) -> str:
    words = []
    for (box, text, conf) in detections:
        clean = re.sub(r'[^a-zA-Z0-9\s]', '',
                       ''.join(c for c in text.strip()
                               if ord(c) < 128 and c.isprintable())).strip()
        if len(clean) >= 2:
            words.append(clean)
    return ' '.join(words)


def draw_label(frame, text: str, x1: int, y1: int,
               color=(0, 200, 80)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, 0.48, 1)
    ly = max(y1 - 6, th + 4)
    cv2.rectangle(frame, (x1, ly - th - 4),
                  (x1 + tw + 4, ly + bl), color, -1)
    cv2.putText(frame, text, (x1 + 2, ly),
                font, 0.48, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(frame, n_text: int, n_obj: int, last: str,
             fps: float, motion: bool, priority: str):
    h, w = frame.shape[:2]
    # Bottom bar
    cv2.rectangle(frame, (0, h - 62), (w, h), (15, 15, 15), -1)
    prio_col = (0, 60, 220) if priority == 'DANGER' else (0, 200, 80)
    cv2.putText(frame,
        f"FPS:{fps:5.1f}   TEXT:{n_text}   OBJ:{n_obj}"
        f"   MOTION:{'YES' if motion else ' NO '}"
        f"   PRIORITY:{priority}",
        (8, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.44, prio_col, 1)
    last_short = (last[:72] + '…') if len(last) > 72 else last
    cv2.putText(frame, f"LAST: {last_short}",
        (8, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1)


def connect_camera(url: str, retries: int = 5, delay: int = 3):
    for i in range(1, retries + 1):
        print(f"[INFO] Connecting … attempt {i}/{retries}")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print(f"[OK] Connected: {url}")
            return cap
        print(f"[WARN] Failed. Retrying in {delay}s …")
        time.sleep(delay)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"[INFO] Stream: {STREAM_URL}\n")

    # Initialise subsystems
    logger    = Logger(DB_PATH)
    nlp       = NLPPipeline()
    tts       = TTSEngine()
    motion_d  = MotionDetector()
    text_det  = TextDetector()
    obj_det   = ObjectDetector()

    cap = connect_camera(STREAM_URL)
    if cap is None:
        print("[ERROR] Could not connect to camera. Exiting.")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # State
    frame_count     = 0          # fps counter (resets every second)
    yolo_counter    = 0          # independent YOLO cadence counter
    fps             = 0.0
    fps_ts          = time.time()
    last_raw        = ""
    last_speak_ts   = 0.0
    last_obj_labels = set()
    text_dets       = []
    obj_dets        = []
    motion_now      = False
    current_priority= "NORMAL"
    show_original   = True       # start in annotated view

    print("[CONTROLS]")
    print("  q — quit          o — toggle text-only / annotated view")
    print("  s — save frame    r — print DB summary\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame lost — reconnecting …")
            cap.release()
            time.sleep(2)
            cap = connect_camera(STREAM_URL, retries=3, delay=2)
            if cap is None:
                break
            continue

        frame = cv2.flip(frame, -1)   # correct inverted ESP32-CAM mount
        frame_h, frame_w = frame.shape[:2]

        # ── FPS ───────────────────────────────────────────────────────────────
        frame_count  += 1
        yolo_counter += 1
        elapsed = time.time() - fps_ts
        if elapsed >= 1.0:
            fps        = frame_count / elapsed
            fps_ts     = time.time()
            frame_count = 0

        # ── Motion-triggered OCR ──────────────────────────────────────────────
        motion_now = motion_d.check(frame)
        if motion_now:
            text_det.submit(frame)

        # ── YOLO every N frames ───────────────────────────────────────────────
        if yolo_counter % YOLO_EVERY == 0:
            obj_det.submit(frame)

        # ── Fetch latest results ──────────────────────────────────────────────
        text_dets = text_det.get_merged()
        obj_dets  = obj_det.get()

        # ── Log new YOLO objects (deduplicated) ───────────────────────────────
        current_labels = {o[0] for o in obj_dets}
        new_objs = [o for o in obj_dets if o[0] not in last_obj_labels]
        if new_objs:
            logger.log_objects(new_objs)
        last_obj_labels = current_labels

        # ── Speak logic ───────────────────────────────────────────────────────
        if text_dets:
            raw = build_raw_sentence(text_dets)
            now = time.time()
            if len(raw) > 3:
                is_new      = raw != last_raw
                is_cooldown = (now - last_speak_ts) >= COOLDOWN

                if is_new or is_cooldown:
                    corrected        = nlp.correct(raw)
                    ents             = nlp.entities(corrected)
                    current_priority = nlp.priority(corrected)
                    speech           = NLPPipeline.enrich(
                                           corrected, ents, obj_dets)

                    # Directional pan — mean over all detected text boxes
                    pans = [compute_pan(box, frame_w)
                            for (box, _, _) in text_dets]
                    pan  = float(np.mean(pans)) if pans else 0.0

                    avg_conf = float(np.mean([c for (_, _, c) in text_dets]))
                    logger.log_text(raw, corrected, ents,
                                    current_priority, avg_conf)
                    tts.speak(speech, pan=pan, priority=current_priority)

                    tag = '⚠  DANGER' if current_priority == 'DANGER' else 'DETECTED'
                    print(f'[{tag}]  pan={pan:+.2f}  "{speech}" ')

                    last_raw      = raw
                    last_speak_ts = now
        else:
            # No text — reset state and stop audio if playing
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                print("[INFO] No text — audio stopped")
            last_raw         = ""
            last_speak_ts    = 0.0
            current_priority = "NORMAL"

        # ── Display ───────────────────────────────────────────────────────────
        display = frame.copy()

        if show_original:
            # ── Annotated view: OCR boxes + YOLO boxes ────────────────────────
            for (box, text, conf) in text_dets:
                col = (0, 40, 230) if nlp.priority(text) == 'DANGER' \
                      else (0, 230, 80)
                pts = np.array([[int(p[0]), int(p[1])] for p in box],
                               dtype=np.int32)
                cv2.polylines(display, [pts], True, col, 2)
                if SHOW_LABEL:
                    x1, y1, _, _ = expand_box(box, 0, frame_w, frame_h)
                    draw_label(display, f"{text} ({conf:.0%})", x1, y1, col)

            for (lbl, conf, bbox) in obj_dets:
                x1, y1, x2, y2 = (int(v) for v in bbox)
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 170, 0), 2)
                draw_label(display, f"{lbl} {conf:.0%}",
                           x1, y1, (255, 170, 0))
            mode_tag = "ANNOTATED"
        else:
            # ── Text-only mask view ───────────────────────────────────────────
            mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            for (box, _, _) in text_dets:
                x1, y1, x2, y2 = expand_box(box, PADDING, frame_w, frame_h)
                mask[y1:y2, x1:x2] = 255
            mask    = cv2.dilate(mask,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                          iterations=2)
            display = cv2.bitwise_and(frame,
                          cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            if SHOW_LABEL:
                for (box, text, conf) in text_dets:
                    x1, y1, _, _ = expand_box(box, PADDING, frame_w, frame_h)
                    draw_label(display, f"{text} ({conf:.0%})", x1, y1)
            mode_tag = "TEXT-ONLY"

        # Top bar
        cv2.rectangle(display, (0, 0), (frame_w, 24), (15, 15, 15), -1)
        cv2.putText(display,
            f"[{mode_tag}]  Q=quit  O=toggle  S=save  R=db",
            (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)

        draw_hud(display, len(text_dets), len(obj_dets),
                 last_raw, fps, motion_now, current_priority)

        cv2.imshow("ESP32-CAM — Assistive Reader v2", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            show_original = not show_original
        elif key == ord('s'):
            fname = f"capture_{int(time.time())}.png"
            cv2.imwrite(fname, display)
            print(f"[SAVED] {fname}")
        elif key == ord('r'):
            t, o = logger.summary()
            print(f"\n[DB SUMMARY]  text_detections={t}  |  object_detections={o}")
            print("[RECENT TEXT DETECTIONS]")
            for row in logger.recent_texts(5):
                print(f"  {row[0]}  [{row[2]}]  {row[1]}")
            print()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    t, o = logger.summary()
    print(f"\n[DONE] Session log → {DB_PATH}")
    print(f"       Text events : {t}")
    print(f"       Object events: {o}")


if __name__ == "__main__":
    main()