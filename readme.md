
# ARIA: Augmented Reality Intelligence Assistant

`ARIA` is a wearable, end-to-end intelligence system designed to enhance human perception through real-time environmental analysis. By integrating computer vision, natural language processing, and spatial audio feedback, ARIA transforms raw visual data into actionable intelligence delivered via a heads-up display (HUD) and voice interface.

---

### Core Architecture
**ARIA** operates on a "Sense-Process-Act" loop, leveraging a sophisticated stack of embedded systems and machine learning models:

- **Visual Perception: Utilizes YOLO** (You Only Look Once) for real-time object detection and spatial awareness.

- **Textual Intelligence: Implements OCR** (Optical Character Recognition) to digitize environmental text, paired with NLP (Natural Language Processing) for context extraction and summarization.

- **Multimodal Interaction**: A seamless bridge between camera input and audio/visual output, providing a hands-free "copilot" experience.

---

### Key Features
👁️ **Real-Time Object Recognition**
Powered by a custom-trained YOLO backbone, ARIA identifies objects in the user's periphery, providing distance estimation and classification labels directly onto the HUD.

📜 **Contextual OCR & Translation**
Instantly read and interpret signs, documents, or digital screens. The NLP engine doesn't just read words; it understands intent, allowing for real-time translation or action-item extraction.

🎧 **Adaptive Audio Feedback**
For situations where the HUD is too intrusive, ARIA uses text-to-speech (TTS) to provide whispered audio cues, ensuring the user remains informed without breaking eye contact with their environment.

🖥️ **Cyberpunk HUD Aesthetic**
The interface is designed for high contrast and low cognitive load, utilizing a stylized "Cyberpunk" aesthetic that prioritizes critical telemetry and threat detection.

## Technical Stack
| Component | Technology |
|------------------|------------|
|Primary Language |Python / C++ |
| Object Detection | YOLOv8 / YOLOv10 |
| OCR Engine | Tesseract / EasyOCR |
|NLP|Transformers (HuggingFace) / OpenAI API|
|Hardware Target|NVIDIA Jetson Nano / Raspberry Pi 5 / OAK-D|
|HUD Rendering|OpenCV / Pygame|

---

### System Workflow
1. Ingestion: High-definition wide-angle camera feed is captured at 30+ FPS.

2. Processing Pipeline: * Frame is passed to the YOLO inference engine for bounding box generation.

    - Selected regions of interest (ROI) are sent to the OCR pipeline.

    - Text strings are processed via NLP for intent mapping.

3. Rendering: The HUD overlay is composited over the live feed.

4. Feedback: Audio alerts are triggered based on priority logic (e.g., "Person detected at 2 o'clock").


###  Installation & Setup
`[!IMPORTANT]`

Ensure all CUDA drivers are correctly mapped if running on NVIDIA hardware to maintain real-time inference speeds.


```
# Clone the repository
git clone https://github.com/username/ARIA.git

# Install dependencies
pip install -r requirements.txt

# Initialize ARIA
python main.py --mode hud --voice-enabled

```
---

###  Future Roadmap
- iometric Integration: Heart rate and stress level monitoring displayed on-screen.

- SLAM Mapping: Simultaneous Localization and Mapping for indoor navigation.

- Gesture Control: Integrating Hand-tracking for HUD menu navigation.

**ARIA** — Redefining the boundary between human perception and digital intelligence.

### Prototype
 ![Image](https://github.com/user-attachments/assets/acbdcb03-e725-48d9-92cd-b6f792021ddd)

 ---
 
![Image](https://github.com/user-attachments/assets/a6dcbb3c-9619-4dd3-8bca-f99bed77e77a)


