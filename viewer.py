import serial
import cv2
import numpy as np
import base64

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'  # Replace with your ESP32-CAM's port
BAUD_RATE = 460800

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.setDTR(False) 
        ser.setRTS(False)
        print(f"Connected to {SERIAL_PORT}. Waiting for strict chunked frames...")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    frame_data = ""
    receiving_frame = False

    while True:
        try:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            
            if line == "FRAME_START":
                frame_data = "" 
                receiving_frame = True
                
            elif line == "FRAME_END":
                receiving_frame = False
                if len(frame_data) > 0:
                    try:
                        jpg_data = base64.b64decode(frame_data)
                        np_arr = np.frombuffer(jpg_data, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if img is not None:
                            img_scaled = cv2.resize(img, (640, 480))
                            cv2.imshow('Base64 Strict Stream', img_scaled)
                    except Exception as e:
                        pass # Ignore frames that still somehow break

            # CRITICAL: Only accept lines that have our strict "C:" stamp
            elif receiving_frame and line.startswith("C:"):
                # Slice off the "C:" and keep the pure image data
                frame_data += line[2:]
                
            elif receiving_frame and len(line) > 0:
                # We caught a sneaky background log trying to break our image!
                print(f"Intercepted and blocked stray log: {line}")

        except Exception as e:
            pass
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ser.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()