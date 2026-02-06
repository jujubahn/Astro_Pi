# main.py
# Astro Pi Mission Space Lab – ISS speed experiment (REPLAY & FLIGHT SAFE)
# Works in Online Tool, Thonny, local Python - NO @replay dependency

from time import time, sleep
from pathlib import Path
import math
import cv2
import numpy as np

# Conditional imports for compatibility
try:
    from picamera import PiCamera
    PICAMERA_AVAILABLE = True
except ImportError:
    print("picamera not available - using dummy camera")
    PICAMERA_AVAILABLE = False

try:
    from sense_hat import SenseHat
    SENSEHAT_AVAILABLE = True
except ImportError:
    print("SenseHAT not available - using dummy")
    SENSEHAT_AVAILABLE = False

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MAX_IMAGES = 30  # Reduced for faster testing
CAPTURE_INTERVAL = 8.0  # seconds
CAMERA_RESOLUTION = (1024, 768)  # Smaller for speed
CAMERA_FOV_DEG = 62.2
ISS_ALTITUDE_M = 408000.0

# Create data directory
Path("data/images").mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# DUMMY CAMERA/SENSEHAT FOR REPLAY TOOL
# -----------------------------------------------------------------------------
class DummyCamera:
    def __init__(self):
        self.resolution = (1024, 768)
        self.framerate = 0.5
    
    def capture(self, filename):
        # Create dummy image with gradient for testing
        h, w = self.resolution
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            img[i, :] = [0, int(255*i/h), 255]
        cv2.imwrite(filename, img)
        print(f"Dummy captured: {filename}")

class DummySenseHat:
    def show_message(self, text, scroll_speed=0.05, **kwargs):
        print(f"SenseHAT: {text}")
    
    def clear(self):
        print("SenseHAT cleared")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def ground_scale_m_per_pixel():
    fov_rad = math.radians(CAMERA_FOV_DEG)
    angle_per_px = fov_rad / CAMERA_RESOLUTION[0]
    return ISS_ALTITUDE_M * math.tan(angle_per_px)

def pixel_displacement(img1_path, img2_path):
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return None
    
    # Downscale for matching speed
    scale = 0.5
    h1, w1 = img1.shape
    img1_small = cv2.resize(img1, (int(w1*scale), int(h1*scale)))
    img2_small = cv2.resize(img2, (int(w1*scale), int(h1*scale)))
    
    orb = cv2.ORB_create(nfeatures=800)
    kp1, des1 = orb.detectAndCompute(img1_small, None)
    kp2, des2 = orb.detectAndCompute(img2_small, None)
    
    if des1 is None or des2 is None or len(kp1) < 8:
        return None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if len(matches) < 10:
        return None
    
    # Calculate displacements
    displacements = []
    for match in matches[:80]:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        displacements.append(dist / scale)  # Scale back up
    
    return float(np.median(displacements))

def robust_median(values):
    if not values:
        return 0.0
    arr = np.array(values)
    return float(np.median(arr))

# -----------------------------------------------------------------------------
# MAIN PROGRAM (NO @replay DEPENDENCY)
# -----------------------------------------------------------------------------
def main():
    print("🚀 Astro Pi ISS Speed Experiment Starting...")
    
    # Initialize camera
    if PICAMERA_AVAILABLE:
        camera = PiCamera()
        camera.resolution = CAMERA_RESOLUTION
        camera.framerate = 0.5
        sleep(2)  # Warmup
    else:
        camera = DummyCamera()
    
    # Initialize SenseHAT
    if SENSEHAT_AVAILABLE:
        sense = SenseHat()
        sense.clear()
    else:
        sense = DummySenseHat()
    
    # Capture images
    images = []
    print(f"Capturing {MAX_IMAGES} images...")
    
    for i in range(MAX_IMAGES):
        timestamp = time()
        filename = f"data/images/img_{i:02d}.jpg"
        
        camera.capture(filename)
        images.append((timestamp, filename))
        
        print(f"✓ Image {i+1}/{MAX_IMAGES}")
        
        # Brief pause + main interval
        sleep(0.2)
        if i < MAX_IMAGES - 1:
            sleep(CAPTURE_INTERVAL)
    
    if PICAMERA_AVAILABLE:
        camera.close()
    
    # Analyze motion between frames
    print("📊 Computing ground speed...")
    scale_m_per_px = ground_scale_m_per_pixel()
    speeds = []
    
    for i in range(len(images) - 1):
        t1, f1 = images[i]
        t2, f2 = images[i + 1]
        
        dt = t2 - t1
        if dt < 2.0:
            continue
        
        shift = pixel_displacement(f1, f2)
        if shift is None:
            continue
        
        distance_m = shift * scale_m_per_px
        speed_kmh = (distance_m / dt) / 1000.0
        
        # Ground track sanity check (ISS ground speed ~5-25km/h)
        if 0.5 < speed_kmh < 35:
            speeds.append(speed_kmh)
            print(f"Pair {i}: {shift:.1f}px shift → {speed_kmh:.1f} km/h")
    
    # Final result
    final_speed = robust_median(speeds)
    print(f"🎯 FINAL RESULT: {final_speed:.2f} km/h")
    
    # Write flight result file
    with open("result.txt", "w") as f:
        f.write(f"{final_speed:.6f}\n")
    
    # Display result
    if sense:
        sense.show_message(f"ISS:{final_speed:.1f}kmh", scroll_speed=0.03)

# -----------------------------------------------------------------------------
# SAFE EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        with open("result.txt", "w") as f:
            f.write("0.000000\n")
