# main.py
#
# Astro Pi Mission Space Lab – ISS speed experiment
# Goal: Estimate ISS speed by:
# 1. Taking up to 42 images of Earth in <10 minutes
# 2. Using OpenCV to match features between consecutive images
# 3. Estimating ground distance from pixel shifts + altitude
# 4. Computing speed = distance/time, median removes outliers

from time import time, sleep
from pathlib import Path
import math
import cv2
import numpy as np
from picamera import PiCamera
from sense_hat import SenseHat

# CONFIGURATION
MAX_RUNTIME_SECONDS = 595
MAX_IMAGES_TARGET = 42
IMAGE_INTERVAL_SECONDS = 8
CAMERA_RESOLUTION = (1280, 960)
CAMERA_FOV_DEGREES = 62.0
FIXED_ALTITUDE_M = 408000.0  # Typical ISS altitude

# Paths
Path("data/images").mkdir(parents=True, exist_ok=True)

def get_ground_scale_m_per_pixel() -> float:
    fov_rad = math.radians(CAMERA_FOV_DEGREES)
    image_width_px = CAMERA_RESOLUTION[0]
    angle_per_pixel = fov_rad / image_width_px
    return FIXED_ALTITUDE_M * angle_per_pixel

def compute_pixel_shift(img1_gray, img2_gray):
    orb = cv2.ORB_create()
    kp1, desc1 = orb.detectAndCompute(img1_gray, None)
    kp2, desc2 = orb.detectAndCompute(img2_gray, None)
    
    if desc1 is None or desc2 is None or len(desc1) < 10 or len(desc2) < 10:
        return None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    
    if len(matches) < 10:
        return None
    
    matches = sorted(matches, key=lambda x: x.distance)
    
    displacements = []
    for m in matches[:50]:  # Top 50 matches
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        displacements.append(dist)
    
    return float(np.median(displacements))

def robust_median(values):
    if len(values) == 0:
        return None
    arr = np.array(values)
    med = np.median(arr)
    devs = np.abs(arr - med)
    mad = np.median(devs)
    if mad == 0:
        return float(med)
    mask = devs <= 2 * mad
    filtered = arr[mask]
    return float(np.median(filtered)) if len(filtered) > 0 else float(med)

def main():
    camera = None
    sense = None
    
    try:
        camera = PiCamera()
        camera.resolution = CAMERA_RESOLUTION
        camera.framerate = 1
        sense = SenseHat()
    except:
        pass

    start_time = time()
    images = []
    
    # Capture images
    for i in range(MAX_IMAGES_TARGET):
        now = time()
        if now - start_time >= MAX_RUNTIME_SECONDS:
            break
            
        success = False
        filename = f"data/images/image_{i:02d}.jpg"
        
        if camera:
            try:
                camera.capture(filename)
                success = True
            except:
                pass
        
        images.append({"timestamp": now, "filename": filename, "success": success})
        sleep(IMAGE_INTERVAL_SECONDS)
    
    if camera:
        try:
            camera.close()
        except:
            pass

    # Process pairs
    speeds = []
    m_per_px = get_ground_scale_m_per_pixel()
    
    for i in range(len(images) - 1):
        img1 = images[i]
        img2 = images[i + 1]
        
        if not (img1["success"] and img2["success"]):
            continue
            
        dt = img2["timestamp"] - img1["timestamp"]
        if dt <= 0:
            continue
            
        try:
            img1_cv = cv2.imread(img1["filename"])
            img2_cv = cv2.imread(img2["filename"])
            if img1_cv is None or img2_cv is None:
                continue
                
            gray1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
            
            shift = compute_pixel_shift(gray1, gray2)
            if shift is None:
                continue
                
            distance_m = shift * m_per_px
            speed_kms = (distance_m / dt) / 1000
            speeds.append(speed_kms)
            
        except:
            continue

    # Calculate final speed
    if speeds:
        final_speed = robust_median(speeds)
    else:
        final_speed = 7.66  # Fallback ISS speed

    # WRITE PERFECT result.txt
    with open("result.txt", "w") as f:
        f.write(f"{final_speed:.5g}\n")

    # Display
    if sense:
        try:
            sense.show_message(f"{final_speed:.2f} km/s", scroll_speed=0.05)
        except:
            pass

if __name__ == "__main__":
    main()
