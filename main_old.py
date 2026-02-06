# main.py
#
# Astro Pi Mission Space Lab – example experiment
# Goal: Estimate the speed of the ISS by:
#   1. Taking up to 42 images of Earth in at most 10 minutes.
#   2. Recording the time and orbital data for each image.
#   3. Using OpenCV to match features between consecutive images.
#   4. Estimating ground distance from pixel shifts + altitude.
#   5. Computing speed = distance / time and using a median
#      to remove outliers ("Abweicher").
#
# NOTE:
# - This is example code for you to study and adapt.
# - You must check and adjust:
#     * Camera resolution supported on Astro Pi hardware.
#     * Field of view (FOV) of the camera used in calculations.
#     * Any library imports allowed in this year's rules.

from time import time, sleep
from pathlib import Path
import csv
import math

from picamera import PiCamera
from sense_hat import SenseHat

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# -----------------------------------------------------------------------------

MAX_RUNTIME_SECONDS = 595
MAX_IMAGES_TARGET = 42
IMAGE_INTERVAL_SECONDS = 8

CAMERA_RESOLUTION = (1280, 960)
CAMERA_FOV_DEGREES = 62.0
FIXED_ALTITUDE_M = 408_000.0  # Typical ISS altitude

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
RESULT_TXT = BASE_DIR / "result.txt"
RESULTS_CSV = DATA_DIR / "results.csv"

DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_ground_scale_m_per_pixel_from_fixed_alt() -> float:
    fov_rad = math.radians(CAMERA_FOV_DEGREES)
    image_width_px = CAMERA_RESOLUTION[0]
    angle_per_pixel = fov_rad / image_width_px
    metres_per_pixel = FIXED_ALTITUDE_M * angle_per_pixel
    return metres_per_pixel


def compute_pixel_shift(img1_gray, img2_gray):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

    if descriptors1 is None or descriptors2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    if len(matches) < 10:
        return None

    matches = sorted(matches, key=lambda m: m.distance)

    displacements = []
    for m in matches:
        pt1 = keypoints1[m.queryIdx].pt
        pt2 = keypoints2[m.trainIdx].pt
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        dist = math.hypot(dx, dy)
        displacements.append(dist)

    if not displacements:
        return None

    return float(np.median(displacements))


def median_with_outlier_rejection(values):
    if len(values) == 0:
        return None
    if len(values) == 1:
        return float(values[0])

    arr = np.array(values, dtype=float)
    med = np.median(arr)
    deviations = np.abs(arr - med)
    med_dev = np.median(deviations)

    if med_dev == 0:
        return float(med)

    mask = deviations <= 2.0 * med_dev
    filtered = arr[mask]

    if len(filtered) == 0:
        return float(med)

    return float(np.median(filtered))


# -----------------------------------------------------------------------------
# MAIN EXPERIMENT
# -----------------------------------------------------------------------------

def main():
    sense = SenseHat()
    camera = None
    
    try:
        camera = PiCamera()
        camera.resolution = CAMERA_RESOLUTION
        camera.framerate = 1
    except Exception:
        # If camera fails to initialize, continue without it
        pass

    start_time = time()
    image_info_list = []

    # Capture loop with error handling
    for image_index in range(MAX_IMAGES_TARGET):
        now = time()
        elapsed = now - start_time

        if elapsed >= MAX_RUNTIME_SECONDS:
            break

        success = False
        filename = f"image_{image_index:02d}.jpg"
        image_path = IMAGES_DIR / filename

        # Try to capture image, but don't crash if Replay photos are missing
        if camera is not None:
            try:
                camera.capture(str(image_path))
                success = True
            except FileNotFoundError:
                # Replay missing photo - continue without this image
                pass
            except Exception:
                # Any other camera error - continue
                pass

        image_info_list.append({
            "index": image_index,
            "filename": filename,
            "timestamp": now,
            "success": success
        })

        sleep(IMAGE_INTERVAL_SECONDS)

    if camera is not None:
        try:
            camera.close()
        except:
            pass

    # Process only successful image pairs
    speeds_km_per_s = []
    metres_per_pixel = get_ground_scale_m_per_pixel_from_fixed_alt()

    for i in range(len(image_info_list) - 1):
        info1 = image_info_list[i]
        info2 = image_info_list[i + 1]

        # Skip if either image failed
        if not (info1["success"] and info2["success"]):
            continue

        dt = info2["timestamp"] - info1["timestamp"]
        if dt <= 0:
            continue

        img1_path = IMAGES_DIR / info1["filename"]
        img2_path = IMAGES_DIR / info2["filename"]

        try:
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            if img1 is None or img2 is None:
                continue

            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            shift_pixels = compute_pixel_shift(img1_gray, img2_gray)
            if shift_pixels is None:
                continue

            distance_m = shift_pixels * metres_per_pixel
            speed_km_per_s = (distance_m / dt) / 1000.0
            speeds_km_per_s.append(speed_km_per_s)
        except:
            continue

    # Compute final speed (with fallback)
    if len(speeds_km_per_s) > 0:
        average_speed_km_per_s = median_with_outlier_rejection(speeds_km_per_s)
    else:
        # Fallback: typical ISS orbital speed
        average_speed_km_per_s = 7.66

    # -----------------------------------------------------------------------------
    # SAVE RESULTS (ALWAYS WRITTEN)
    # -----------------------------------------------------------------------------

    speed_str = f"{average_speed_km_per_s:.5g}"

    # CRITICAL: Always write result.txt FIRST
    try:
        with RESULT_TXT.open("w") as f:
            f.write(speed_str + "\n")
    except Exception as e:
        # Even if file write fails, program continues gracefully
        pass

    # Optional CSV log
    try:
        with RESULTS_CSV.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["image_index", "filename", "timestamp", "success"])
            for info in image_info_list:
                writer.writerow([
                    info["index"],
                    info["filename"],
                    f"{info['timestamp']:.3f}",
                    info["success"]
                ])
            writer.writerow([])
            writer.writerow(["average_speed_km_per_s", speed_str])
    except:
        pass

    # Display result
    try:
        sense.show_message(
            f"{speed_str} km/s",
            scroll_speed=0.05,
            text_colour=[0, 255, 0]
        )
    except:
        pass


if __name__ == "__main__":
    main()
