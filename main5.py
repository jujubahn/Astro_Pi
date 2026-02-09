""" 
Astro Pi ISS Speed Estimation Program
Captures images and analyzes ground motion to estimate ISS orbital velocity.
"""

from time import time, sleep
import math
import cv2
import numpy as np
from pathlib import Path

# Try to import libraries - use graceful fallbacks for testing
try:
    from picamzero import Camera
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False

try:
    from sense_hat import SenseHat
    SENSEHAT_AVAILABLE = True
except ImportError:
    SENSEHAT_AVAILABLE = False

# =========================================================================
# CONFIGURATION
# =========================================================================
NUM_IMAGES = 42                    # Number of images to capture
CAPTURE_INTERVAL = 8               # Seconds between captures
MAX_RUNTIME = 595                  # Maximum runtime in seconds (~10 min)
IMAGE_FOLDER = "data"              # Folder to store images

# Create image folder
Path(IMAGE_FOLDER).mkdir(exist_ok=True)

# Feature detection parameters (ORB algorithm)
ORB_FEATURES = 2000                # Number of ORB features to detect per image
MIN_MATCHES = 10                   # Minimum matches required to accept a frame pair
MIN_KEYPOINTS = 10                 # Minimum keypoints needed for feature detection

# Outlier rejection parameters
MAD_THRESHOLD = 2.0                # Median Absolute Deviation multiplier for filtering

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def get_ground_scale_m_per_pixel():
    """
    Calculate the scale factor: meters per pixel on the ground.
    
    Uses:
    - ISS altitude: 408 km
    - Camera field of view: 62 degrees
    - Image resolution: varies by camera
    
    Returns:
        float: Meters per pixel
    """
    ISS_ALTITUDE_M = 408000.0
    CAMERA_FOV_DEG = 62.0
    
    # Assume standard image width (picamzero uses 4056 pixels by default)
    IMAGE_WIDTH = 4056
    
    fov_rad = math.radians(CAMERA_FOV_DEG)
    angle_per_pixel = fov_rad / IMAGE_WIDTH
    return ISS_ALTITUDE_M * math.tan(angle_per_pixel)


def find_pixel_shift(image1_path, image2_path):
    """
    Find how many pixels the ground moved between two consecutive images.
    
    Uses ORB (Oriented FAST and Rotated BRIEF) feature detection to:
    1. Find distinctive landmarks in each image
    2. Match landmarks between images
    3. Calculate the median displacement
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
    
    Returns:
        float: Median pixel displacement, or None if matching failed
    """
    # Read images in grayscale
    img1 = cv2.imread(str(image1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(image2_path), cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return None
    
    # Initialize ORB detector using configured feature count
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Check if enough features were found
    if des1 is None or des2 is None or len(kp1) < MIN_KEYPOINTS or len(kp2) < MIN_KEYPOINTS:
        return None

    # Match features between images using Brute Force matcher with Hamming distance
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf_matcher.match(des1, des2)

    # Need at least minimum matches to accept result
    if len(matches) < MIN_MATCHES:
        return None

    # Calculate displacements for each matched feature (use all matches)
    displacements = []
    for match in matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        displacement = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        displacements.append(displacement)

    # Return median displacement (robust to outliers)
    return float(np.median(displacements))


def calculate_robust_median(values):
    """
    Calculate median with MAD-based outlier rejection, return mean of filtered values.
    
    Args:
        values: List of numbers
    
    Returns:
        float: Mean of values within 2*MAD of median
    """
    if not values or len(values) == 0:
        return None
    
    arr = np.array(values)
    median = np.median(arr)
    
    # Calculate MAD (Median Absolute Deviation)
    absolute_deviations = np.abs(arr - median)
    mad = np.median(absolute_deviations)

    # If all values are the same
    if mad == 0:
        return float(median)

    # Keep values within MAD_THRESHOLD * MAD of median
    filtered = arr[absolute_deviations <= MAD_THRESHOLD * mad]

    return float(np.mean(filtered))


# =========================================================================
# MAIN PROGRAM
# =========================================================================

def main():
    """
    Main program: Capture images and estimate ISS speed.
    """
    print("=" * 60)
    print("Astro Pi ISS Speed Estimation")
    print("=" * 60)
    
    # Initialize camera
    if CAMERA_AVAILABLE:
        camera = Camera()
        print("[OK] Camera initialized (picamzero)")
    else:
        print("[ERROR] Camera library not available")
        return
    
    # Initialize SenseHAT (optional, for display)
    sense = None
    if SENSEHAT_AVAILABLE:
        try:
            sense = SenseHat()
            sense.clear()
            print("[OK] SenseHAT initialized")
        except:
            print("[ERROR] SenseHAT initialization failed")
    
    # Phase 1: Capture image sequence
    print(f"\nPhase 1: Capturing {NUM_IMAGES} images...")
    print(f"Interval: {CAPTURE_INTERVAL} seconds")
    
    image_files = []
    image_times = []
    # Compute speeds on-the-fly; keep only previous image to save storage
    speed_estimates = []
    prev_image = None
    prev_time = None
    # Precompute ground scale for on-the-fly calculation
    scale_m_per_pixel = get_ground_scale_m_per_pixel()
    start_time = time()
    
    try:
        for i in range(NUM_IMAGES):
            elapsed = time() - start_time
            if elapsed >= MAX_RUNTIME:
                print(f"Time limit reached ({elapsed:.1f}s)")
                break
            
            # Generate filename
            image_name = f"image_{i:02d}.jpg"
            image_path = image_name
            
            # Capture image
            capture_time = time()
            camera.take_photo(image_path)
            
            image_files.append(image_path)
            image_times.append(capture_time)

            print(f"  [{i+1:2d}/{NUM_IMAGES}] Captured {image_name}")

            # If a previous image exists, process the pair now
            if prev_image is not None:
                dt = capture_time - prev_time
                if dt > 0:
                    pixel_shift = find_pixel_shift(prev_image, image_path)
                    if pixel_shift is not None:
                        ground_distance = pixel_shift * scale_m_per_pixel
                        speed_ms = ground_distance / dt
                        speed_kms = speed_ms / 1000.0
                        speed_estimates.append(speed_kms)
                        print(f"    Pair proc: {prev_image} -> {image_name}: {pixel_shift:.1f}px -> {speed_kms:.4f} km/s")
                    else:
                        print(f"    Pair proc: {prev_image} -> {image_name}: no match")
                else:
                    print(f"    Pair proc: invalid dt {dt}")

                # Remove previous image to avoid storing all images
                try:
                    Path(prev_image).unlink()
                except Exception:
                    pass

            prev_image = image_path
            prev_time = capture_time

            # Wait before next capture (except for last image)
            if i < NUM_IMAGES - 1:
                sleep(CAPTURE_INTERVAL)
        
        print(f"[OK] Capture complete: {len(image_files)} images")
    
    except Exception as e:
        print(f"[ERROR] Capture error: {e}")
        return
    
    # Phase 2: Process image pairs
    print(f"\nPhase 2: Analyzing image motion...")
    
    if len(image_files) < 2:
        print(f"[ERROR] Need at least 2 images, got {len(image_files)}")
        return
    
    # If we already computed speeds during capture, use those estimates
    if len(speed_estimates) > 0:
        print(f"Using {len(speed_estimates)} on-the-fly speed estimates collected during capture")
    else:
        # Get ground scale
        scale_m_per_pixel = get_ground_scale_m_per_pixel()
        print(f"Ground scale: {scale_m_per_pixel:.2f} m/pixel")

        # Analyze consecutive image pairs (fallback)
        for i in range(len(image_files) - 1):
            img1_path = image_files[i]
            img2_path = image_files[i + 1]
            t1 = image_times[i]
            t2 = image_times[i + 1]

            # Time between images
            dt = t2 - t1

            if dt <= 0:
                print(f"  Pair {i}: Invalid time delta")
                continue

            # Find pixel shift
            pixel_shift = find_pixel_shift(img1_path, img2_path)

            if pixel_shift is None:
                print(f"  Pair {i}: No features matched")
                continue

            # Convert to speed
            ground_distance = pixel_shift * scale_m_per_pixel
            speed_ms = ground_distance / dt
            speed_kms = speed_ms / 1000.0

            speed_estimates.append(speed_kms)

            print(f"  Pair {i}: {pixel_shift:.1f}px -> {speed_kms:.4f} km/s")
    
    # Phase 3: Calculate final estimate
    print(f"\nPhase 3: Final calculation...")
    
    if len(speed_estimates) == 0:
        print("[ERROR] No valid speed estimates calculated")
        final_speed = 0.0
    else:
        final_speed = calculate_robust_median(speed_estimates)
        print(f"Valid estimates: {len(speed_estimates)}")
        print(f"Speed estimates range: {min(speed_estimates):.4f} - {max(speed_estimates):.4f} km/s")
        print(f"Final estimate (median): {final_speed:.4f} km/s")
    
    # Phase 4: Write result
    print(f"\nPhase 4: Writing result...")
    
    # Format to 5 significant figures (as per requirements)
    # Use .4f format for values like 7.1234
    result_string = "{:.4f}".format(final_speed)
    
    try:
        with open("result.txt", "w") as f:
            f.write(result_string)
        print(f"[OK] Result written to result.txt: {result_string}")
    except Exception as e:
        print(f"[ERROR] Error writing result: {e}")
        return
    
    # Optional: Display on SenseHAT
    if sense:
        try:
            sense.show_message(f"Speed: {final_speed:.2f} km/s", scroll_speed=0.05)
        except:
            pass
    
    print("\n" + "=" * 60)
    print("Program complete!")
    print("=" * 60)


# =========================================================================
# PROGRAM ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Write default result on error
        try:
            with open("result.txt", "w") as f:
                f.write("0.0000")
        except:
            pass
