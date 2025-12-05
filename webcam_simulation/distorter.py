# --- Imports ---

import cv2
import numpy as np
import random

# importing OpenCL(class) from ocl 
# (used in jitter_color and white_balance_shifter)
from ocl import OpenCL
ocl = OpenCL()

# --- Presets of the amount effects and severity ---

PRESETS = {
    "custom": {
        "severity": 1.0,
        "light_level": 1.0,  # normal light
        "effects": ["temporal_instability"]
    },
    "none": {
        "severity": 0.0,
        "light_level": 1.0,  # normal light
        "effects": []
    },
        "light": {
        "severity": 0.2,
        "light_level": 1.2,
        "effects": ["noise", "jitter_color", "white_balance_shift"]
    },
        "medium": {
        "severity": 0.5,
        "light_level": 1.0,
        "effects": ["noise", "jitter_color", "white_balance_shift", 
                    "rolling_shutter", "blur", "jpeg_compress"]
    },
        "heavy": {
        "severity": 0.8,
        "light_level": 0.7,
        "effects": ["noise", "jitter_color", "white_balance_shift", 
                    "rolling_shutter", "warp", "chromatic_aberration", 
                    "blur", "jpeg_compress", "temporal_instability"]
    },
        "webcam_realistic": {
        "severity": 0.5,
        "light_level": 0.8,
        "effects": ["noise","jitter_color","white_balance_shift",
                    "rolling_shutter","blur","chromatic_aberration",
                    "jpeg_compress", "temporal_instability"]
    }
}


# --- Frame distorter ---

class FrameDistorter:
    def __init__(self, preset="webcam_realistic"):
        config = PRESETS[preset]
        self.severity = config["severity"]
        self.light_level = config.get("light_level", 1.0)
        self.effects = []
        self.effects_obj = Effects(self.light_level)

        # Define effect priority (lower number = applied first)
        EFFECT_PRIORITY = {
            "noise": 0,                 # Sensor/ISO noise happens first
            "jitter_color": 1,          # Brightness/contrast/tint readout adjustments
            "white_balance_shift": 2,   # Color drift after initial readout
            "rolling_shutter": 3,       # Per-row readout offset occurs during capture
            "warp": 4,                  # Geometric distortions applied after row offsets
            "chromatic_aberration": 5,  # Optical color fringing before defocus blur
            "blur": 6,                  # Lens defocus / softening happens after CA
            "jpeg_compress": 7,         # Compression applied to the final frame
            "temporal_instability": 8,  # Frame-level drops/duplicates happen last
        }

        # Sort the preset effects automatically by priority
        config_effects = sorted(config["effects"], key=lambda e: EFFECT_PRIORITY.get(e, 99))

        # Automatically map effect names to functions
        for name in config_effects:
            if hasattr(self.effects_obj, name):
                self.effects.append(getattr(self.effects_obj, name))
            else:
                raise ValueError(f"Effect '{name}' not found.")
            
        print(f"[FrameDistorter] Effects applied in order: {[fn.__name__ for fn in self.effects]}")

    # Apply all effects
    def apply(self, frame):
        # Adjust global brightness first
        frame = np.clip(frame.astype(np.float32) * self.light_level, 0, 255).astype(np.uint8)

        for effect in self.effects:
            # noise & jitter_color require light_level
            if effect.__name__ in ["noise", "jitter_color"]:
                frame = effect(frame, self.severity, self.light_level)

            # white balance drift needs only severity
            elif effect.__name__ == "white_balance_shift":
                frame = effect(frame, self.severity)

            else:
                frame = effect(frame, self.severity)
        return frame



# --- Effects ---

class Effects:

    def __init__(self, light_level):
        # White balance gains (continuous drift)
        self.light_level = light_level
        self.wb_r = 1.0
        self.wb_g = 1.0
        self.wb_b = 1.0

    # --- Sensor noise ---

    @staticmethod
    def noise(frame, severity, light_level):
        """
        Adds realistic sensor noise (grain-like) instead of sparse pixel spikes.
        Looks like ISO noise and reacts to brightness.
        """

        # Ensure float32 for calculation
        img = frame.astype(np.float32)

        # Base noise amount (adjust these to taste)
        base_sigma = 3     # minimum grain
        max_sigma  = 35    # maximum grain

        # Severity controls sigma linearly
        sigma = base_sigma + (max_sigma - base_sigma) * severity

        # Light level increases noise when dark:
        # dark → more noise, bright → less
        light_factor = 1.0 / max(light_level, 0.15)
        sigma *= light_factor

        # Generate Gaussian noise
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)

        # Add noise
        noised = img + noise

        # Clip to valid range
        noised = np.clip(noised, 0, 255).astype(np.uint8)

        return noised



    # --- Color jitter (brightness/contrast) ---

    @staticmethod
    def jitter_color(frame, severity, light_level):
        # 1. brightness
        max_brightness = 40 * severity
        brightness = np.random.uniform(-max_brightness, max_brightness)
        brightness *= 1.0 / max(light_level, 0.1)

        # 2. contrast
        contrast = np.random.uniform(1 - 0.4 * severity, 1 + 0.4 * severity)

        # 3. tint
        max_tint = 0.08 * severity
        gains = (
            1 + np.random.uniform(-max_tint, max_tint),  # R
            1 + np.random.uniform(-max_tint, max_tint),  # G
            1 + np.random.uniform(-max_tint, max_tint)   # B
        )

        # GPU-accelerated processing
        return ocl.run_jitter(frame, brightness, contrast, gains)
    
    def rolling_shutter(self, frame, severity, amplitude=2.0):
        # generate slowly-varying row offsets; severity controls amplitude
        h, w, _ = frame.shape
        # base wobble scaled by severity and darkness maybe
        max_offset = amplitude * severity
        # generate smooth per-row offsets via a low-frequency sine + small noise
        freq = 2.0 / max(1, h/100.0)
        rows = np.arange(h).astype(np.float32)
        phase = random.uniform(0, 2*np.pi)
        sine = np.sin(rows * freq + phase) * max_offset
        noise = (np.random.rand(h).astype(np.float32) - 0.5) * (max_offset * 0.2)
        row_offset = sine + noise
        return ocl.run_rolling_shutter(frame, row_offset)



    # --- Warp / elastic distortion ---

    def warp(self, frame, severity):
        # create low-res displacement map like before but then run on GPU
        h, w, _ = frame.shape
        max_strength = 5.0
        strength = severity * max_strength
        # choose scale (bigger = smoother)
        scale = 32 if max(w,h) >= 1280 else 24
        dx_small = (np.random.rand(h // scale + 1, w // scale + 1) - 0.5) * strength
        dy_small = (np.random.rand(h // scale + 1, w // scale + 1) - 0.5) * strength
        dx = cv2.resize(dx_small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        dy = cv2.resize(dy_small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)

        # build map_x, map_y (pixel coords)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        return ocl.run_warp(frame, map_x, map_y)


    # --- jpeg compression blocking ---

    def jpeg_compress(self, frame, severity):
        # map severity to quality: severity 0->100, 1->10
        quality = int(100 - severity * 90)
        block_size = 8 if frame.shape[1] <= 1280 else 16
        return ocl.run_jpeg_approx(frame, block_size=block_size, quality=quality)



    # --- White balance shift (real webcam drift) ---

    def white_balance_shift(self, frame, severity):
        # How strong the drift is depending on severity and light level
        # Much stronger when light is low
        stability = 1.2 - self.light_level       # 0.2 at bright light → 1.2 at darkness
        strength = severity * stability * 0.02   # drift per frame (2% max per frame in darkness)

        # Add tiny drift to each channel (continuous)
        self.wb_r += random.uniform(-strength, strength)
        self.wb_g += random.uniform(-strength, strength)
        self.wb_b += random.uniform(-strength, strength)

        # Soft clamp so it doesn’t go crazy
        self.wb_r = max(0.7, min(1.3, self.wb_r))
        self.wb_g = max(0.7, min(1.3, self.wb_g))
        self.wb_b = max(0.7, min(1.3, self.wb_b))

        gains = (self.wb_r, self.wb_g, self.wb_b)
        return ocl.run_white_balance(frame, gains)



    # --- Soft-Focus Blur (Gaussian Blur) ---

    @staticmethod
    def blur(frame, severity=1.0):
        radius = 0.8 + severity * 2.0  # scale to 0.8–2.8

        if radius <= 0:
            return frame

        # Convert to UMat for OpenCL acceleration
        frame_gpu = cv2.UMat(frame)

        # Kernel size must be odd
        k = max(3, int(radius * 4) | 1)

        # Apply Gaussian blur on GPU
        blurred_gpu = cv2.GaussianBlur(frame_gpu, (k, k), sigmaX=radius)

        # Convert back to normal NumPy array
        blurred = blurred_gpu.get()
        return blurred


    # --- Frame-Rate Instability (simulate dropped or repeated frames) ---

    @staticmethod
    def temporal_instability(frames, drop_prob=0.1, duplicate_prob=0.1):
        """
        frames: list of NumPy arrays (HxWxC)
        returns: list of NumPy arrays (modified)
        """
        new_frames = []
        for f in frames:
            r = random.random()

            if r < drop_prob:
                # drop frame
                continue

            new_frames.append(f)

            # duplicate frame sometimes
            if random.random() < duplicate_prob:
                new_frames.append(f.copy())

        return new_frames


    # --- Chromatic Aberration (Channel Shift) ---

    @staticmethod
    def chromatic_aberration(frame, severity):
        """
        severity: 0.0 = none
                1.0 = strong color separation
        """
        H, W, C = frame.shape
        assert C == 3

        frame_f = frame.astype(np.float32)

        # shift range (in pixels)
        max_shift = 2.0   # realistic webcam aberration is tiny (1–2 pixels)

        dx_r = severity * max_shift
        dy_r = 0.0

        dx_b = -severity * max_shift
        dy_b = 0.0

        def shift_channel(channel, dx, dy):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            channel_gpu = cv2.UMat(channel)
            shifted = cv2.warpAffine(channel_gpu, M, (W, H),
                                    borderMode=cv2.BORDER_REPLICATE)
            return shifted.get()

        R = shift_channel(frame_f[:, :, 2], dx_r, dy_r)
        G = frame_f[:, :, 1]
        B = shift_channel(frame_f[:, :, 0], dx_b, dy_b)

        result = cv2.merge([B, G, R])
        return np.clip(result, 0, 255).astype(np.uint8)



# --- Test the distorter ---

if __name__ == "__main__":
    
    frame = cv2.imread(r"C:\Users\ejadmax\code\optical-laptop-communication\webcam_simulation\test_bitgrid.png")
    distorter = FrameDistorter(preset="custom")
    distorted_frame = distorter.apply(frame)
    cv2.imwrite("distorted.jpg", distorted_frame)