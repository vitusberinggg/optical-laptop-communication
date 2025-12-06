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
        "light_level": 1.0,
        "effects": ["temporal_instability"]
    },
    "none": {
        "severity": 0.0,
        "light_level": 1.0,
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

            # temporal instability is special: because it doesn't take severity or light_level
            elif effect.__name__ == "temporal_instability":
                frame = effect(frame)

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
        Webcam-realistic noise (GPU).
        """
        return ocl.run_noise(frame, severity, light_level)





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
        """
        Subtle soft-focus blur for webcam realism.
        severity: 0.0 = none, 1.0 = max subtle blur
        """
        # Map severity to sigma radius (~0.5–2.0 pixels)
        radius = 0.5 + severity * 2.0
        return ocl.run_blur(frame, radius=radius)




    # --- Frame-Rate Instability ---

    _last_frame = None
    _frozen_frame = None
    _freeze_timer = 0

    @staticmethod
    def temporal_instability(frame,
                             drop_prob=0.05,
                             freeze_prob=0.03,
                             freeze_duration=3):
        """
        Returns exactly ONE frame every call.
        Simulates:
        - dropped updates (frame repeats)
        - short freezes
        - jitter in motion

        frame: HxWxC ndarray
        """

        # Initialize state on first frame
        if Effects._last_frame is None:
            Effects._last_frame = frame.copy()
            return frame

        # If we are currently frozen: keep outputting frozen frame
        if Effects._freeze_timer > 0:
            Effects._freeze_timer -= 1
            return Effects._frozen_frame

        r = random.random()

        # Start a freeze
        if r < freeze_prob:
            Effects._freeze_timer = freeze_duration
            Effects._frozen_frame = Effects._last_frame.copy()
            return Effects._frozen_frame

        # Drop update (frame repeats)
        if r < freeze_prob + drop_prob:
            return Effects._last_frame

        # Otherwise: update normally
        Effects._last_frame = frame.copy()
        return frame


    # --- Chromatic Aberration (Channel Shift) ---

    @staticmethod
    def chromatic_aberration(frame, severity):
        return ocl.run_chromatic_aberration(frame, severity)




# --- Test the distorter ---

if __name__ == "__main__":
    
    frame = cv2.imread(r"C:\Users\ejadmax\code\optical-laptop-communication\webcam_simulation\test_bitgrid.png")
    distorter = FrameDistorter(preset="custom")
    distorted_frame = distorter.apply(frame)
    cv2.imwrite("distorted.jpg", distorted_frame)