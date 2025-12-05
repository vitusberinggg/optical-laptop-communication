# --- Imports ---

import cv2
import numpy as np
import random

# --- Presets of the amount effects and severity ---

PRESETS = {
    "custom preset": {
        "severity": 0.5,
        "light_level": 1.0,  # normal light
        "effects": ["noise", "jitter_color", "blur", "jpeg_compress", "white_balance_shift"]
    },
    "none": {
        "severity": 0.0,
        "light_level": 1.0,  # normal light
        "effects": []
    },
    "light": {
        "severity": 0.2,
        "light_level": 1.2,  # slightly bright
        "effects": ["noise", "jitter_color", "white_balance_shift"]
    },
    "medium": {
        "severity": 0.5,
        "light_level": 1.0,  # normal
        "effects": ["noise", "jitter_color", "blur", "jpeg_compress", "white_balance_shift"]
    },
    "heavy": {
        "severity": 0.8,
        "light_level": 0.7,  # darker
        "effects": ["noise", "jitter_color", "blur", "warp", "jpeg_compress", "white_balance_shift"]
    },
    "webcam_realistic": {
        "severity": 0.5,
        "light_level": 0.8,  # slightly dim
        "effects": ["noise", "jitter_color", "white_balance_shift", "blur", "jpeg_compress"]
    }
}


# --- Frame distorter ---

class FrameDistorter:
    def __init__(self, preset="webcam_realistic"):
        config = PRESETS[preset]
        self.severity = config["severity"]
        self.light_level = config.get("light_level", 1.0)
        self.effects = []

        # Automatically map effect names to functions
        for name in config["effects"]:
            if hasattr(Effects, name):
                self.effects.append(getattr(Effects, name))
            else:
                raise ValueError(f"Effect '{name}' not found in Effects class.")

    # Apply all effects
    def apply(self, frame):
        # Adjust global brightness first
        frame = np.clip(frame.astype(np.float32) * self.light_level, 0, 255).astype(np.uint8)

        for effect in self.effects:
            # Pass light_level only to effects that need it
            if effect.__name__ in ["add_noise", "jitter_color"]:
                frame = effect(frame, self.severity, self.light_level)
            else:
                frame = effect(frame, self.severity)
        return frame



# --- Effects ---

class Effects:

    # --- Sensor noise ---

    @staticmethod
    def add_noise(frame, severity, light_level):
        """
        Adds realistic sensor-like noise to a frame.
        severity: 0.0 → minimal noise, 1.0 → strong noise
        """
        h, w, c = frame.shape
        noisy = frame.copy().astype(np.float32)

        # Determine % of pixels to corrupt based on severity
        min_amount, max_amount = 0.005, 0.05  # 0.5% → 5%
        amount = min_amount + (max_amount - min_amount) * severity
        n = int(amount * h * w)

        # Randomly choose pixel coordinates
        ys = np.random.randint(0, h, n)
        xs = np.random.randint(0, w, n)

        # scale noise by light level
        noise_intensity = int(20 * (1.0 / max(light_level, 0.1)))  # avoid division by zero
        noise = np.random.randint(-noise_intensity, noise_intensity + 1, (n, 3))

        # Apply noise
        noisy[ys, xs] += noise

        # Clip values to [0, 255] and convert back to uint8
        return np.clip(noisy, 0, 255).astype(np.uint8)


    # --- Color jitter (brightness/contrast) ---

    @staticmethod
    def jitter_color(frame, severity, light_level):
        brightness = int(severity * 40)     # ±40
        contrast = 1 + (severity * 0.4)     # ±40%

        # Random brightness shift
        b = random.randint(-brightness, brightness)
        b = int(b * (1.0 / max(light_level, 0.1)))
        # Random contrast shift
        c = random.uniform(1 - severity * 0.4, 1 + severity * 0.4)

        # Convert to UMat for OpenCL
        frame_gpu = cv2.UMat(frame.astype(np.float32))

        # Apply brightness (addition)
        jittered = cv2.add(frame_gpu, b)

        # Apply contrast (multiplication)
        jittered = cv2.multiply(jittered, c)

        # Clip values to [0,255] on GPU
        jittered = cv2.min(cv2.max(jittered, 0), 255)

        # Convert back to uint8 NumPy array
        return jittered.get().astype(np.uint8)


    # --- Warp / elastic distortion ---

    @staticmethod
    def warp_frame(frame, severity):
        """
        Realistic warp/frame distortion for webcam-like artifacts.
        severity: 0.0 → no effect, 1.0 → strong effect
        """
        h, w, _ = frame.shape

        # Determine warp strength based on severity
        max_strength = 5  # maximum displacement in pixels for strong effect
        strength = severity * max_strength

        # Determine scale of low-res displacement map (smaller = more local distortions)
        min_scale, max_scale = 15, 40  # smaller scale → finer distortions
        scale = int(max_scale - (max_scale - min_scale) * severity)

        # Generate low-res random displacement maps
        dx_small = (np.random.rand(h // scale + 1, w // scale + 1) - 0.5) * strength
        dy_small = (np.random.rand(h // scale + 1, w // scale + 1) - 0.5) * strength

        # Upscale to full frame
        dx = cv2.resize(dx_small, (w, h), interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(dy_small, (w, h), interpolation=cv2.INTER_CUBIC)

        # Optional smoothing for realism
        blur_sigma = max(1.0, 3.0 * (1 - severity))  # stronger effect = less blur
        dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

        # Build remap grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        # Move everything to GPU/OpenCL
        frame_gpu = cv2.UMat(frame)
        map_x_gpu = cv2.UMat(map_x)
        map_y_gpu = cv2.UMat(map_y)

        # Apply remap on GPU
        warped_gpu = cv2.remap(frame_gpu, map_x_gpu, map_y_gpu, cv2.INTER_LINEAR)

        # Convert back to CPU
        warped = warped_gpu.get()
        return warped


    # --- jpeg compression blocking ---

    @staticmethod
    def jpeg_compress(frame, severity):
        encode_quality = int(70 - severity * 50)   # quality 70 → 20

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), encode_quality]
        _, encimg = cv2.imencode('.jpg', frame, encode_param)
        decoded = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        return decoded


    # --- White balance shift (real webcam drift) ---

    @staticmethod
    def white_balance_shift(frame, severity):
        # Convert to float and to UMat for OpenCL
        shifted_gpu = cv2.UMat(frame.astype(np.float32))

        # Random per-channel multipliers
        r_shift = 1 + random.uniform(-0.15, 0.15) * severity
        g_shift = 1 + random.uniform(-0.15, 0.15) * severity
        b_shift = 1 + random.uniform(-0.15, 0.15) * severity

        # Split channels (GPU-backed)
        b, g, r = cv2.split(shifted_gpu)

        # Multiply each channel
        b = cv2.multiply(b, b_shift)
        g = cv2.multiply(g, g_shift)
        r = cv2.multiply(r, r_shift)

        # Merge back
        merged = cv2.merge([b, g, r])

        # Clip and convert back to uint8
        result = cv2.min(cv2.max(merged, 0), 255).get().astype(np.uint8)
        return result


    # --- Soft-Focus Blur (Gaussian Blur) ---

    @staticmethod
    def blur(frame, radius=1.2):
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
    def chromatic_aberration(frame, shift_r=(1.0, 0.0), shift_b=(-1.0, 0.0)):
        """
        frame: HxWxC NumPy array (dtype=np.uint8)
        shift_r, shift_b: pixel shifts for R and B channels (dx, dy)
        """
        H, W, C = frame.shape
        assert C == 3, "Frame must have 3 channels"

        # Convert to float for precision
        frame_f = frame.astype(np.float32)

        def shift_channel(channel, dx, dy):
            # Build translation matrix
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            channel_gpu = cv2.UMat(channel)
            shifted = cv2.warpAffine(channel_gpu, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
            return shifted.get()

        R = shift_channel(frame_f[:, :, 2], *shift_r)
        G = frame_f[:, :, 1]
        B = shift_channel(frame_f[:, :, 0], *shift_b)

        result = cv2.merge([B, G, R])
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
