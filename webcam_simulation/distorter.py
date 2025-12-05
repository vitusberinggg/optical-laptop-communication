# --- Imports ---

import cv2
import numpy as np
import random

# --- Presets of the amount effects and severity ---

PRESETS = {
    "light": {
        "severity": 0.2,
        "effects": [
            "noise",
            "jitter_color"
        ]
    },

    "medium": {
        "severity": 0.5,
        "effects": [
            "noise",
            "jitter_color",
            "motion_blur",
            "jpeg_compress"
        ]
    },

    "heavy": {
        "severity": 0.8,
        "effects": [
            "noise",
            "jitter_color",
            "motion_blur",
            "warp",
            "jpeg_compress",
            "white_balance_shift"
        ]
    },

    "webcam_realistic": {
        "severity": 0.6,
        "effects": [
            "noise",                 # low-light noise
            "jitter_color",          # brightness/contrast auto-adjust
            "white_balance_shift",   # color drift
            "motion_blur",           # movement blur
            "jpeg_compress"          # compression blocks
        ]
    }
}


# --- Frame distorter ---

class FrameDistorter:
    def __init__(self, preset="webcam_realistic"):
        config = PRESETS[preset]
        self.severity = config["severity"]
        self.effects = []
        self.effect_names = config["effects"]

    # add the wanted effects
    def add_effect(self, effect_fn):
        self.effects.append(effect_fn)

    # applying the effects in the given frame
    def apply(self, frame):
        for effect in self.effects:
            frame = effect(frame)
        return frame


# --- Effects ---


class Effects:

    # --- Sensor noise ---

    @staticmethod
    def add_noise(frame, severity):
        # amount of pixels to corrupt
        amount = 0.005 + severity * 0.05   # 0.5% → 5%
        noisy = frame.copy()

        h, w, _ = noisy.shape
        n = int(amount * h * w)

        ys = np.random.randint(0, h, n)
        xs = np.random.randint(0, w, n)
        noisy[ys, xs] = np.random.randint(0, 256, (n, 3))

        return noisy


    # --- Color jitter (brightness/contrast) ---

    @staticmethod
    def jitter_color(frame, severity):
        brightness = int(severity * 40)     # ±40
        contrast = 1 + (severity * 0.4)     # ±40%

        # Random brightness shift
        b = random.randint(-brightness, brightness)
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
        strength = severity * 10  # warp intensity
        h, w, _ = frame.shape

        # Random displacement maps
        dx = (np.random.rand(h, w) - 0.5) * strength
        dy = (np.random.rand(h, w) - 0.5) * strength

        # Original grid
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
