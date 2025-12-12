'''
Use this command to install pyopencl:
pip install pyopencl
'''

import pyopencl as cl
import numpy as np
import os

class OpenCL:
    def __init__(self):

        
        # Video path

        base = os.path.dirname(__file__)
        kernel_path = os.path.join(base, "kernel.cl")

        #--- Initialize OpenCL context and kernels ---

        # OpenCL memory flags
        self.mf = cl.mem_flags

        # Parameters for noise
        self.base_sigma = 0.0
        self.signal_sigma = 0.0
        self.seed = 0

        # Parameters for color jitter
        self.brightness = 0.0
        self.contrast = 1.0
        self.jitter_r_gain = 1.0
        self.jitter_g_gain = 1.0
        self.jitter_b_gain = 1.0

        # Parameters for white balance
        self.white_r_gain = 1.0
        self.white_g_gain = 1.0 
        self.white_b_gain = 1.0

        # Parameters for warp
        self.map_x_buf = None
        self.map_y_buf = None

        # Parameters for rolling shutter
        self.row_buf = None

        # Parameters for JPEG approximation
        self.block_size = 8
        self.luma_scale = 1.0

        # Parameters for Gaussian blur
        self.kernel_buf = None
        self.ksize = 0  # Kernel size for Gaussian blur

        # Parameters for chromatic aberration
        self.dx_r = 0.0
        self.dy_r = 0.0
        self.dx_b = 0.0
        self.dy_b = 0.0



        # Use Intel GPU by default if available
        platforms = cl.get_platforms()
        device = None

        for p in platforms:
            for d in p.get_devices():
                if "Intel" in d.vendor and d.type & cl.device_type.GPU:
                    device = d
                    break

        # fallback to CPU if no GPU found
        if device is None:
            device = platforms[0].get_devices()[0]

        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)

        # Load kernel.cl from file
        kernel_src = self.load_kernel_file(kernel_path)
        self.prg = cl.Program(self.ctx, kernel_src).build()
        
        # Create OpenCL kernels to able to reuse them
        self.kernel_jpeg_approx = cl.Kernel(self.prg, "jpeg_approx")
        self.kernel_image_distortion = cl.Kernel(self.prg, "image_distortion")

    @staticmethod
    def load_kernel_file(path):
        with open(path, 'r') as f:
            return f.read()
        

    # --- Noise ---
        
    def run_noise(self, severity, light_level):
        # print("[DEBUG] noise")

        # Noise parameters
        self.base_sigma = 2.0 * severity
        self.signal_sigma = 0.02 * severity
        light_factor = 1.0 / max(light_level, 0.15)
        self.base_sigma *= light_factor
        self.signal_sigma *= light_factor

        # Generate a different seed per frame
        self.seed = np.random.randint(0, 0x7FFFFFFF, dtype=np.uint32)


    # --- Jitter color ---
    
    def run_jitter(self, brightness, contrast, gains):
        # print("[DEBUG] jitter")
        self.jitter_r_gain, self.jitter_g_gain, self.jitter_b_gain = gains

        self.brightness = brightness
        self.contrast = contrast
    

    # --- White balance shifter ---

    def run_white_balance(self, gains):
        # print("[DEBUG] white balance")

        self.white_r_gain, self.white_g_gain, self.white_b_gain = gains
    

    # ---------- Warp (GPU remap) ----------
    
    def run_warp(self, map_x, map_y):
        # print("[DEBUG] warp")
        """
        frame: HxWx3 uint8
        map_x,map_y: HxW float32 pixel coords
        """

        self.map_x_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(map_x.astype(np.float32)))
        self.map_y_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(map_y.astype(np.float32)))


    # ---------- Rolling shutter ----------

    def run_rolling_shutter(self, row_offset):
        # print("[DEBUG] rilling shutter")
        """
        row_offset: 1D float32 array of length height (per-row horizontal offsets in pixels)
        """

        self.row_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, 
                                 hostbuf=np.ascontiguousarray(row_offset.astype(np.float32)))


    # ---------- JPEG approximation ----------

    def run_jpeg_approx(self, block_size=8, quality=50):
        # print("[DEBUG] jpeg")
        """
        block_size: 8 (typical) or 16 for stronger blocking
        quality: 0..100 (higher = less aggressive compression)
        This routine is an approximation: lowers luma precision and averages chroma per-block.
        """

        self.block_size = block_size
        # convert quality to luma_scale: lower quality -> larger step
        # clamp quality
        q = max(1, min(100, int(quality)))
        # map quality 100->1.0 (no quant), quality 1->10.0 (very coarse)
        self.luma_scale = 0.5 + (101 - q) * 0.2

    
    # --- Blur ---

    def run_blur(self, radius=1.5):
        # print("[DEBUG] blur")
        """
        frame: HxWx3 uint8
        radius: Gaussian sigma in pixels
        """

        # Create kernel weights
        self.ksize = max(3, min(15, int(radius*4)|1))  # odd, 3–15
        khalf = self.ksize // 2
        x = np.arange(-khalf, khalf+1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / radius)**2)
        kernel /= np.sum(kernel)

        self.kernel_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=kernel)

    
    # --- Chromatic Abberation ---

    def run_chromatic_aberration(self, severity):
        # print("[DEBUG] chromatic aberration")
        """
        Applies subtle RGB channel shift like real webcam chromatic aberration.
        severity: 0.0 = none, 1.0 = maximum shift (~2 pixels)
        """

        max_shift = 2.0
        self.dx_r = severity * max_shift
        self.dy_r = 0.0
        self.dx_b = -severity * max_shift
        self.dy_b = 0.0
    

    # --- Image distortion ---
    
    def run_image_distortion(self, frame, which_effects):
        """
        This method is a placeholder for collective effects that might be applied to the frame.
        It can be extended to apply multiple effects in a single pass.
        """
        # This method can be implemented to combine multiple effects

        #print("Running image distortion with effects:", which_effects)

        mask_np = np.array(which_effects, dtype=np.uint32)
        bitmask_buf = cl.Buffer(
            self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=mask_np
        )

        h, w, _ = frame.shape

        src = np.ascontiguousarray(frame)
        out = np.empty_like(src)

        img_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=src)
        img_temp = cl.Buffer(self.ctx, self.mf.READ_WRITE, out.nbytes)

        self.kernel_image_distortion(
            self.queue,
            (w, h), None,

            # Input and output images
            img_buf,
            img_temp,

            # Image dimensions
            np.int32(w),
            np.int32(h),

            # Bitmask for active effects
            bitmask_buf,

            # Noise parameters
            np.float32(self.base_sigma),
            np.float32(self.signal_sigma),
            np.uint32(self.seed),

            # Jitter parameters
            np.float32(self.brightness),
            np.float32(self.contrast),
            np.float32(self.jitter_r_gain),
            np.float32(self.jitter_g_gain),
            np.float32(self.jitter_b_gain),

            # White balance parameters
            np.float32(self.white_r_gain),
            np.float32(self.white_g_gain),
            np.float32(self.white_b_gain),

            # Warp parameters
            self.map_x_buf,
            self.map_y_buf,

            # Rolling shutter parameter
            self.row_buf,

            # Gaussian blur kernel
            self.kernel_buf,
            np.int32(self.ksize),

            # Chromatic aberration parameters
            np.float32(self.dx_r),
            np.float32(self.dy_r),
            np.float32(self.dx_b),
            np.float32(self.dy_b)
        )

        out = np.empty_like(frame)

        if which_effects[7] == 1:
            
            # launch blocks grid
            bx = (w + self.block_size - 1) // self.block_size
            by = (h + self.block_size - 1) // self.block_size
   
            self.kernel_jpeg_approx(
                self.queue,
                (bx, by), None,
                img_buf,
                img_temp,
                np.int32(w),
                np.int32(h),
                np.int32(self.block_size),
                np.float32(self.luma_scale)
            )

            cl.enqueue_copy(self.queue, out, img_temp)

        else:
            cl.enqueue_copy(self.queue, out, img_buf)

        return out


