'''
Use this command to install pyopencl:
pip install pyopencl
'''


import pyopencl as cl
import numpy as np

class OpenCL:
    def __init__(self):
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
        kernel_src = self.load_kernel_file("kernel.cl")
        self.prg = cl.Program(self.ctx, kernel_src).build()

    @staticmethod
    def load_kernel_file(path):
        with open(path, 'r') as f:
            return f.read()
    
    # --- Jitter color ---
    
    def run_jitter(self, frame, brightness, contrast, gains):
        h, w, _ = frame.shape
        r_gain, g_gain, b_gain = gains

        # transfer frame to GPU
        mf = cl.mem_flags
        img_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=frame)

        # run kernel
        self.prg.jitter(
            self.queue, 
            (w, h), None,
            img_buf,
            np.float32(brightness),
            np.float32(contrast),
            np.float32(r_gain),
            np.float32(g_gain),
            np.float32(b_gain),
            np.int32(w),
            np.int32(h)
        )

        # copy back to CPU
        out = np.empty_like(frame)
        cl.enqueue_copy(self.queue, out, img_buf)
        return out
    
    # --- White balance shifter ---

    def run_white_balance(self, frame, gains):
        h, w, _ = frame.shape
        r_gain, g_gain, b_gain = gains

        mf = cl.mem_flags
        img_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=frame)

        self.prg.white_balance(
            self.queue,
            (w, h), None,
            img_buf,
            np.float32(r_gain),
            np.float32(g_gain),
            np.float32(b_gain),
            np.int32(w),
            np.int32(h)
        )

        out = np.empty_like(frame)
        cl.enqueue_copy(self.queue, out, img_buf)
        return out
    
    # ---------- Warp (GPU remap) ----------
    def run_warp(self, frame, map_x, map_y):
        """
        frame: HxWx3 uint8
        map_x,map_y: HxW float32 pixel coords
        """
        h, w, _ = frame.shape
        mf = cl.mem_flags
        src = np.ascontiguousarray(frame)
        out = np.empty_like(src)
        img_in = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src)
        img_out = cl.Buffer(self.ctx, mf.WRITE_ONLY, out.nbytes)
        map_x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(map_x.astype(np.float32)))
        map_y_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(map_y.astype(np.float32)))

        self.prg.warp(
            self.queue,
            (w, h), None,
            img_in,
            img_out,
            map_x_buf,
            map_y_buf,
            np.int32(w),
            np.int32(h)
        )

        cl.enqueue_copy(self.queue, out, img_out)
        return out

    # ---------- Rolling shutter ----------
    def run_rolling_shutter(self, frame, row_offset):
        """
        row_offset: 1D float32 array of length height (per-row horizontal offsets in pixels)
        """
        h, w, _ = frame.shape
        mf = cl.mem_flags
        src = np.ascontiguousarray(frame)
        out = np.empty_like(src)
        img_in = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src)
        img_out = cl.Buffer(self.ctx, mf.WRITE_ONLY, out.nbytes)
        row_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(row_offset.astype(np.float32)))

        self.prg.rolling_shutter(
            self.queue,
            (w, h), None,
            img_in,
            img_out,
            row_buf,
            np.int32(w),
            np.int32(h)
        )

        cl.enqueue_copy(self.queue, out, img_out)
        return out

    # ---------- JPEG approximation ----------
    def run_jpeg_approx(self, frame, block_size=8, quality=50):
        """
        block_size: 8 (typical) or 16 for stronger blocking
        quality: 0..100 (higher = less aggressive compression)
        This routine is an approximation: lowers luma precision and averages chroma per-block.
        """
        h, w, _ = frame.shape
        mf = cl.mem_flags
        src = np.ascontiguousarray(frame)
        out = np.empty_like(src)
        img_in = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src)
        img_out = cl.Buffer(self.ctx, mf.WRITE_ONLY, out.nbytes)

        # convert quality to luma_scale: lower quality -> larger step
        # clamp quality
        q = max(1, min(100, int(quality)))
        # map quality 100->1.0 (no quant), quality 1->10.0 (very coarse)
        luma_scale = 0.5 + (101 - q) * 0.2

        # launch blocks grid
        bx = (w + block_size - 1) // block_size
        by = (h + block_size - 1) // block_size

        self.prg.jpeg_approx(
            self.queue,
            (bx, by), None,
            img_in,
            img_out,
            np.int32(w),
            np.int32(h),
            np.int32(block_size),
            np.float32(luma_scale)
        )

        cl.enqueue_copy(self.queue, out, img_out)
        return out

