'''
Using this because UMat makes an unwanted blue tint and want to have 
as much of the distortion on the gpu to keep off as much load as possible 
from the cpu

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

        # Kernel will be built once
        self.prg = cl.Program(self.ctx, self.kernel_source()).build()

    @staticmethod
    def kernel_source():
        return r"""
        // ---------- JITTER (existing) ----------
        __kernel void jitter(
            __global uchar *img,
            float brightness,
            float contrast,
            float r_gain,
            float g_gain,
            float b_gain,
            int width,
            int height
        ){
            int x = get_global_id(0);
            int y = get_global_id(1);
            if (x >= width || y >= height) return;
            int idx = (y * width + x) * 3;
            float b = (float)img[idx + 0];
            float g = (float)img[idx + 1];
            float r = (float)img[idx + 2];
            b = (b + brightness) * contrast * b_gain;
            g = (g + brightness) * contrast * g_gain;
            r = (r + brightness) * contrast * r_gain;
            b = clamp(b, 0.0f, 255.0f);
            g = clamp(g, 0.0f, 255.0f);
            r = clamp(r, 0.0f, 255.0f);
            img[idx + 0] = (uchar)b;
            img[idx + 1] = (uchar)g;
            img[idx + 2] = (uchar)r;
        }

        // ---------- WHITE BALANCE ----------
        __kernel void white_balance(
            __global uchar *img,
            float r_gain,
            float g_gain,
            float b_gain,
            int width,
            int height
        ){
            int x = get_global_id(0);
            int y = get_global_id(1);
            if (x >= width || y >= height) return;
            int idx = (y * width + x) * 3;
            float b = (float)img[idx + 0] * b_gain;
            float g = (float)img[idx + 1] * g_gain;
            float r = (float)img[idx + 2] * r_gain;
            b = clamp(b, 0.0f, 255.0f);
            g = clamp(g, 0.0f, 255.0f);
            r = clamp(r, 0.0f, 255.0f);
            img[idx + 0] = (uchar)b;
            img[idx + 1] = (uchar)g;
            img[idx + 2] = (uchar)r;
        }

        // ---------- WARP: sample with bilinear filtering using float maps ----------
        // map_x,map_y are in pixel coordinates (float) size width*height
        __kernel void warp(
            __global uchar *in_img,
            __global uchar *out_img,
            __global float *map_x,
            __global float *map_y,
            int width,
            int height
        ){
            int x = get_global_id(0);
            int y = get_global_id(1);
            if (x >= width || y >= height) return;

            int idx = (y * width + x);
            float sx = map_x[idx];
            float sy = map_y[idx];

            // bilinear sample boundaries
            if (sx < 0.0f || sy < 0.0f || sx > (float)(width - 1) || sy > (float)(height - 1)) {
                // out of bounds: replicate border
                int ix = (int)clamp(sx, 0.0f, (float)(width - 1));
                int iy = (int)clamp(sy, 0.0f, (float)(height - 1));
                int in_idx = (iy * width + ix) * 3;
                out_img[idx*3 + 0] = in_img[in_idx + 0];
                out_img[idx*3 + 1] = in_img[in_idx + 1];
                out_img[idx*3 + 2] = in_img[in_idx + 2];
                return;
            }

            int x0 = (int)floor(sx);
            int y0 = (int)floor(sy);
            int x1 = min(x0 + 1, width - 1);
            int y1 = min(y0 + 1, height - 1);

            float wx = sx - (float)x0;
            float wy = sy - (float)y0;

            int idx00 = (y0 * width + x0) * 3;
            int idx10 = (y0 * width + x1) * 3;
            int idx01 = (y1 * width + x0) * 3;
            int idx11 = (y1 * width + x1) * 3;

            // channel B,G,R
            for (int c=0; c<3; ++c) {
                float v00 = (float)in_img[idx00 + c];
                float v10 = (float)in_img[idx10 + c];
                float v01 = (float)in_img[idx01 + c];
                float v11 = (float)in_img[idx11 + c];

                float v0 = v00 * (1.0f - wx) + v10 * wx;
                float v1 = v01 * (1.0f - wx) + v11 * wx;
                float v = v0 * (1.0f - wy) + v1 * wy;
                out_img[idx*3 + c] = (uchar)clamp(v, 0.0f, 255.0f);
            }
        }

        // ---------- ROLLING SHUTTER / READOUT WOBBLE ----------
        // per-row horizontal offset (in pixels) applied to each row; offset_map is float per row
        __kernel void rolling_shutter(
            __global uchar *in_img,
            __global uchar *out_img,
            __global float *row_offset, // length = height
            int width,
            int height
        ){
            int x = get_global_id(0);
            int y = get_global_id(1);
            if (x >= width || y >= height) return;

            int idx = (y * width + x);
            float ofs = row_offset[y]; // offset in pixels, positive => shift right

            float sx = (float)x + ofs;
            float sy = (float)y;

            // bilinear sample similar to warp (reuse code)
            if (sx < 0.0f || sy < 0.0f || sx > (float)(width - 1) || sy > (float)(height - 1)) {
                int ix = (int)clamp(sx, 0.0f, (float)(width - 1));
                int iy = (int)clamp(sy, 0.0f, (float)(height - 1));
                int in_idx = (iy * width + ix) * 3;
                out_img[idx*3 + 0] = in_img[in_idx + 0];
                out_img[idx*3 + 1] = in_img[in_idx + 1];
                out_img[idx*3 + 2] = in_img[in_idx + 2];
                return;
            }

            int x0 = (int)floor(sx);
            int y0 = (int)floor(sy);
            int x1 = min(x0 + 1, width - 1);
            int y1 = min(y0 + 1, height - 1);

            float wx = sx - (float)x0;
            float wy = sy - (float)y0;

            int idx00 = (y0 * width + x0) * 3;
            int idx10 = (y0 * width + x1) * 3;
            int idx01 = (y1 * width + x0) * 3;
            int idx11 = (y1 * width + x1) * 3;

            for (int c=0; c<3; ++c) {
                float v00 = (float)in_img[idx00 + c];
                float v10 = (float)in_img[idx10 + c];
                float v01 = (float)in_img[idx01 + c];
                float v11 = (float)in_img[idx11 + c];

                float v0 = v00 * (1.0f - wx) + v10 * wx;
                float v1 = v01 * (1.0f - wx) + v11 * wx;
                float v = v0 * (1.0f - wy) + v1 * wy;
                out_img[idx*3 + c] = (uchar)clamp(v, 0.0f, 255.0f);
            }
        }

        // ---------- JPEG_APPROX improved ----------
        __kernel void jpeg_approx(
            __global uchar *in_img,
            __global uchar *out_img,
            int width,
            int height,
            int block_size,
            float luma_scale
        ){
            int bx = get_global_id(0); // block x
            int by = get_global_id(1); // block y

            int bs = block_size;
            int x0 = bx * bs;
            int y0 = by * bs;

            if(x0 >= width || y0 >= height) return;

            int x1 = min(x0 + bs, width);
            int y1 = min(y0 + bs, height);

            // First, compute block average Cb/Cr
            float sumCb = 0.0f;
            float sumCr = 0.0f;
            int count = 0;

            for(int y = y0; y < y1; y++){
                for(int x = x0; x < x1; x++){
                    int idx = (y * width + x) * 3;
                    float R = (float)in_img[idx + 2];
                    float G = (float)in_img[idx + 1];
                    float B = (float)in_img[idx + 0];

                    float Cb = -0.168736f*R - 0.331264f*G + 0.5f*B + 128.0f;
                    float Cr = 0.5f*R - 0.418688f*G - 0.081312f*B + 128.0f;

                    sumCb += Cb;
                    sumCr += Cr;
                    count++;
                }
            }

            float avgCb = sumCb / count;
            float avgCr = sumCr / count;

            // Now process each pixel: quantize Y and reconstruct RGB
            for(int y = y0; y < y1; y++){
                for(int x = x0; x < x1; x++){
                    int idx = (y * width + x) * 3;
                    float R = (float)in_img[idx + 2];
                    float G = (float)in_img[idx + 1];
                    float B = (float)in_img[idx + 0];

                    // Convert to YCbCr
                    float Y = 0.299f*R + 0.587f*G + 0.114f*B;

                    // Quantize Y per block
                    float qY = round(Y / luma_scale) * luma_scale;

                    // Use averaged Cb/Cr
                    float Cb = avgCb - 128.0f;
                    float Cr = avgCr - 128.0f;

                    // Reconstruct RGB
                    float Rq = qY + 1.402f * Cr;
                    float Gq = qY - 0.344136f*Cb - 0.714136f*Cr;
                    float Bq = qY + 1.772f*Cb;

                    out_img[idx + 2] = (uchar)clamp(Rq, 0.0f, 255.0f);
                    out_img[idx + 1] = (uchar)clamp(Gq, 0.0f, 255.0f);
                    out_img[idx + 0] = (uchar)clamp(Bq, 0.0f, 255.0f);
                }
            }
        }

        """

    
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

