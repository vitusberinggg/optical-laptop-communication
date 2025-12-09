// kernels.cl
// OpenCL kernels for webcam frame distortion effects
// Jitter, White Balance, Warp, Rolling Shutter, JPEG Approx

#pragma OPENCL EXTENSION cl_khr_fp32 : enable


// Simple hash function to generate pseudo-random per-pixel noise - NOISE

inline uint hash(uint x, uint y, uint c, uint seed) {
    uint h = x * 374761393u + y * 668265263u + c * 2147483647u + seed * 1274126177u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h;
}


inline void swap_images(
    __global uchar *src,
    __global uchar *dst,
    int width,
    int height
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int idx = (y*width + x)*3;

    src[idx+0] = dst[idx+0];
    src[idx+1] = dst[idx+1];
    src[idx+2] = dst[idx+2];
}

// -------- NOISE ----------

inline void add_noise(
    __global uchar *img,
    float base_sigma,
    float signal_sigma,
    int width,
    int height,
    uint seed
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    for (int c = 0; c < 3; c++) {
        uint h = hash(x, y, c, seed);
        float noise = ((float)(h & 0xFFFF) / 65535.0f - 0.5f); // -0.5..0.5
        float pixel = (float)img[idx + c];
        float sigma = base_sigma + signal_sigma * pixel;
        pixel += noise * sigma * 2.0f; // scale to match standard deviation
        img[idx + c] = (uchar)clamp(pixel, 0.0f, 255.0f);
    }
}


// ------- JITTER ----------

inline void jitter(
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

    img[idx + 0] = (uchar)clamp(b, 0.0f, 255.0f);
    img[idx + 1] = (uchar)clamp(g, 0.0f, 255.0f);
    img[idx + 2] = (uchar)clamp(r, 0.0f, 255.0f);
}


// ---------- WHITE BALANCE ----------

inline void white_balance(
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

    img[idx + 0] = (uchar)clamp(b, 0.0f, 255.0f);
    img[idx + 1] = (uchar)clamp(g, 0.0f, 255.0f);
    img[idx + 2] = (uchar)clamp(r, 0.0f, 255.0f);
}


// ---------- WARP ----------

inline void warp(
    __global uchar *img,
    __global uchar *temp_img,
    __global float *map_x,
    __global float *map_y,
    int width,
    int height
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float sx = map_x[idx];
    float sy = map_y[idx];

    // Out-of-bounds: replicate border
    if (sx < 0.0f || sy < 0.0f || sx > (float)(width-1) || sy > (float)(height-1)) {
        int ix = (int)clamp(sx, 0.0f, (float)(width-1));
        int iy = (int)clamp(sy, 0.0f, (float)(height-1));
        int in_idx = (iy * width + ix) * 3;
        temp_img[idx*3 + 0] = img[in_idx + 0];
        temp_img[idx*3 + 1] = img[in_idx + 1];
        temp_img[idx*3 + 2] = img[in_idx + 2];

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

    for(int c=0; c<3; ++c){
        float v00 = (float)img[idx00 + c];
        float v10 = (float)img[idx10 + c];
        float v01 = (float)img[idx01 + c];
        float v11 = (float)img[idx11 + c];

        float v0 = v00 * (1.0f - wx) + v10 * wx;
        float v1 = v01 * (1.0f - wx) + v11 * wx;
        float v = v0 * (1.0f - wy) + v1 * wy;
        temp_img[idx*3 + c] = (uchar)clamp(v, 0.0f, 255.0f);
    }
}


// ---------- ROLLING SHUTTER ----------

inline void rolling_shutter(
    __global uchar *img,
    __global uchar *temp_img,
    __global float *row_offset,
    int width,
    int height
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x >= width || y >= height) return;

    int idx = y * width + x;
    float ofs = row_offset[y];

    float sx = (float)x + ofs;
    float sy = (float)y;

    if (sx < 0.0f || sy < 0.0f || sx > (float)(width-1) || sy > (float)(height-1)) {
        int ix = (int)clamp(sx, 0.0f, (float)(width-1));
        int iy = (int)clamp(sy, 0.0f, (float)(height-1));
        int in_idx = (iy * width + ix) * 3;
        temp_img[idx*3 + 0] = img[in_idx + 0];
        temp_img[idx*3 + 1] = img[in_idx + 1];
        temp_img[idx*3 + 2] = img[in_idx + 2];

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

    for(int c=0; c<3; ++c){
        float v00 = (float)img[idx00 + c];
        float v10 = (float)img[idx10 + c];
        float v01 = (float)img[idx01 + c];
        float v11 = (float)img[idx11 + c];

        float v0 = v00 * (1.0f - wx) + v10 * wx;
        float v1 = v01 * (1.0f - wx) + v11 * wx;
        float v = v0 * (1.0f - wy) + v1 * wy;
        temp_img[idx*3 + c] = (uchar)clamp(v, 0.0f, 255.0f);
    }
}


// ---------- JPEG_APPROX ----------

__kernel void jpeg_approx(
    __global uchar *img,
    __global uchar *temp_img,
    int width,
    int height,
    int block_size,
    float luma_scale
){
    int bx = get_global_id(0);
    int by = get_global_id(1);

    int bs = block_size;
    int x0 = bx * bs;
    int y0 = by * bs;
    if(x0 >= width || y0 >= height) return;

    int x1 = min(x0 + bs, width);
    int y1 = min(y0 + bs, height);

    float sumCb = 0.0f;
    float sumCr = 0.0f;
    int count = 0;

    for(int y = y0; y < y1; y++){
        for(int x = x0; x < x1; x++){
            int idx = (y*width + x)*3;
            float R = (float)img[idx + 2];
            float G = (float)img[idx + 1];
            float B = (float)img[idx + 0];

            float Cb = -0.168736f*R - 0.331264f*G + 0.5f*B + 128.0f;
            float Cr = 0.5f*R - 0.418688f*G - 0.081312f*B + 128.0f;

            sumCb += Cb;
            sumCr += Cr;
            count++;
        }
    }

    float avgCb = sumCb / count;
    float avgCr = sumCr / count;

    for(int y = y0; y < y1; y++){
        for(int x = x0; x < x1; x++){
            int idx = (y*width + x)*3;
            float R = (float)img[idx + 2];
            float G = (float)img[idx + 1];
            float B = (float)img[idx + 0];

            float Y = 0.299f*R + 0.587f*G + 0.114f*B;
            float qY = round(Y / luma_scale) * luma_scale;

            float Cb = avgCb - 128.0f;
            float Cr = avgCr - 128.0f;

            float Rq = qY + 1.402f*Cr;
            float Gq = qY - 0.344136f*Cb - 0.714136f*Cr;
            float Bq = qY + 1.772f*Cb;

            temp_img[idx + 2] = (uchar)clamp(Rq, 0.0f, 255.0f);
            temp_img[idx + 1] = (uchar)clamp(Gq, 0.0f, 255.0f);
            temp_img[idx + 0] = (uchar)clamp(Bq, 0.0f, 255.0f);
        }
    }
}


// ---------- GAUSSIAN BLUR ----------

inline void gaussian_blur(
    __global uchar *img,
    __global uchar *temp_img,
    __global float *kernel_array,   // 1D Gaussian weights
    int ksize,
    int width,
    int height
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x >= width || y >= height) return;

    int khalf = ksize / 2;
    float3 acc = (float3)(0.0f, 0.0f, 0.0f);

    for(int i=-khalf; i<=khalf; i++){
        int sx = clamp(x+i, 0, width-1);
        int idx = (y*width + sx)*3;
        float w = kernel_array[i + khalf];
        acc.x += w * (float)img[idx + 0];
        acc.y += w * (float)img[idx + 1];
        acc.z += w * (float)img[idx + 2];
    }

    int idx = (y*width + x)*3;
    temp_img[idx+0] = (uchar)clamp(acc.x, 0.0f, 255.0f);
    temp_img[idx+1] = (uchar)clamp(acc.y, 0.0f, 255.0f);
    temp_img[idx+2] = (uchar)clamp(acc.z, 0.0f, 255.0f);
}


// ---------- CHROMATIC ABERRATION ----------

inline void chromatic_aberration(
    __global uchar *img,
    __global uchar *temp_img,
    int width,
    int height,
    float dx_r,
    float dy_r,
    float dx_b,
    float dy_b
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    // Coordinates for red and blue channels
    int xr = clamp((int)round(x + dx_r), 0, width-1);
    int yr = clamp((int)round(y + dy_r), 0, height-1);
    int xb = clamp((int)round(x + dx_b), 0, width-1);
    int yb = clamp((int)round(y + dy_b), 0, height-1);

    int idx_r = (yr * width + xr) * 3;
    int idx_b = (yb * width + xb) * 3;

    // Apply shift
    temp_img[idx + 0] = img[idx_b + 0]; // B
    temp_img[idx + 1] = img[idx + 1];   // G stays
    temp_img[idx + 2] = img[idx_r + 2]; // R
}



__kernel void image_distortion(

    // Input and output images
    __global uchar *img,
    __global uchar *temp_img,

    // Image dimensions
    int width,
    int height,

    // Bitmask for active effects
    __global uint *mask,

    // Noise parameters
    float base_sigma,
    float signal_sigma,
    uint seed,

    // Jitter parameters
    float brightness,
    float contrast,
    float jitter_r_gain,
    float jitter_g_gain,
    float jitter_b_gain,

    // White balance parameters
    float white_r_gain,
    float white_g_gain,
    float white_b_gain,

    // Warp parameters
    __global float *map_x,
    __global float *map_y,

    // Rolling shutter parameters
    __global float *row_offset,

    // Gaussian blur parameters
    __global float *kernel_array,   // 1D Gaussian weights
    int ksize,

    // Chromatic aberration parameters
    float dx_r,
    float dy_r,
    float dx_b,
    float dy_b
) {

    if (mask[0] == 1){
        // Apply noise
        add_noise(img, base_sigma, signal_sigma, width, height, seed);
    }
    
    if (mask[1] == 1){
        // Apply color jitter
        jitter(img, brightness, contrast, jitter_r_gain, jitter_g_gain, jitter_b_gain, width, height);
    }

    if (mask[2] == 1){
        // Apply white balance
        white_balance(img, white_r_gain, white_g_gain, white_b_gain, width, height);
    }

    if (mask[3] == 1){
        // Apply rolling shutter
        rolling_shutter(img, temp_img, row_offset, width, height);
        swap_images(img, temp_img, width, height);
    }

    if (mask[4] == 1){
        // Apply warp
        warp(img, temp_img, map_x, map_y, width, height);
        swap_images(img, temp_img, width, height);
    }

    if (mask[5] == 1){
        // Apply chromatic aberration
        chromatic_aberration(img, temp_img, width, height, dx_r, dy_r, dx_b, dy_b);
        swap_images(img, temp_img, width, height);
    }

    if (mask[6] == 1){
        // Apply Gaussian blur
        gaussian_blur(img, temp_img, kernel_array, ksize, width, height);
        swap_images(img, temp_img, width, height);
    }
}