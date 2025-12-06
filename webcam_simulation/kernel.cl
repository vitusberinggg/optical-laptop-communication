// kernels.cl
// OpenCL kernels for webcam frame distortion effects
// Jitter, White Balance, Warp, Rolling Shutter, JPEG Approx

#pragma OPENCL EXTENSION cl_khr_fp32 : enable

// ---------- JITTER (brightness/contrast + RGB gains) ----------
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

    img[idx + 0] = (uchar)clamp(b, 0.0f, 255.0f);
    img[idx + 1] = (uchar)clamp(g, 0.0f, 255.0f);
    img[idx + 2] = (uchar)clamp(r, 0.0f, 255.0f);
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

    img[idx + 0] = (uchar)clamp(b, 0.0f, 255.0f);
    img[idx + 1] = (uchar)clamp(g, 0.0f, 255.0f);
    img[idx + 2] = (uchar)clamp(r, 0.0f, 255.0f);
}

// ---------- WARP (bilinear sampling using float maps) ----------
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

    int idx = y * width + x;
    float sx = map_x[idx];
    float sy = map_y[idx];

    // Out-of-bounds: replicate border
    if (sx < 0.0f || sy < 0.0f || sx > (float)(width-1) || sy > (float)(height-1)) {
        int ix = (int)clamp(sx, 0.0f, (float)(width-1));
        int iy = (int)clamp(sy, 0.0f, (float)(height-1));
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

    for(int c=0; c<3; ++c){
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

// ---------- ROLLING SHUTTER ----------
__kernel void rolling_shutter(
    __global uchar *in_img,
    __global uchar *out_img,
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

    for(int c=0; c<3; ++c){
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

// ---------- JPEG_APPROX ----------
__kernel void jpeg_approx(
    __global uchar *in_img,
    __global uchar *out_img,
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

    for(int y = y0; y < y1; y++){
        for(int x = x0; x < x1; x++){
            int idx = (y*width + x)*3;
            float R = (float)in_img[idx + 2];
            float G = (float)in_img[idx + 1];
            float B = (float)in_img[idx + 0];

            float Y = 0.299f*R + 0.587f*G + 0.114f*B;
            float qY = round(Y / luma_scale) * luma_scale;

            float Cb = avgCb - 128.0f;
            float Cr = avgCr - 128.0f;

            float Rq = qY + 1.402f*Cr;
            float Gq = qY - 0.344136f*Cb - 0.714136f*Cr;
            float Bq = qY + 1.772f*Cb;

            out_img[idx + 2] = (uchar)clamp(Rq, 0.0f, 255.0f);
            out_img[idx + 1] = (uchar)clamp(Gq, 0.0f, 255.0f);
            out_img[idx + 0] = (uchar)clamp(Bq, 0.0f, 255.0f);
        }
    }
}
