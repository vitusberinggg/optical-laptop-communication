# bitgrid_debugger.py
"""
Bitgrid debugger utility for visualizing per-cell statistics and sampling used by
your BitColorTracker pipeline.

Usage:
    from bitgrid_debugger import debug_bitgrid
    debug_bitgrid(hsv_roi, LUT, color_names)

If a file named "debug_hsv_roi.npy" exists in the working directory the module
will run a quick self-test when executed as __main__.
"""

import cv2
import numpy as np
import os
from utilities.global_definitions import number_of_rows as ROWS, number_of_columns as COLS
from utilities.color_functions_v3_1 import tracker, dominant_color
from utilities.decoding_functions_v3_1 import decode_bitgrid


def _split_into_cells(hsv, rows=ROWS, cols=COLS):
    """Return list of cell slices and cell geometry (cell_h, cell_w).
    hsv: HxWx3 (HSV) array
    returns: (cells, cell_h, cell_w) where cells is shape (rows, cols, cell_h, cell_w, 3)
    """
    H, W = hsv.shape[:2]
    cell_h = int(np.ceil(H / rows))
    cell_w = int(np.ceil(W / cols))
    padded_H = cell_h * rows
    padded_W = cell_w * cols
    pad_bottom = padded_H - H
    pad_right = padded_W - W

    padded = np.pad(hsv, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='edge')
    # reshape into grid
    cells = padded.reshape(rows, cell_h, cols, cell_w, 3)
    # reorder to (rows, cols, cell_h, cell_w, 3)
    cells = np.transpose(cells, (0, 2, 1, 3, 4))
    return cells, cell_h, cell_w


def compute_cell_stats(cell):
    """Return dict of basic stats for a single cell (HSV).
    cell shape: (cell_h, cell_w, 3)
    """
    H = cell[:, :, 0]
    S = cell[:, :, 1]
    V = cell[:, :, 2]
    stats = {
        'H_min': int(H.min()), 'H_max': int(H.max()), 'H_med': float(np.median(H)),
        'S_min': int(S.min()), 'S_max': int(S.max()), 'S_med': float(np.median(S)),
        'V_min': int(V.min()), 'V_max': int(V.max()), 'V_med': float(np.median(V)),
        'V_mean': float(V.mean()),
    }
    return stats


def debug_bitgrid(hsv_roi, LUT=None, color_names=None, rows=ROWS, cols=COLS, save_path='bitgrid_debug.png'):
    """Enhanced debugger: overlay visual per-cell stats with LUT colors and HSV info"""
    if hsv_roi is None:
        raise ValueError('hsv_roi is None')

    cells, cell_h, cell_w = _split_into_cells(hsv_roi, rows=rows, cols=cols)

    H_total = cell_h * rows
    W_total = cell_w * cols
    overlay = np.zeros((H_total, W_total, 3), dtype=np.uint8)

    cell_stats = [[None for _ in range(cols)] for _ in range(rows)]

    # define visible colors for LUT classes (BGR)
    default_colors = [(0,0,0), (0,0,255), (255,255,255), (0,255,0), (255,0,0)]  # black, red, white, green, blue
    class_colors = {name: default_colors[i % len(default_colors)] for i, name in enumerate(color_names or [])}

    pad_h = max(1, cell_h // 4)
    pad_w = max(1, cell_w // 4)

    for r in range(rows):
        for c in range(cols):
            cell = cells[r, c]
            ch0, ch1 = pad_h, cell.shape[0]-pad_h
            cw0, cw1 = pad_w, cell.shape[1]-pad_w
            sample = cell[ch0:ch1, cw0:cw1] if ch1>ch0 and cw1>cw0 else cell

            stats = compute_cell_stats(sample)

            dominant_class = None
            dominant_class_name = None
            if LUT is not None and color_names is not None:
                Hs = sample[:, :, 0].astype(np.uint16)
                Ss = sample[:, :, 1].astype(np.uint16)
                Vs = sample[:, :, 2].astype(np.uint16)
                classes = LUT[Hs, Ss, Vs]
                values, counts = np.unique(classes, return_counts=True)
                if len(values) > 0:
                    dominant_class = int(values[counts.argmax()])
                    dominant_class_name = color_names[dominant_class]

            stats.update({'dominant_class': dominant_class, 'dominant_class_name': dominant_class_name})
            cell_stats[r][c] = stats

            # draw cell colored by dominant class
            top, left = r*cell_h, c*cell_w
            bottom, right = top+cell_h, left+cell_w
            fill_color = class_colors.get(dominant_class_name, (128,128,128)) if dominant_class_name else (128,128,128)
            overlay[top:bottom, left:right] = fill_color

            # draw green border
            cv2.rectangle(overlay, (left, top), (right, bottom), (0,255,0), 1)

            # text: median V, dominant class, median H/S/V
            texts = [
                f"cls={dominant_class_name}" if dominant_class_name else "cls=?",
                f"H={int(stats['H_med'])}",
                f"S={int(stats['S_med'])}",
                f"V={int(stats['V_med'])}"
            ]
            for i, text in enumerate(texts):
                cv2.putText(overlay, text, (left+2, top+12 + i*14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)

    # save overlay and cell stats
    cv2.imwrite(save_path, overlay)
    print(f"Saved visual debug overlay to: {save_path}")
    np.save('bitgrid_cell_stats.npy', np.array(cell_stats, dtype=object))
    print('Saved per-cell stats to bitgrid_cell_stats.npy')

    return cell_stats


def debug_bitgrid_realtime(hsv_roi, LUT=None, color_names=None, rows=ROWS, cols=COLS, window_name='Bitgrid Debug'):
    """
    Real-time visual debugger for BitColorTracker.
    Displays live overlay showing per-cell stats (dominant class + H/S/V medians)
    """

    if hsv_roi is None:
        raise ValueError('hsv_roi is None')

    cells, cell_h, cell_w = _split_into_cells(hsv_roi, rows=rows, cols=cols)

    H_total = cell_h * rows
    W_total = cell_w * cols
    overlay = np.zeros((H_total, W_total, 3), dtype=np.uint8)

    # visible colors for LUT classes (BGR)
    default_colors = [(0,0,0), (0,0,255), (255,255,255), (0,255,0), (255,0,0)]
    class_colors = {name: default_colors[i % len(default_colors)] for i, name in enumerate(color_names or [])}

    pad_h = max(1, cell_h // 4)
    pad_w = max(1, cell_w // 4)

    for r in range(rows):
        for c in range(cols):
            cell = cells[r, c]
            ch0, ch1 = pad_h, cell.shape[0]-pad_h
            cw0, cw1 = pad_w, cell.shape[1]-pad_w
            sample = cell[ch0:ch1, cw0:cw1] if ch1>ch0 and cw1>cw0 else cell

            H = sample[:, :, 0]
            S = sample[:, :, 1]
            V = sample[:, :, 2]
            H_med = int(np.median(H))
            S_med = int(np.median(S))
            V_med = int(np.median(V))

            dominant_class_name = None
            if LUT is not None and color_names is not None:
                Hs = sample[:, :, 0].astype(np.uint16)
                Ss = sample[:, :, 1].astype(np.uint16)
                Vs = sample[:, :, 2].astype(np.uint16)
                classes = LUT[Hs, Ss, Vs]
                values, counts = np.unique(classes, return_counts=True)
                if len(values) > 0:
                    dominant_class = int(values[counts.argmax()])
                    dominant_class_name = color_names[dominant_class]

            # draw cell colored by dominant class
            top, left = r*cell_h, c*cell_w
            bottom, right = top+cell_h, left+cell_w
            fill_color = class_colors.get(dominant_class_name, (128,128,128)) if dominant_class_name else (128,128,128)
            overlay[top:bottom, left:right] = fill_color

            # draw green border
            cv2.rectangle(overlay, (left, top), (right, bottom), (0,255,0), 1)

            # overlay text: dominant class + H/S/V medians
            texts = [
                f"cls={dominant_class_name}" if dominant_class_name else "cls=?",
                f"H={H_med}",
                f"S={S_med}",
                f"V={V_med}"
            ]
            for i, text in enumerate(texts):
                cv2.putText(overlay, text, (left+2, top+12 + i*14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)

    return overlay


if __name__ == '__main__':
    
    img = cv2.imread(r"C:\Users\ejadmax\code\optical-laptop-communication\debugging\debug_img.png")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    try:
        from utilities.color_functions_v3_1 import build_color_LUT
        # try to build LUT from common default ranges if available in module
        default_ranges = {
            'red': (np.array([0,100,100]), np.array([10,255,255])),
            'red2': (np.array([160,100,100]), np.array([179,255,255])),
            'white': (np.array([0,0,220]), np.array([180,25,255])),
            'black': (np.array([0,0,0]), np.array([180,255,35])),
            'green': (np.array([45,80,80]), np.array([75,255,255])),
            'blue': (np.array([95,120,70]), np.array([130,255,255]))
        }
        LUT, color_names = build_color_LUT(default_ranges)
        tracker.colors(LUT, color_names)
    except Exception:
        LUT = None
        color_names = None

    for i in range (5):
        if i < 4:
            decode_bitgrid(hsv, True, False, False)
        else:
            decode_bitgrid(hsv, True, False, True)

    message = decode_bitgrid(hsv, False, True, False)
    print(f"Decoded message: {message}")
    