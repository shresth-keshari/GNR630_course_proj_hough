"""
Hough Transform Suite — T3 Compliant (Satellite Optimized)
==========================================================
Problem T3:
  1. Apply gradient operator on an image containing linear edges.
  2. Threshold the gradient magnitude → binary edge map.
  3. Implement Hough transform using BOTH (m-b and ρ-θ).
  4. Display both accumulator arrays and detected lines.

Rule compliance:
  Built-in functions are used ONLY for I/O and display.
  All math, gradients, and accumulators are pure NumPy.
"""

import customtkinter as ctk
from tkinter import filedialog
import cv2                        
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
import threading

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — PRIMITIVE OPERATIONS  (pure NumPy)
# ══════════════════════════════════════════════════════════════════════════════

def make_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0: size += 1
    k = size // 2
    y, x = np.mgrid[-k : k + 1, -k : k + 1]
    g = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    return (g / g.sum()).astype(np.float64)

def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img.astype(np.float64), ((ph, ph), (pw, pw)), mode='reflect')
    h, w   = img.shape
    out    = np.zeros((h, w), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i : i + h, j : j + w]
    return out

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — GRADIENT OPERATOR  (T3 Step 1)
# ══════════════════════════════════════════════════════════════════════════════

_KX = np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]], dtype=np.float64)
_KY = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=np.float64)

def apply_gradient_operator(blurred: np.ndarray):
    Gx = convolve2d(blurred, _KX)
    Gy = convolve2d(blurred, _KY)
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    direction = np.arctan2(Gy, Gx)
    return Gx, Gy, magnitude, direction

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — THRESHOLD GRADIENT MAGNITUDE  (T3 Step 2)
# ══════════════════════════════════════════════════════════════════════════════

def threshold_magnitude(magnitude: np.ndarray, threshold: float) -> np.ndarray:
    return (magnitude >= threshold).astype(np.uint8) * 255

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3a — HOUGH: SLOPE / INTERCEPT  m-b  (T3 Step 3a)
# ══════════════════════════════════════════════════════════════════════════════

def mb_accumulator(edges: np.ndarray, m_steps: int = 200, b_steps: int = 400, m_max: float = 8.0):
    h, w   = edges.shape
    ys, xs = np.nonzero(edges)

    m_vals = np.linspace(-m_max, m_max, m_steps)
    b_min  = -(h + m_max * w)
    b_max  =   h + m_max * w
    b_vals = np.linspace(b_min, b_max, b_steps)
    acc    = np.zeros((b_steps, m_steps), dtype=np.int32)

    if len(xs) == 0: return acc, m_vals, b_vals

    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)
    b_computed = ys_f[:, None] - xs_f[:, None] * m_vals[None, :]

    b_range = b_max - b_min
    b_idxs  = np.round((b_computed - b_min) / b_range * (b_steps - 1)).astype(np.int32)
    b_idxs  = np.clip(b_idxs, 0, b_steps - 1)
    m_idxs  = np.broadcast_to(np.arange(m_steps)[None, :], b_idxs.shape)

    np.add.at(acc, (b_idxs.ravel(), m_idxs.ravel()), 1)
    return acc, m_vals, b_vals

def draw_mb_lines(img_bgr: np.ndarray, edges: np.ndarray, peaks, m_vals: np.ndarray, b_vals: np.ndarray,
                  color=(0, 140, 255), thickness: int=2, tolerance: float=2.0, max_gap: int=20, min_length: int=15):
    out = img_bgr.copy()
    ys, xs = np.nonzero(edges)
    if len(xs) == 0: return out
        
    xs_f, ys_f = xs.astype(np.float64), ys.astype(np.float64)

    for bi, mi, _ in peaks:
        m, b = float(m_vals[mi]), float(b_vals[bi])
        dist = np.abs(ys_f - (m * xs_f + b))
        mask = dist <= tolerance
        if not np.any(mask): continue
            
        ix, iy = xs[mask], ys[mask]
        sort_idx = np.argsort(ix)
        ix_sorted, iy_sorted = ix[sort_idx], iy[sort_idx]
        
        diffs = np.sqrt(np.diff(ix_sorted.astype(np.float64))**2 + np.diff(iy_sorted.astype(np.float64))**2)
        split_indices = np.where(diffs > max_gap)[0] + 1
        
        for x_seg, y_seg in zip(np.split(ix_sorted, split_indices), np.split(iy_sorted, split_indices)):
            if len(x_seg) == 0: continue
            if np.sqrt((x_seg[-1] - x_seg[0])**2 + (y_seg[-1] - y_seg[0])**2) >= min_length:
                cv2.line(out, (int(x_seg[0]), int(y_seg[0])), (int(x_seg[-1]), int(y_seg[-1])), color, thickness, cv2.LINE_AA)
    return out

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3b — HOUGH: POLAR  ρ-θ  (T3 Step 3b)
# ══════════════════════════════════════════════════════════════════════════════

def rtheta_accumulator(edges: np.ndarray, theta_res_deg: float = 1.0, rho_res: int = 1):
    h, w   = edges.shape
    D      = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
    thetas = np.deg2rad(np.arange(0.0, 180.0, theta_res_deg))
    rhos   = np.arange(-D, D + 1, rho_res)
    acc    = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    ys, xs = np.nonzero(edges)
    if len(xs) == 0: return acc, thetas, rhos

    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    rho_vals = xs[:, None] * cos_t[None, :] + ys[:, None] * sin_t[None, :]
    rho_idxs = np.clip(np.round((rho_vals + D) / rho_res).astype(np.int32), 0, len(rhos) - 1)
    t_idxs = np.broadcast_to(np.arange(len(thetas))[None, :], rho_idxs.shape)
    np.add.at(acc, (rho_idxs.ravel(), t_idxs.ravel()), 1)
    return acc, thetas, rhos

def draw_rtheta_lines(img_bgr: np.ndarray, edges: np.ndarray, peaks, thetas: np.ndarray, rhos: np.ndarray,
                      color=(0, 255, 80), thickness: int=2, tolerance: float=2.0, max_gap: int=20, min_length: int=15):
    out = img_bgr.copy()
    ys, xs = np.nonzero(edges)
    if len(xs) == 0: return out
        
    xs_f, ys_f = xs.astype(np.float64), ys.astype(np.float64)

    for ri, ti, _ in peaks:
        rho, theta = float(rhos[ri]), float(thetas[ti])
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        dist = np.abs(xs_f * cos_t + ys_f * sin_t - rho)
        mask = dist <= tolerance
        if not np.any(mask): continue
            
        ix, iy = xs[mask], ys[mask]
        t = ix * (-sin_t) + iy * cos_t
        
        sort_idx = np.argsort(t)
        t_sorted, x_sorted, y_sorted = t[sort_idx], ix[sort_idx], iy[sort_idx]
        
        diffs = np.diff(t_sorted)
        split_indices = np.where(diffs > max_gap)[0] + 1
        
        for t_seg, x_seg, y_seg in zip(np.split(t_sorted, split_indices), np.split(x_sorted, split_indices), np.split(y_sorted, split_indices)):
            if len(t_seg) == 0: continue
            if (t_seg[-1] - t_seg[0]) >= min_length:
                cv2.line(out, (int(x_seg[0]), int(y_seg[0])), (int(x_seg[-1]), int(y_seg[-1])), color, thickness, cv2.LINE_AA)
    return out

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — PEAK EXTRACTION  (NMS over any 2-D accumulator)
# ══════════════════════════════════════════════════════════════════════════════

def extract_peaks(acc: np.ndarray, n_peaks: int, nhood: int = 20, threshold_ratio: float = 0.3):
    A = acc.astype(np.float32)
    thresh = threshold_ratio * float(A.max())
    peaks = []
    for _ in range(n_peaks):
        idx = int(np.argmax(A))
        ri, ci = np.unravel_index(idx, A.shape)
        val = float(A[ri, ci])
        if val < thresh: break
        peaks.append((ri, ci, val))
        r0, r1 = max(0, ri - nhood), min(A.shape[0], ri + nhood + 1)
        c0, c1 = max(0, ci - nhood), min(A.shape[1], ci + nhood + 1)
        A[r0:r1, c0:c1] = 0.0
    return peaks

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — SMART AUTO-SUGGEST (TEXTURE AWARE)
# ══════════════════════════════════════════════════════════════════════════════

def auto_suggest_params(img_bgr: np.ndarray) -> dict:
    """
    Evaluates image variance to determine if the image is highly textured
    (like satellite data). Adjusts blur and peak thresholds accordingly to
    capture intricate road networks while ignoring building noise.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    # Calculate basic local variance to guess "texture busyness"
    mean_k = np.ones((5, 5)) / 25.0
    local_mean = convolve2d(gray, mean_k)
    local_var = convolve2d((gray - local_mean)**2, mean_k)
    avg_var = np.mean(local_var)
    
    # If high variance (satellite image), we need aggressive settings
    is_satellite = avg_var > 500  

    blur_k = 7 if is_satellite else 5
    blur_sig = 2.5 if is_satellite else 1.5
    
    gauss = make_gaussian_kernel(blur_k, blur_sig)
    blurred = convolve2d(gray, gauss)
    _, _, mag, _ = apply_gradient_operator(blurred)

    nz = mag[mag > 0].ravel()
    
    # For satellite, we want to capture roads but drop roofs.
    # Roads are often strong edges, but colony roads are weak.
    grad_thresh = int(np.clip(np.percentile(nz, 75 if is_satellite else 65), 15, 200)) if len(nz) > 100 else 40

    return dict(
        blur_k=blur_k, blur_sigma=blur_sig,
        grad_thresh=grad_thresh,
        m_steps=200, b_steps=400, m_max=8.0,
        theta_res=1.0, rho_res=1,
        # AGGRESSIVE PEAK SETTINGS FOR COMPLEX ROADS
        n_peaks=150 if is_satellite else 40,   # Massive increase in max lines
        nhood=10 if is_satellite else 20,      # Allow lines to be closer together
        thresh_ratio=0.15 if is_satellite else 0.3, # Allow much weaker lines (colony roads)
        max_gap=30, min_length=20, line_thickness=2,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(img_bgr: np.ndarray, p: dict) -> dict:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    bk = p['blur_k'] if p['blur_k'] % 2 == 1 else p['blur_k'] + 1
    gauss = make_gaussian_kernel(bk, p['blur_sigma'])
    blurred = convolve2d(gray, gauss)          

    Gx, Gy, magnitude, direction = apply_gradient_operator(blurred)
    edges = threshold_magnitude(magnitude, p['grad_thresh'])

    gap, ml, th = int(p['max_gap']), int(p['min_length']), int(p['line_thickness'])

    mb_acc, m_vals, b_vals = mb_accumulator(edges, m_steps=int(p['m_steps']), b_steps=int(p['b_steps']), m_max=float(p['m_max']))
    mb_peaks = extract_peaks(mb_acc, int(p['n_peaks']), int(p['nhood']), float(p['thresh_ratio']))
    img_mb = draw_mb_lines(img_bgr, edges, mb_peaks, m_vals, b_vals, color=(0, 140, 255), thickness=th, max_gap=gap, min_length=ml)

    rt_acc, thetas, rhos = rtheta_accumulator(edges, theta_res_deg=float(p['theta_res']), rho_res=int(p['rho_res']))
    rt_peaks = extract_peaks(rt_acc, int(p['n_peaks']), int(p['nhood']), float(p['thresh_ratio']))
    img_rt = draw_rtheta_lines(img_bgr, edges, rt_peaks, thetas, rhos, color=(0, 255, 80), thickness=th, max_gap=gap, min_length=ml)

    img_combined = draw_rtheta_lines(img_bgr, edges, rt_peaks, thetas, rhos, color=(0, 255, 80), thickness=th, max_gap=gap, min_length=ml)
    img_combined = draw_mb_lines(img_combined, edges, mb_peaks, m_vals, b_vals, color=(0, 140, 255), thickness=th, max_gap=gap, min_length=ml)

    return dict(
        blurred=blurred, magnitude=magnitude, edges=edges, Gx=Gx, Gy=Gy,
        mb_acc=mb_acc, m_vals=m_vals, b_vals=b_vals, mb_peaks=mb_peaks,
        rt_acc=rt_acc, thetas=thetas, rhos=rhos, rt_peaks=rt_peaks,
        img_mb=img_mb, img_rt=img_rt, img_combined=img_combined,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — GUI
# ══════════════════════════════════════════════════════════════════════════════

ACCENT   = "#00e676"
BG_DARK  = "#0d0d0d"
BG_MID   = "#161616"
BG_PANEL = "#1c1c1c"
MB_CLR   = "#ff8c00"
RT_CLR   = "#00e676"

class HoughApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hough Transform Suite — T3 Compliant")
        self.geometry("1560x980")
        ctk.set_appearance_mode("dark")
        self.configure(fg_color=BG_DARK)
        self.img_bgr, self.canvas_widget, self.inputs = None, None, {}
        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self):
        sb = ctk.CTkScrollableFrame(self, width=330, fg_color=BG_MID)
        sb.pack(side="left", fill="y", padx=(12, 0), pady=12)
        self.sidebar = sb

        ctk.CTkLabel(sb, text="HOUGH SUITE", font=ctk.CTkFont("Courier New", 19, "bold"), text_color=ACCENT).pack(pady=(16, 2))
        self._div()

        for txt, cmd, h in [("  RUN PIPELINE", self._run_threaded, 50), ("  Upload Image", self.load_image, 40), ("  Auto-Suggest Params", self.auto_suggest, 36)]:
            ctk.CTkButton(sb, text=txt, font=ctk.CTkFont("Courier New", 13, "bold"), fg_color="#003d1a", hover_color="#005924", text_color=ACCENT, height=h, command=cmd).pack(fill="x", padx=12, pady=(0, 4))
        self._div()

        self._sec("  PREPROCESSING")
        self.inputs['blur_k'] = self._inp("Gaussian kernel size (odd)", 7)
        self.inputs['blur_sigma'] = self._inp("Gaussian sigma", 2.5)
        self._div()

        self._sec("  GRADIENT & THRESHOLD")
        self.inputs['grad_thresh'] = self._inp("Gradient magnitude threshold", 60)
        self._div()

        self._sec(f"  M-B HOUGH  (slope / intercept)", color=MB_CLR)
        self.inputs['m_steps'] = self._inp("Slope bins", 200)
        self.inputs['b_steps'] = self._inp("Intercept bins", 400)
        self.inputs['m_max'] = self._inp("Slope range ±m_max", 8.0)
        self._div()

        self._sec(f"  ρ-θ  HOUGH  (polar)", color=RT_CLR)
        self.inputs['theta_res'] = self._inp("Theta resolution (degrees)", 1.0)
        self.inputs['rho_res'] = self._inp("Rho resolution (pixels)", 1)
        self._div()

        self._sec("  PEAK EXTRACTION  (shared)")
        self.inputs['n_peaks'] = self._inp("Max peaks", 150)
        self.inputs['nhood'] = self._inp("NMS neighbourhood (bins)", 10)
        self.inputs['thresh_ratio'] = self._inp("Peak threshold ratio", 0.15)
        self._div()

        self._sec("  DISPLAY & FINITE LINES")
        self.inputs['max_gap'] = self._inp("Max line gap (pixels)", 30)
        self.inputs['min_length'] = self._inp("Min segment length", 20)
        self.inputs['line_thickness'] = self._inp("Line thickness (px)", 2)

    def _div(self): ctk.CTkFrame(self.sidebar, height=1, fg_color="#252525").pack(fill="x", padx=8, pady=5)
    def _sec(self, title, color="#777"): ctk.CTkLabel(self.sidebar, text=title, font=ctk.CTkFont("Courier New", 11, "bold"), text_color=color).pack(anchor="w", padx=14, pady=(10, 2))
    
    def _inp(self, label, default):
        ctk.CTkLabel(self.sidebar, text=label, font=ctk.CTkFont("Courier New", 11), text_color="#bbb").pack(anchor="w", padx=14, pady=(4, 0))
        e = ctk.CTkEntry(self.sidebar, font=ctk.CTkFont("Courier New", 12), fg_color="#0d0d0d", border_color="#2e2e2e", text_color=ACCENT, height=32)
        e.insert(0, str(default))
        e.pack(fill="x", padx=14, pady=(2, 0))
        return e

    def _build_main(self):
        self.main = ctk.CTkFrame(self, fg_color=BG_DARK)
        self.main.pack(side="right", fill="both", expand=True, padx=12, pady=12)
        bar = ctk.CTkFrame(self.main, height=52, fg_color=BG_PANEL, corner_radius=8)
        bar.pack(fill="x", pady=(0, 10))
        bar.pack_propagate(False)
        self.status_lbl = ctk.CTkLabel(bar, text="Upload an image to begin.", font=ctk.CTkFont("Courier New", 12), text_color="#555")
        self.status_lbl.pack(side="left", padx=16)
        self.graph_frame = ctk.CTkFrame(self.main, fg_color=BG_MID, corner_radius=8)
        self.graph_frame.pack(fill="both", expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not path: return
        img = cv2.imread(path)
        if img is None: return self._status("Failed to load image.", True)
        h, w = img.shape[:2]
        if max(h, w) > 800:
            sc = 800 / max(h, w)
            img = cv2.resize(img, (int(w * sc), int(h * sc)))
        self.img_bgr = img
        self._status(f"Loaded {img.shape[1]} × {img.shape[0]} px — click Auto-Suggest or Run Pipeline")

    def auto_suggest(self):
        if self.img_bgr is None: return self._status("Upload an image first.", True)
        suggested = auto_suggest_params(self.img_bgr)
        for k, v in suggested.items():
            if k in self.inputs:
                self.inputs[k].delete(0, "end")
                self.inputs[k].insert(0, str(v))
        self._status("Auto-suggest complete.")

    def _run_threaded(self):
        if self.img_bgr is None: return self._status("Upload an image first.", True)
        self._status("Processing… (manual convolution — please wait a moment)")
        self.update()
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        try:
            p = {k: int(float(v.get())) if 'steps' in k or 'res' in k and 'theta' not in k or 'peaks' in k or 'nhood' in k or 'gap' in k or 'length' in k or 'thickness' in k else float(v.get()) for k, v in self.inputs.items()}
            # Fix specific types
            p['blur_k'] = int(p['blur_k'])
            p['rho_res'] = int(p['rho_res'])
            
            result = run_pipeline(self.img_bgr, p)
            self.after(0, lambda: self._show_results(result))
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.after(0, lambda: self._status(f"Error: {exc}", True))

    def _show_results(self, r):
        self._status("Rendering…")
        if self.canvas_widget: self.canvas_widget.destroy()

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor(BG_DARK)
        gs  = fig.add_gridspec(2, 3, left=0.04, right=0.98, top=0.93, bottom=0.04, wspace=0.12, hspace=0.22)

        def ax_(row, col):
            a = fig.add_subplot(gs[row, col])
            a.set_facecolor(BG_MID)
            a.tick_params(colors="#444", labelsize=7)
            for sp in a.spines.values(): sp.set_edgecolor("#252525")
            return a

        def ttl(a, txt, color="#dcdcdc"): a.set_title(txt, color=color, fontsize=9, fontfamily="monospace", pad=6)

        a = ax_(0, 0)
        a.imshow(cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB))
        ttl(a, "1. Original image")
        a.axis("off")

        a = ax_(0, 1)
        a.imshow(r['magnitude'] / (r['magnitude'].max() + 1e-9), cmap="hot")
        ttl(a, "2. Gradient magnitude  |∇I|  (Sobel)")
        a.axis("off")

        a = ax_(0, 2)
        a.imshow(r['edges'], cmap="gray")
        ttl(a, f"3. Thresholded edges  ({int(np.sum(r['edges'] > 0))} px)")
        a.axis("off")

        a = ax_(1, 0)
        a.imshow(np.log1p(r['mb_acc']), cmap="magma", aspect="auto", extent=[r['m_vals'].min(), r['m_vals'].max(), r['b_vals'].max(), r['b_vals'].min()])
        if r['mb_peaks']:
            pm, pb = zip(*[(r['m_vals'][mi], r['b_vals'][bi]) for bi, mi, _ in r['mb_peaks']])
            a.scatter(pm, pb, c=MB_CLR, marker="+", s=80, linewidths=1.5, zorder=5)
        a.set_xlabel("Slope  m", color="#666", fontsize=8); a.set_ylabel("Intercept  b", color="#666", fontsize=8)
        ttl(a, f"4. m-b acc  ·  {len(r['mb_peaks'])} peaks", MB_CLR)

        a = ax_(1, 1)
        a.imshow(np.log1p(r['rt_acc']), cmap="inferno", aspect="auto", extent=[np.rad2deg(r['thetas']).min(), np.rad2deg(r['thetas']).max(), r['rhos'].max(), r['rhos'].min()])
        if r['rt_peaks']:
            pt, pr = zip(*[(np.rad2deg(r['thetas'][ti]), r['rhos'][ri]) for ri, ti, _ in r['rt_peaks']])
            a.scatter(pt, pr, c=RT_CLR, marker="+", s=80, linewidths=1.5, zorder=5)
        a.set_xlabel("θ  (degrees)", color="#666", fontsize=8); a.set_ylabel("ρ  (pixels)", color="#666", fontsize=8)
        ttl(a, f"5. ρ-θ acc  ·  {len(r['rt_peaks'])} peaks", RT_CLR)

        a = ax_(1, 2)
        a.imshow(cv2.cvtColor(r['img_combined'], cv2.COLOR_BGR2RGB))
        legend_handles = [Line2D([0], [0], color=RT_CLR, lw=2, label=f"ρ-θ ({len(r['rt_peaks'])})"), Line2D([0], [0], color=MB_CLR, lw=2, label=f"m-b ({len(r['mb_peaks'])})")]
        a.legend(handles=legend_handles, loc="upper right", fontsize=7, facecolor="#1c1c1c", edgecolor="#444", labelcolor="white")
        ttl(a, "6. Detected lines  —  green = ρ-θ   orange = m-b")
        a.axis("off")

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        self.canvas_widget = canvas.get_tk_widget()
        self.canvas_widget.configure(bg=BG_DARK)
        self.canvas_widget.pack(fill="both", expand=True, padx=4, pady=4)
        plt.close(fig)

        self._status(f"Done  —  ρ-θ: {len(r['rt_peaks'])} lines   |   m-b: {len(r['mb_peaks'])} lines")
    
    def _status(self, msg, error=False):
        """Updates the status bar text and color."""
        if hasattr(self, 'status_lbl') and self.status_lbl.winfo_exists():
            self.status_lbl.configure(
                text=msg,
                text_color="#cc4444" if error else "#777"
            )
if __name__ == "__main__":
    app = HoughApp()
    app.mainloop()