"""
Hough Transform Suite — T3 Compliant (Satellite Optimized)
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ══════════════════════════════════════════════════════════════════════════════
#  MATHEMATICAL IMPLEMENTATION (Strictly Intact)
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

_KX = np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]], dtype=np.float64)
_KY = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=np.float64)

def apply_gradient_operator(blurred: np.ndarray):
    Gx = convolve2d(blurred, _KX)
    Gy = convolve2d(blurred, _KY)
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    direction = np.arctan2(Gy, Gx)
    return Gx, Gy, magnitude, direction

def threshold_magnitude(magnitude: np.ndarray, threshold: float) -> np.ndarray:
    return (magnitude >= threshold).astype(np.uint8) * 255

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

def auto_suggest_params(img_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mean_k = np.ones((5, 5)) / 25.0
    local_mean = convolve2d(gray, mean_k)
    local_var = convolve2d((gray - local_mean)**2, mean_k)
    avg_var = np.mean(local_var)
    
    is_satellite = avg_var > 500  

    blur_k = 7 if is_satellite else 5
    blur_sig = 2.5 if is_satellite else 1.5
    
    gauss = make_gaussian_kernel(blur_k, blur_sig)
    blurred = convolve2d(gray, gauss)
    _, _, mag, _ = apply_gradient_operator(blurred)
    nz = mag[mag > 0].ravel()
    
    grad_thresh = int(np.clip(np.percentile(nz, 75 if is_satellite else 65), 15, 200)) if len(nz) > 100 else 40

    return dict(
        blur_k=blur_k, blur_sigma=blur_sig,
        grad_thresh=grad_thresh,
        m_steps=200, b_steps=400, m_max=8.0,
        theta_res=1.0, rho_res=1,
        n_peaks=150 if is_satellite else 40,
        nhood=10 if is_satellite else 20,
        thresh_ratio=0.15 if is_satellite else 0.3,
        max_gap=30, min_length=20, line_thickness=2,
    )

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

    # Combined image retained for mathematical compliance, but not rendered in UI to separate the outputs
    img_combined = draw_rtheta_lines(img_bgr, edges, rt_peaks, thetas, rhos, color=(0, 255, 80), thickness=th, max_gap=gap, min_length=ml)
    img_combined = draw_mb_lines(img_combined, edges, mb_peaks, m_vals, b_vals, color=(0, 140, 255), thickness=th, max_gap=gap, min_length=ml)

    return dict(
        blurred=blurred, magnitude=magnitude, edges=edges, Gx=Gx, Gy=Gy,
        mb_acc=mb_acc, m_vals=m_vals, b_vals=b_vals, mb_peaks=mb_peaks,
        rt_acc=rt_acc, thetas=thetas, rhos=rhos, rt_peaks=rt_peaks,
        img_mb=img_mb, img_rt=img_rt, img_combined=img_combined,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  GUI APPLICATION (Refactored to Tkinter + Tabs)
# ══════════════════════════════════════════════════════════════════════════════

class HoughTkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hough Transform Suite — Standard UI")
        self.geometry("1400x850")
        
        self.img_bgr = None
        self.is_dark_mode = True
        
        self.themes = {
            "light": {"bg": "#f0f0f0", "fg": "#000000", "entry_bg": "#ffffff", "entry_fg": "#000000", "plot_bg": "#ffffff"},
            "dark":  {"bg": "#2b2b2b", "fg": "#ffffff", "entry_bg": "#404040", "entry_fg": "#ffffff", "plot_bg": "#3c3f41"}
        }

        # 1. Left Panel (Controls)
        self.control_frame = tk.Frame(self, width=400)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=15)

        self.btn_theme = tk.Button(self.control_frame, text="☀️ Toggle Light Mode", command=self.toggle_theme, font=("Arial", 9))
        self.btn_theme.pack(fill=tk.X, pady=(0, 10))

        self.btn_upload = tk.Button(self.control_frame, text="📁 Upload Image", command=self.load_image, font=("Arial", 10, "bold"), bg="#1a73e8", fg="white")
        self.btn_upload.pack(fill=tk.X, pady=2)

        self.btn_auto = tk.Button(self.control_frame, text="✨ Auto-Suggest Params", command=self.auto_suggest, font=("Arial", 10), bg="#17a2b8", fg="white")
        self.btn_auto.pack(fill=tk.X, pady=2)

        self.lbl_file = tk.Label(self.control_frame, text="No image selected")
        self.lbl_file.pack(fill=tk.X, pady=2)

        tk.Frame(self.control_frame, height=2, bg="#ccc").pack(fill=tk.X, pady=5)
        self.lbl_params = tk.Label(self.control_frame, text="⚙ Parameters", font=("Arial", 11, "bold"))
        self.lbl_params.pack(anchor="w")

        # Dictionary for inputs
        self.params = {
            "blur_k": tk.IntVar(value=7),
            "blur_sigma": tk.DoubleVar(value=2.5),
            "grad_thresh": tk.IntVar(value=60),
            "m_steps": tk.IntVar(value=200),
            "b_steps": tk.IntVar(value=400),
            "m_max": tk.DoubleVar(value=8.0),
            "theta_res": tk.DoubleVar(value=1.0),
            "rho_res": tk.IntVar(value=1),
            "n_peaks": tk.IntVar(value=150),
            "nhood": tk.IntVar(value=10),
            "thresh_ratio": tk.DoubleVar(value=0.15),
            "max_gap": tk.IntVar(value=30),
            "min_length": tk.IntVar(value=20),
            "line_thickness": tk.IntVar(value=2)
        }

        labels = {
            "blur_k": "Gauss Size (odd):", "blur_sigma": "Gauss Sigma:",
            "grad_thresh": "Gradient Thresh:", "m_steps": "Slope Bins:",
            "b_steps": "Intercept Bins:", "m_max": "Slope ±m_max:",
            "theta_res": "Theta Res (°):", "rho_res": "Rho Res (px):",
            "n_peaks": "Max Peaks:", "nhood": "NMS Radius:",
            "thresh_ratio": "Peak Thresh Ratio:", "max_gap": "Max Line Gap:",
            "min_length": "Min Length:", "line_thickness": "Line Thickness:"
        }

        # Grid layout for inputs to save vertical space
        self.entry_widgets = []
        self.label_widgets = [self.lbl_file, self.lbl_params]
        
        param_container = tk.Frame(self.control_frame)
        param_container.pack(fill=tk.X)
        self.entry_widgets.append(param_container)

        keys = list(self.params.keys())
        for i in range(0, len(keys), 2):
            k1 = keys[i]
            k2 = keys[i+1] if i+1 < len(keys) else None

            lbl1 = tk.Label(param_container, text=labels[k1], anchor="w", width=16)
            lbl1.grid(row=i//2, column=0, pady=2, sticky="w")
            ent1 = tk.Entry(param_container, textvariable=self.params[k1], width=6)
            ent1.grid(row=i//2, column=1, pady=2, padx=(0, 10))
            self.label_widgets.append(lbl1)
            self.entry_widgets.append(ent1)

            if k2:
                lbl2 = tk.Label(param_container, text=labels[k2], anchor="w", width=16)
                lbl2.grid(row=i//2, column=2, pady=2, sticky="w")
                ent2 = tk.Entry(param_container, textvariable=self.params[k2], width=6)
                ent2.grid(row=i//2, column=3, pady=2)
                self.label_widgets.append(lbl2)
                self.entry_widgets.append(ent2)

        tk.Frame(self.control_frame, height=2, bg="#ccc").pack(fill=tk.X, pady=10)

        self.btn_run = tk.Button(self.control_frame, text="▶ Run Pipeline", command=self.run_pipeline_ui, font=("Arial", 10, "bold"), bg="#28a745", fg="white")
        self.btn_run.pack(fill=tk.X, pady=5)
        
        self.lbl_status = tk.Label(self.control_frame, text="Status: Ready", fg="green")
        self.lbl_status.pack(fill=tk.X, pady=2)
        self.label_widgets.append(self.lbl_status)

        # ─── Parameter Guide ───
        guide_text = (
            "📖 PARAMETER GUIDE\n\n"
            "• Gauss Size & Sigma: Controls blurring to reduce noise.\n"
            "  ↑ Increase to smooth out complex, irrelevant textures.\n"
            "• Gradient Thresh: Cutoff for edge mapping.\n"
            "  ↑ Increase: Only sharpest edges survive.\n"
            "  ↓ Decrease: Faint edges are retained.\n"
            "• Slope/Intercept Bins (m-b): Hough space resolution.\n"
            "  ↑ Increase: Higher precision, slower processing.\n"
            "• Theta/Rho Res (ρ-θ): Polar space precision.\n"
            "  ↓ Decrease (e.g. 0.5°): Finer angular sensitivity.\n"
            "• Max Peaks: Absolute limit on detected lines.\n"
            "• NMS Radius: Spacing between lines.\n"
            "  ↑ Increase: Forces lines to be further apart.\n"
            "• Peak Thresh Ratio: Minimum relative strength of a line.\n"
            "  ↓ Decrease (e.g. 0.15): Allows detection of faint roads.\n"
            "• Max Line Gap: allowed pixel gap to connect segments.\n"
            "• Min Length: Minimum length of a valid line.\n"
        )
        self.txt_guide = tk.Text(self.control_frame, wrap=tk.WORD, height=14, font=("Consolas", 8))
        self.txt_guide.insert(tk.END, guide_text)
        self.txt_guide.config(state=tk.DISABLED)
        self.txt_guide.pack(fill=tk.BOTH, expand=True, pady=10)

        # 2. Right Panel (Tabs for Images)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.axes = {}
        self.canvases = {}
        self.figures = {}

        # M-B and Rho-Theta lines separated into distinct tabs
        tab_names = [
            "Original Image", "Gradient Magnitude", "Thresholded Edges",
            "m-b Accumulator", "m-b Lines", "ρ-θ Accumulator", "ρ-θ Lines"
        ]

        for name in tab_names:
            frame = tk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            
            fig = Figure(figsize=(6, 5), dpi=100)
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            self.figures[name] = fig
            self.axes[name] = ax
            self.canvases[name] = canvas

        self.apply_theme()

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.btn_theme.config(text="🌙 Toggle Dark Mode" if not self.is_dark_mode else "☀️ Toggle Light Mode")
        self.apply_theme()

    def apply_theme(self):
        theme = self.themes["dark"] if self.is_dark_mode else self.themes["light"]
        self.config(bg=theme["bg"])
        self.control_frame.config(bg=theme["bg"])
        
        for widget in self.label_widgets:
            widget.config(bg=theme["bg"], fg=theme["fg"])
            
        for widget in self.entry_widgets:
            if isinstance(widget, tk.Entry):
                widget.config(bg=theme["entry_bg"], fg=theme["entry_fg"], insertbackground=theme["fg"])
            elif isinstance(widget, tk.Frame):
                widget.config(bg=theme["bg"])
                
        self.txt_guide.config(bg=theme["entry_bg"], fg=theme["entry_fg"])

        for name, fig in self.figures.items():
            fig.patch.set_facecolor(theme["plot_bg"])
            self.axes[name].set_facecolor(theme["plot_bg"])
            self.canvases[name].draw()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff")])
        if path:
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                if max(h, w) > 800:
                    sc = 800 / max(h, w)
                    img = cv2.resize(img, (int(w * sc), int(h * sc)))
                self.img_bgr = img
                self.lbl_file.config(text=os.path.basename(path))
                self.lbl_status.config(text="Status: Image Loaded. Awaiting Run.")
            else:
                messagebox.showerror("Error", "Could not read the image.")

    def auto_suggest(self):
        if self.img_bgr is None:
            messagebox.showwarning("Notice", "Please upload an image first.")
            return
        suggested = auto_suggest_params(self.img_bgr)
        for k, v in suggested.items():
            if k in self.params:
                self.params[k].set(v)
        self.lbl_status.config(text="Status: Auto-Suggest Complete.")

    def run_pipeline_ui(self):
        if self.img_bgr is None:
            messagebox.showwarning("Notice", "Please upload an image first.")
            return
        
        try:
            p_dict = {k: var.get() for k, var in self.params.items()}
        except tk.TclError:
            messagebox.showerror("Input Error", "Please ensure all parameters are valid numbers.")
            return

        self.lbl_status.config(text="Status: Processing...", fg="orange")
        self.update_idletasks()

        try:
            # Run Mathematical Pipeline
            r = run_pipeline(self.img_bgr, p_dict)

            def to_rgb(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Clear all plots
            for ax in self.axes.values(): 
                ax.clear()
                ax.axis('off')

            # Render Tab 1: Original
            self.axes["Original Image"].imshow(to_rgb(self.img_bgr))

            # Render Tab 2: Magnitude
            self.axes["Gradient Magnitude"].imshow(r['magnitude'] / (r['magnitude'].max() + 1e-9), cmap="hot")

            # Render Tab 3: Edges
            self.axes["Thresholded Edges"].imshow(r['edges'], cmap="gray")

            # Render Tab 4: M-B Accumulator
            ax = self.axes["m-b Accumulator"]
            ax.axis('on')
            ax.imshow(np.log1p(r['mb_acc']), cmap="magma", aspect="auto", extent=[r['m_vals'].min(), r['m_vals'].max(), r['b_vals'].max(), r['b_vals'].min()])
            if r['mb_peaks']:
                pm, pb = zip(*[(r['m_vals'][mi], r['b_vals'][bi]) for bi, mi, _ in r['mb_peaks']])
                ax.scatter(pm, pb, c="#ff8c00", marker="+", s=80, linewidths=1.5, zorder=5)
            ax.set_xlabel("Slope m", color=self.themes["dark"]["fg"] if self.is_dark_mode else "black")
            ax.set_ylabel("Intercept b", color=self.themes["dark"]["fg"] if self.is_dark_mode else "black")
            ax.tick_params(colors=self.themes["dark"]["fg"] if self.is_dark_mode else "black")

            # Render Tab 5: M-B Lines (Separated)
            self.axes["m-b Lines"].imshow(to_rgb(r['img_mb']))

            # Render Tab 6: Rho-Theta Accumulator
            ax = self.axes["ρ-θ Accumulator"]
            ax.axis('on')
            ax.imshow(np.log1p(r['rt_acc']), cmap="inferno", aspect="auto", extent=[np.rad2deg(r['thetas']).min(), np.rad2deg(r['thetas']).max(), r['rhos'].max(), r['rhos'].min()])
            if r['rt_peaks']:
                pt, pr = zip(*[(np.rad2deg(r['thetas'][ti]), r['rhos'][ri]) for ri, ti, _ in r['rt_peaks']])
                ax.scatter(pt, pr, c="#00e676", marker="+", s=80, linewidths=1.5, zorder=5)
            ax.set_xlabel("Theta (degrees)", color=self.themes["dark"]["fg"] if self.is_dark_mode else "black")
            ax.set_ylabel("Rho (pixels)", color=self.themes["dark"]["fg"] if self.is_dark_mode else "black")
            ax.tick_params(colors=self.themes["dark"]["fg"] if self.is_dark_mode else "black")

            # Render Tab 7: Rho-Theta Lines (Separated)
            self.axes["ρ-θ Lines"].imshow(to_rgb(r['img_rt']))

            # Update canvases
            for canvas in self.canvases.values():
                canvas.draw()

            self.lbl_status.config(text=f"Done! m-b: {len(r['mb_peaks'])} | ρ-θ: {len(r['rt_peaks'])}", fg="green")

        except Exception as e:
            self.lbl_status.config(text="Status: Failed.", fg="red")
            messagebox.showerror("Execution Error", str(e))

if __name__ == "__main__":
    app = HoughTkApp()
    app.mainloop()