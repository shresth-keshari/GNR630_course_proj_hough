import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# STEP 1: Load and preprocess satellite image
# ─────────────────────────────────────────────
def preprocess(img_bgr, blur_ksize=3, blur_sigma=1.0):
    """
    Convert to grayscale and apply Gaussian blur.
    For satellite imagery, a small kernel is preferred
    because roads are thin and we don't want to blur them away.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)
    return blurred

# ─────────────────────────────────────────────
# STEP 2: Edge detection
# ─────────────────────────────────────────────
def detect_edges(gray, low=50, high=150):
    """
    Canny is strongly preferred over simple Sobel thresholding for
    satellite images because:
    - It uses hysteresis (two thresholds) so weak edges connected
      to strong ones are kept, isolated noise is dropped.
    - It does non-maximum suppression: edges are THIN (1px wide).
      Thin edges = clean sinusoids = sharp accumulator peaks.

    low/high: lower ratio ~1:2 or 1:3 is typical.
    For satellite images, try low=30, high=100 first.
    """
    edges = cv2.Canny(gray, low, high, apertureSize=3, L2gradient=True)
    return edges

# ─────────────────────────────────────────────
# STEP 3A: Standard Hough — vectorized, correct
# ─────────────────────────────────────────────
def hough_accumulator(edges, theta_res_deg=1.0, rho_res=1):
    """
    Builds the (rho, theta) accumulator correctly and efficiently.

    theta runs 0 to 180 degrees (not -90 to 90).
    rho runs -D to +D where D is the image diagonal.
    The accumulator fill is fully vectorized — no Python loops.
    """
    h, w = edges.shape
    D = int(np.ceil(np.sqrt(h**2 + w**2)))

    # Theta bins: 0° to just below 180°
    thetas = np.deg2rad(np.arange(0, 180, theta_res_deg))
    # Rho bins: -D to +D
    rhos = np.arange(-D, D + 1, rho_res)

    cos_t = np.cos(thetas)   # shape: (N_theta,)
    sin_t = np.sin(thetas)   # shape: (N_theta,)

    # Get all edge pixel coordinates
    ys, xs = np.nonzero(edges)  # shape: (N_pts,)

    # Vectorized accumulator fill:
    # xs[:, None] broadcasts xs over all theta values at once
    # Result shape: (N_pts, N_theta)
    rho_vals = xs[:, None] * cos_t[None, :] + ys[:, None] * sin_t[None, :]

    # Map rho values to bin indices
    # rho_vals range is [-D, D], we shift by D and divide by rho_res
    rho_idxs = np.round((rho_vals + D) / rho_res).astype(np.int32)

    # Clip to valid range
    rho_idxs = np.clip(rho_idxs, 0, len(rhos) - 1)

    # Build theta index array matching shape of rho_idxs
    theta_idxs = np.tile(np.arange(len(thetas)), (len(xs), 1))  # (N_pts, N_theta)

    # Fill accumulator using np.add.at (handles duplicate indices correctly)
    acc = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    np.add.at(acc, (rho_idxs.ravel(), theta_idxs.ravel()), 1)

    return acc, thetas, rhos

# ─────────────────────────────────────────────
# STEP 3B: Peak extraction with NMS
# ─────────────────────────────────────────────
def extract_peaks(acc, n_peaks, nhood=20, threshold_ratio=0.3):
    """
    Find local maxima in the accumulator.

    threshold_ratio: ignore any peak below this fraction of the max vote.
    This is important for satellite images to avoid spurious detections
    from building edges, shadows, etc.

    nhood: suppress this many bins around each found peak.
    Too small → duplicate detections of the same road.
    Too large → misses parallel roads close in angle/distance.
    """
    acc_copy = acc.copy().astype(np.float32)
    threshold = threshold_ratio * acc_copy.max()
    peaks = []

    for _ in range(n_peaks):
        idx = np.argmax(acc_copy)
        rho_i, theta_i = np.unravel_index(idx, acc_copy.shape)
        val = acc_copy[rho_i, theta_i]

        if val < threshold:
            break  # Remaining peaks are too weak

        peaks.append((rho_i, theta_i, int(val)))

        # Suppress neighborhood to prevent re-detecting same line
        r0 = max(0, rho_i - nhood)
        r1 = min(acc.shape[0], rho_i + nhood + 1)
        t0 = max(0, theta_i - nhood)
        t1 = min(acc.shape[1], theta_i + nhood + 1)
        acc_copy[r0:r1, t0:t1] = 0

    return peaks

# ─────────────────────────────────────────────
# STEP 4: Draw infinite lines (Standard Hough)
# ─────────────────────────────────────────────
def draw_hough_lines(img_bgr, peaks, thetas, rhos, color=(0, 255, 0), thickness=2):
    result = img_bgr.copy()
    h, w = result.shape[:2]

    for rho_i, theta_i, votes in peaks:
        rho = rhos[rho_i]
        theta = thetas[theta_i]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Extend line far in both directions
        x1 = int(x0 + 3000 * (-b))
        y1 = int(y0 + 3000 * (a))
        x2 = int(x0 - 3000 * (-b))
        y2 = int(y0 - 3000 * (a))
        cv2.line(result, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    return result

# ─────────────────────────────────────────────
# STEP 5 (BETTER): Probabilistic Hough for roads
# ─────────────────────────────────────────────
def detect_road_segments(edges,
                          rho=1,
                          theta=np.pi/180,
                          threshold=80,
                          min_line_length=60,
                          max_line_gap=25):
    """
    Probabilistic Hough Transform — the right tool for road detection.

    threshold:       minimum votes a line must get to be considered.
                     Tune this based on image resolution.
                     Too low → spurious lines from buildings/shadows.
                     Too high → misses faint or partial roads.

    min_line_length: minimum pixel length of a line segment to keep.
                     Set to roughly the minimum road length you care about
                     in pixels. Kills short noise segments.

    max_line_gap:    maximum gap (pixels) to bridge when merging collinear
                     segments. Set to roughly the width of an intersection
                     or tree canopy gap. This is the key parameter for
                     handling road interruptions.
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    return lines  # Shape: (N, 1, 4) — each row is [x1, y1, x2, y2]

def draw_segments(img_bgr, lines, color=(0, 255, 0), thickness=2):
    result = img_bgr.copy()
    if lines is None:
        return result
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return result

# ─────────────────────────────────────────────
# STEP 6: Full pipeline — put it all together
# ─────────────────────────────────────────────
def run_pipeline(img_path,
                 canny_low=30, canny_high=100,
                 hough_threshold=80,
                 min_line_length=60,
                 max_line_gap=30):

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {img_path}")

    # Resize if very large (satellite images can be huge)
    max_dim = 1024
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Preprocessing
    gray = preprocess(img, blur_ksize=3, blur_sigma=1.0)
    edges = detect_edges(gray, low=canny_low, high=canny_high)

    # Standard Hough (for understanding the accumulator)
    acc, thetas, rhos = hough_accumulator(edges, theta_res_deg=1.0, rho_res=1)
    peaks = extract_peaks(acc, n_peaks=30, nhood=20, threshold_ratio=0.3)
    img_standard = draw_hough_lines(img, peaks, thetas, rhos)

    # Probabilistic Hough (for actual road segment detection)
    lines = detect_road_segments(
        edges,
        threshold=hough_threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap
    )
    img_prob = draw_segments(img, lines)

    # Visualize
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#1a1a1a')
    for ax in axs.flatten():
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.title.set_color('white')

    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Original")

    axs[0, 1].imshow(gray, cmap='gray')
    axs[0, 1].set_title("Grayscale + Blur")

    axs[0, 2].imshow(edges, cmap='gray')
    axs[0, 2].set_title("Canny Edges")

    axs[1, 0].imshow(np.log1p(acc), cmap='hot', aspect='auto')
    axs[1, 0].set_title("Hough Accumulator (ρ-θ)")

    axs[1, 1].imshow(cv2.cvtColor(img_standard, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title("Standard Hough Lines")

    axs[1, 2].imshow(cv2.cvtColor(img_prob, cv2.COLOR_BGR2RGB))
    axs[1, 2].set_title("Probabilistic Hough Segments")

    plt.tight_layout()
    plt.savefig("hough_output.png", dpi=150, bbox_inches='tight')
    plt.show()

    n_segments = len(lines) if lines is not None else 0
    print(f"Standard Hough peaks found: {len(peaks)}")
    print(f"Probabilistic Hough segments found: {n_segments}")

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline(
        "../New folder/100794_sat.jpg",
        canny_low=30,
        canny_high=100,
        hough_threshold=80,
        min_line_length=60,   # tune: road must be at least this many pixels long
        max_line_gap=30        # tune: gaps up to this size get bridged
    )


# import customtkinter as ctk
# from tkinter import filedialog
# import cv2
# import numpy as np
# from scipy.signal import convolve2d
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# # ==========================================
# # MATHEMATICAL FUNCTIONS (From Scratch)
# # ==========================================
# def gaussian_kernel(size, sigma):
#     ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
#     xx, yy = np.meshgrid(ax, ax)
#     kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
#     return kernel / np.sum(kernel)

# def extract_hough_peaks(acc, num_peaks, neighborhood_size):
#     acc_copy = np.copy(acc)
#     peaks = []
#     votes = []
#     for _ in range(num_peaks):
#         idx = np.argmax(acc_copy)
#         rho_idx, theta_idx = np.unravel_index(idx, acc_copy.shape)
#         vote = acc_copy[rho_idx, theta_idx]
#         if vote == 0:
#             break
#         peaks.append((rho_idx, theta_idx))
#         votes.append(vote)
        
#         # Suppress neighborhood
#         r_min = max(0, rho_idx - neighborhood_size)
#         r_max = min(acc.shape[0], rho_idx + neighborhood_size)
#         t_min = max(0, theta_idx - neighborhood_size)
#         t_max = min(acc.shape[1], theta_idx + neighborhood_size)
#         acc_copy[r_min:r_max, t_min:t_max] = 0
#     return peaks, votes

# # ==========================================
# # MAIN GUI APPLICATION
# # ==========================================
# class AdvancedHoughApp(ctk.CTk):
#     def __init__(self):
#         super().__init__()
#         self.title("Advanced Hough Transform Suite")
#         self.geometry("1600x900")
#         ctk.set_appearance_mode("Dark")

#         self.img_color = None
#         self.canvas_widget = None

#         # --- Sidebar (Scrollable for many inputs) ---
#         self.sidebar = ctk.CTkScrollableFrame(self, width=280)
#         self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

#         ctk.CTkLabel(self.sidebar, text="1. Image Processing", font=("Arial", 16, "bold")).pack(pady=(10, 5))
#         ctk.CTkButton(self.sidebar, text="Upload Image", command=self.load_image).pack(pady=5)
        
#         self.auto_btn = ctk.CTkButton(self.sidebar, text="🧠 Auto-Analyze & Suggest", command=self.auto_suggest, fg_color="#b58b00", hover_color="#8a6a00")
#         self.auto_btn.pack(pady=10)

#         # Inputs Dictionary for easy access
#         self.inputs = {}

#         def add_input(label_text, default_val):
#             ctk.CTkLabel(self.sidebar, text=label_text).pack(anchor="w", padx=10)
#             entry = ctk.CTkEntry(self.sidebar)
#             entry.insert(0, str(default_val))
#             entry.pack(fill="x", padx=10, pady=(0, 10))
#             return entry

#         ctk.CTkLabel(self.sidebar, text="2. Blur Parameters", font=("Arial", 14, "bold")).pack(pady=(10, 5))
#         self.inputs['k_size'] = add_input("Kernel Size (Odd int):", 5)
#         self.inputs['sigma'] = add_input("Sigma (float):", 1.5)

#         ctk.CTkLabel(self.sidebar, text="3. Edge Detection", font=("Arial", 14, "bold")).pack(pady=(10, 5))
#         self.inputs['thresh'] = add_input("Gradient Threshold (int):", 80)

#         ctk.CTkLabel(self.sidebar, text="4. Hough Resolution", font=("Arial", 14, "bold")).pack(pady=(10, 5))
#         self.inputs['rho_res'] = add_input("Rho Resolution (px):", 1)
#         self.inputs['theta_res'] = add_input("Theta Resolution (deg):", 1.0)

#         ctk.CTkLabel(self.sidebar, text="5. Peak Extraction", font=("Arial", 14, "bold")).pack(pady=(10, 5))
#         self.inputs['nhood'] = add_input("Peak Neighborhood (px):", 15)
#         self.inputs['n_lines'] = add_input("Number of Lines to Draw:", 20)
        
#         self.run_btn = ctk.CTkButton(self.sidebar, text="▶ Run Manual Math Pipeline", command=self.process_image, fg_color="green", hover_color="darkgreen", height=40)
#         self.run_btn.pack(pady=20, fill="x", padx=10)

#         # --- Main Display Area ---
#         self.main_container = ctk.CTkFrame(self)
#         self.main_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

#         # Metrics Header
#         self.metrics_frame = ctk.CTkFrame(self.main_container, height=60, fg_color="#1e1e1e")
#         self.metrics_frame.pack(fill="x", pady=(0, 10))
#         self.metric_label = ctk.CTkLabel(self.metrics_frame, text="Upload an image to begin.", font=("Arial", 16))
#         self.metric_label.pack(pady=15)

#         # Graph Area
#         self.graph_frame = ctk.CTkFrame(self.main_container)
#         self.graph_frame.pack(fill="both", expand=True)

#     def load_image(self):
#         file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
#         if file_path:
#             self.img_color = cv2.imread(file_path)
#             max_dim = 500
#             h, w = self.img_color.shape[:2]
#             if max(h, w) > max_dim:
#                 scale = max_dim / max(h, w)
#                 self.img_color = cv2.resize(self.img_color, (int(w * scale), int(h * scale)))
#             self.metric_label.configure(text=f"Image Loaded: {self.img_color.shape[1]}x{self.img_color.shape[0]}. Ready to analyze.")

#     def auto_suggest(self):
#         if self.img_color is None:
#             self.metric_label.configure(text="Error: Upload an image first!")
#             return
            
#         gray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        
#         # 1. Texture/Noise Analysis via Laplacian
#         lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#         if lap_var > 1000: k, sig = 9, 2.5
#         elif lap_var > 300: k, sig = 5, 1.5
#         else: k, sig = 3, 0.8
        
#         # 2. Gradient Distribution for Threshold
#         sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
#         sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
#         mag = np.sqrt(sx**2 + sy**2)
#         thresh = int(np.mean(mag) + np.std(mag))
        
#         # 3. Scale-based neighborhood
#         nhood = max(5, int(min(gray.shape) * 0.03))

#         # Update text boxes
#         updates = {'k_size': k, 'sigma': sig, 'thresh': max(20, min(thresh, 250)), 'nhood': nhood, 'rho_res': 1, 'theta_res': 1.0, 'n_lines': 20}
#         for key, val in updates.items():
#             self.inputs[key].delete(0, 'end')
#             self.inputs[key].insert(0, str(val))
            
#         self.metric_label.configure(text="Parameters Auto-Suggested successfully! Click Run.")

#     def process_image(self):
#         if self.img_color is None:
#             return

#         self.metric_label.configure(text="Calculating heavy matrix math... Please wait.")
#         self.update()

#         try:
#             # Parse inputs
#             k = int(self.inputs['k_size'].get())
#             sig = float(self.inputs['sigma'].get())
#             eth = float(self.inputs['thresh'].get())
#             rho_res = int(self.inputs['rho_res'].get())
#             theta_res = float(self.inputs['theta_res'].get())
#             nhood = int(self.inputs['nhood'].get())
#             nl = int(self.inputs['n_lines'].get())

#             img_gray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY).astype(np.float64)

#             # 1. Blur
#             g_kernel = gaussian_kernel(k, sig)
#             img_blur = convolve2d(img_gray, g_kernel, mode='same', boundary='symm')

#             # 2. Sobel & Mag
#             Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#             Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#             grad_x = convolve2d(img_blur, Kx, mode='same', boundary='symm')
#             grad_y = convolve2d(img_blur, Ky, mode='same', boundary='symm')
#             grad_mag = np.sqrt(grad_x**2 + grad_y**2)
#             edges = (grad_mag > eth).astype(np.uint8)

#             # 3. Polar Accumulator
#             y_idxs, x_idxs = np.nonzero(edges)
#             thetas = np.deg2rad(np.arange(-90.0, 90.0, theta_res))
#             h, w = img_gray.shape
#             diag_len = int(np.ceil(np.sqrt(w**2 + h**2)))
#             rhos = np.arange(-diag_len, diag_len, rho_res)

#             acc_rho_theta = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
#             cos_t = np.cos(thetas)
#             sin_t = np.sin(thetas)

#             for i in range(len(x_idxs)):
#                 rho_vals = x_idxs[i] * cos_t + y_idxs[i] * sin_t
#                 # Find closest index in rhos array
#                 rho_idxs = np.round((rho_vals + diag_len) / rho_res).astype(int)
#                 # Ensure bounds
#                 valid = (rho_idxs >= 0) & (rho_idxs < len(rhos))
#                 for t_idx in range(len(thetas)):
#                     if valid[t_idx]:
#                         acc_rho_theta[rho_idxs[t_idx], t_idx] += 1

#             # 4. Cartesian (m-c) Accumulator
#             ms = np.linspace(-5.0, 5.0, 200) 
#             c_min, c_max = -h * 5, h * 5
#             cs = np.linspace(c_min, c_max, 400)
#             acc_m_c = np.zeros((len(cs), len(ms)), dtype=np.uint64)

#             for i in range(len(x_idxs)):
#                 c_vals = y_idxs[i] - ms * x_idxs[i]
#                 c_idxs_arr = (np.digitize(c_vals, cs) - 1)
#                 valid_mask = (c_idxs_arr >= 0) & (c_idxs_arr < len(cs))
#                 acc_m_c[c_idxs_arr[valid_mask], np.arange(len(ms))[valid_mask]] += 1

#             # 5. Extract Peaks
#             peak_indices, votes = extract_hough_peaks(acc_rho_theta, nl, nhood)
            
#             # Prepare markers and draw lines
#             img_result = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB).copy()
#             marker_polar = []
#             marker_cartesian = []
            
#             for rho_idx, theta_idx in peak_indices:
#                 rho, theta = rhos[rho_idx], thetas[theta_idx]
                
#                 # Markers for Polar
#                 marker_polar.append((np.rad2deg(theta), rho))
                
#                 # Markers for Cartesian (avoiding div by zero for vertical lines)
#                 if abs(np.sin(theta)) > 1e-4:
#                     m = -np.cos(theta) / np.sin(theta)
#                     c = rho / np.sin(theta)
#                     if -5.0 <= m <= 5.0 and c_min <= c <= c_max:
#                         marker_cartesian.append((m, c))

#                 # Draw actual line
#                 a, b = np.cos(theta), np.sin(theta)
#                 x0, y0 = a * rho, b * rho
#                 pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
#                 pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
#                 cv2.line(img_result, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

#             # Update Metrics
#             avg_vote = int(np.mean(votes)) if votes else 0
#             max_vote = votes[0] if votes else 0
#             stats_text = (f"📊 METRICS | Image: {w}x{h} | Edge Pixels: {len(x_idxs):,} | "
#                           f"Lines Detected: {len(votes)} | Max Vote: {max_vote} | Avg Vote: {avg_vote}")
#             self.metric_label.configure(text=stats_text)

#             self.plot_results(img_gray, grad_mag, edges, img_result, 
#                               acc_m_c, ms, cs, acc_rho_theta, thetas, rhos,
#                               marker_polar, marker_cartesian)

#         except Exception as e:
#             self.metric_label.configure(text=f"Error during calculation: {str(e)}")

#     def plot_results(self, img_orig, grad_mag, edges, img_result, 
#                      acc_m_c, ms, cs, acc_rho_theta, thetas, rhos,
#                      marker_polar, marker_cartesian):
#         if self.canvas_widget:
#             self.canvas_widget.destroy()

#         # Create 2x3 Matplotlib Dashboard
#         fig, axs = plt.subplots(2, 3, figsize=(15, 8))
#         fig.patch.set_facecolor('#2b2b2b')
#         for ax in axs.flatten():
#             ax.set_facecolor('#2b2b2b')
#             ax.tick_params(colors='white')
#             ax.title.set_color('white')
#             ax.xaxis.label.set_color('white')
#             ax.yaxis.label.set_color('white')

#         # Row 1
#         axs[0, 0].imshow(img_orig, cmap='gray')
#         axs[0, 0].set_title("1. Original Grayscale")
        
#         axs[0, 1].imshow(grad_mag, cmap='gray')
#         axs[0, 1].set_title("2. Sobel Gradient Magnitude")
        
#         axs[0, 2].imshow(edges, cmap='gray')
#         axs[0, 2].set_title("3. Binary Edges")

#         # Row 2
#         axs[1, 0].imshow(img_result)
#         axs[1, 0].set_title("4. Final Detected Lines")

#         # m-c Plot with Markers
#         axs[1, 1].imshow(np.log1p(acc_m_c), cmap='hot', aspect='auto', extent=[ms.min(), ms.max(), cs.max(), cs.min()])
#         axs[1, 1].set_title("5. m-c Accumulator (Cartesian)")
#         if marker_cartesian:
#             m_vals, c_vals = zip(*marker_cartesian)
#             axs[1, 1].scatter(m_vals, c_vals, color='lime', marker='x', s=60, label='Peaks')
#             axs[1, 1].legend(loc="upper right", facecolor='#1e1e1e', labelcolor='white')

#         # rho-theta Plot with Markers
#         axs[1, 2].imshow(np.log1p(acc_rho_theta), cmap='hot', aspect='auto', extent=[np.rad2deg(thetas).min(), np.rad2deg(thetas).max(), rhos.max(), rhos.min()])
#         axs[1, 2].set_title("6. ρ-θ Accumulator (Polar)")
#         if marker_polar:
#             t_vals, r_vals = zip(*marker_polar)
#             axs[1, 2].scatter(t_vals, r_vals, color='lime', marker='x', s=60, label='Peaks')
#             axs[1, 2].legend(loc="upper right", facecolor='#1e1e1e', labelcolor='white')

#         plt.tight_layout()

#         canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
#         canvas.draw()
#         self.canvas_widget = canvas.get_tk_widget()
#         self.canvas_widget.pack(fill="both", expand=True)

# if __name__ == "__main__":
#     app = AdvancedHoughApp()
#     app.mainloop()