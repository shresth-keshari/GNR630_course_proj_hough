# Hough Transform Suite — Satellite Optimized

A Python-based, interactive GUI application for detecting linear features (like roads) in high-resolution satellite imagery. This project implements the standard Hough Transform from scratch using pure NumPy arrays, avoiding built-in OpenCV detection functions to remain rule-compliant for academic/technical constraints.

It includes implementations for both the Slope-Intercept (`m-b`) and Polar (`ρ-θ`) parameter spaces, along with custom logic to draw **finite line segments** rather than infinite lines across the image boundaries.

---

## 📸 Visual Showcase

*(Replace the placeholder paths below with the actual paths to your screenshots)*

![Hough Transform GUI - Main Interface](docs/images/gui_main_placeholder.png)
*The main CustomTkinter interface with adjustable parameters for pre-processing, Hough spaces, and finite line rendering.*

**Test Cases:**

| Complex Satellite Imagery | Standard Edge Detection |
| :---: | :---: |
| ![Satellite Test](docs/images/test_satellite_placeholder.png) | ![Simple Test](docs/images/test_simple_placeholder.png) |
| *Aggressive NMS and auto-suggested parameters picking up fragmented colony roads.* | *Clean detection of distinct linear features on standard test images.* |

---

## 🚀 How to Run

There are two ways to run this application. If you just want to use the tool without messing with Python environments, use the executable.

### Method 1: Standalone Executable (Recommended)
You do not need to install Python or any dependencies to use this method. 

1. Open the main project folder.
2. Locate and double-click the **`satellite_hough_gui_v6.exe`** file.
3. The GUI will launch automatically.

### Method 2: Running from Source (Python Virtual Environment)
If the executable doesn't work, or if you want to modify the code, you can run the raw Python script.

1. **Activate your virtual environment:**
   * **Windows (PowerShell):** `.\env311\Scripts\Activate.ps1`
   * **Linux/Mac:** `source env311/bin/activate`
2. **Install dependencies** (if not already installed):
   ```bash
   pip install numpy opencv-python matplotlib customtkinter



Code Explanation & PipelineThe pipeline is structured into primitive mathematical operations to ensure full control over the arrays and accumulators.

1. Preprocessing & Gradients (Sections 0–2)Custom Convolution: A pure NumPy convolve2d function is used alongside a custom-generated Gaussian kernel to smooth out image noise.Sobel Operators: Horizontal and vertical gradients ($G_x$, $G_y$) are computed manually to find edge magnitudes.Thresholding: Converts the continuous gradient magnitudes into a strict binary edge map.

2. The Hough Transforms (Sections 3a & 3b)The suite calculates accumulator arrays for two different parameter spaces:Slope-Intercept (m-b): Iterates through possible slopes and intercepts. (Limited by a maximum slope parameter m_max to prevent near-vertical asymptote issues).Polar (ρ-θ): The standard approach, calculating the perpendicular distance from the origin (ρ) for angles 0 to 180 degrees (θ).

3. Finite Line Rendering LogicInstead of drawing lines that start and end at the extreme edges of the image, this tool implements custom grouping logic to draw realistic, finite line segments (crucial for accurate road tracking).For both transforms, the code:Validates edge pixels that fall within a strict tolerance distance of the mathematical line.Sorts these pixels along the direction of the line.Checks the distance between adjacent pixels. If the gap exceeds the max_gap parameter, the line is "broken" into separate segments.Only draws segments that meet the min_length threshold, effectively filtering out noise and keeping distinct road networks isolated.

4. Peak Extraction (Section 4)A custom 2D Non-Maximum Suppression (NMS) algorithm. It finds the highest peak in the accumulator, records it, and zeroes out a surrounding neighborhood (nhood) to prevent clustering multiple identical lines on the exact same road edge.

5. Texture-Aware Auto-Suggest (Section 5)A smart parameter initialization feature. By calculating the local variance of the grayscale image, the system guesses if the uploaded image is a highly textured satellite shot (where variance is huge due to buildings, trees, and roads) or a simple, clean test image.If satellite imagery is detected, it automatically cranks up the blur, increases the allowable peaks (up to 150), and lowers the threshold ratio to pick up weaker, tiny colony roads.

6. Graphical Interface (Section 7)Built entirely in CustomTkinter and matplotlib backend integrations, providing an intuitive dark-mode interface to adjust parameters dynamically, render accumulator heatmaps, and visualize the final superimposed segments.