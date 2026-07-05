# MTF & RMS Optical Image Quality Analyzer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**An interactive web-based tool for lens sharpness evaluation (MTF) and optical aberration simulation (RMS spot diagrams).** Implements the ISO 12233:2014 Slanted-Edge method with a pure NumPy physics engine—no proprietary software required.

## ✨ Key Features

| Module | Description |
|--------|-------------|
| **📷 MTF Analysis** | Upload any image with a sharp edge, calculate the Modulation Transfer Function curve, and extract the **MTF50** metric (lens sharpness indicator) in one click. |
| **⚛️ RMS Spot Simulation** | Simulate geometric aberrations (defocus, coma) via ray tracing. Generate dynamic spot diagrams and compute the RMS radius to evaluate focus quality. |
| **🛡️ Robust Algorithm** | Built-in boundary clamping and fallback mechanisms ensure stable operation even with edges near the image border. |
| **📊 Interactive Visuals** | Plotly-powered zoomable spot diagrams and Matplotlib-based professional MTF curves. |

## 🛠️ Tech Stack

- **Frontend**: Streamlit (no HTML/CSS required)
- **Math Engine**: NumPy, SciPy (FFT, gradients)
- **Image Processing**: OpenCV, Pillow
- **Visualization**: Matplotlib, Plotly

## 📂 Project Structure
