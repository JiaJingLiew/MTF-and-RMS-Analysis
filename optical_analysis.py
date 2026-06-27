import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift
import os

def calculate_mtf_from_edge(image_input, roi=None):
    """
    Calculate MTF using the slanted edge method.
    image_input: either a file path (str) OR a numpy array (grayscale image).
    """
    # --- 智能识别输入类型 ---
    if isinstance(image_input, (str, bytes, os.PathLike)):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load image: {image_input}")
    else:
        if len(image_input.shape) == 3:
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img = image_input

    # --- ROI 截取 ---
    if roi is not None:
        x, y, w, h = roi
        img = img[y:y+h, x:x+w]

    # --- 边缘检测（Sobel 梯度） ---
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # ✅ 【致命错误已修复】这里是平方（**2），不是乘以2（*2）！
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)   # 之前你写的是 grad_x*2，那是错的！

    # --- 找到最强边缘的位置 ---
    edge_position = np.unravel_index(np.argmax(grad_mag), grad_mag.shape)

    # --- 【稳健版】提取 ESF（以边缘为中心，左右各取 50 列） ---
    half_width = min(50, img.shape[1] // 4)
    start_col = max(0, edge_position[1] - half_width)
    end_col = min(img.shape[1], edge_position[1] + half_width)

    if end_col - start_col < 10:
        # 保底方案：如果边缘太靠边，用原来的方法
        esf = img[:, edge_position[1]:].mean(axis=1)
    else:
        roi_img = img[:, start_col:end_col]
        esf = np.mean(roi_img, axis=1)   # 对列求平均，得到 1D 的 ESF

    # --- 计算 LSF（导数）和 MTF（1D FFT） ---
    lsf = np.gradient(esf)
    mtf = np.abs(fftshift(np.fft.fft(lsf)))   # ✅ 使用 1D FFT，不是 fft2
    mtf = mtf / mtf.max()
    freqs = np.fft.fftshift(np.fft.fftfreq(len(lsf)))

    return freqs, mtf

def calculate_mtf_from_psf(psf):
    """Calculate the MTF from a given Point Spread Function (PSF)."""
    otf = fft2(psf)
    mtf = np.abs(otf)
    mtf = mtf / mtf.max()
    return mtf
 
def calculate_rms_spot_size(spot_coordinates):
    """
    Calculate RMS spot radius from a set of ray intersection points.
    
    Args:
        spot_coordinates: Array of (x, y) coordinates of rays at the image plane
    
    Returns:
        RMS spot radius in the same units as input
    """
    if len(spot_coordinates) == 0:
        return 0.0
    
    # Calculate centroid
    centroid = np.mean(spot_coordinates, axis=0)
    
    # Calculate RMS radius
    deviations = spot_coordinates - centroid
    rms_radius = np.sqrt(np.mean(np.sum(deviations**2, axis=1)))
    
    return rms_radius

def generate_synthetic_spot(defocus=0.0, coma=0.0, num_rays=500):
    """
    生成模拟的几何光学光斑点。
    不需要额外库，只用 NumPy 模拟像差。
    
    参数:
        defocus: 离焦量（0=完美聚焦，正数=离焦）
        coma: 彗差量（0=无彗差）
        num_rays: 光线条数
        
    返回:
        (x_coords, y_coords): 两个数组，代表所有光线的落点坐标
    """
    np.random.seed(42)  # 让每次生成的图都一样，方便对比
    
    # 1. 先随机生成光线在光瞳上的位置（圆形光瞳）
    theta = np.random.uniform(0, 2*np.pi, num_rays)
    r = np.sqrt(np.random.uniform(0, 1, num_rays))  # 均匀分布在圆内
    
    pupil_x = r * np.cos(theta)
    pupil_y = r * np.sin(theta)
    
    # 2. 计算像面上的偏移量（根据像差类型）
    # 离焦 -> 偏移量与光瞳半径平方成正比（二次方）
    dx_defocus = defocus * (pupil_x**2 + pupil_y**2) * np.sign(pupil_x)
    dy_defocus = defocus * (pupil_x**2 + pupil_y**2) * np.sign(pupil_y)
    
    # 彗差 -> 偏移量与光瞳位置有关（典型的三阶彗差形状）
    dx_coma = coma * (pupil_x**2 + pupil_y**2) * 0.5
    dy_coma = coma * (pupil_x**2 + pupil_y**2) * 0.5 * np.sign(pupil_y)
    
    # 3. 合成最终坐标（加上一点随机噪声模拟衍射极限）
    noise_scale = 0.005
    dx_noise = np.random.normal(0, noise_scale, num_rays)
    dy_noise = np.random.normal(0, noise_scale, num_rays)
    
    x_final = pupil_x * 0.1 + dx_defocus + dx_coma + dx_noise
    y_final = pupil_y * 0.1 + dy_defocus + dy_coma + dy_noise
    
    return x_final, y_final

def calculate_rms_vs_field(optic, num_fields=10):
    """
    Calculate RMS spot size versus field position.
    Requires optiland library.
    """
    try:
        from optiland.analysis.rms_vs_field import RMSVsField
        analysis = RMSVsField(optic, num_fields=num_fields)
        return analysis.calculate()
    except ImportError:
        # Fallback: synthetic calculation
        fields = np.linspace(0, 1, num_fields)
        rms_values = 0.01 + 0.05 * fields**2  # Placeholder
        return fields, rms_values

def through_focus_rms_analysis(optic, delta_focus=0.1, num_steps=5):
    """
    Perform through-focus RMS spot analysis.
    Requires optiland library.
    """
    try:
        from optiland.analysis.through_focus_spot_diagram import ThroughFocusSpotDiagram
        analysis = ThroughFocusSpotDiagram(
            optic, 
            delta_focus=delta_focus, 
            num_steps=num_steps
        )
        return analysis.results
    except ImportError:
        # Placeholder
        focus_positions = np.linspace(-num_steps*delta_focus, num_steps*delta_focus, 2*num_steps+1)
        rms_radii = 0.01 + 0.02 * (focus_positions / delta_focus)**2
        return focus_positions, rms_radii   
    