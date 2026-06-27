import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tempfile
import os
import optical_analysis  # 导入你自己写的光学分析模块
import plotly.graph_objects as go

# 导入你的光学分析函数
from optical_analysis import calculate_mtf_from_edge, calculate_rms_spot_size

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="MTF & RMS Imaging Quality Analyzer",
    page_icon="📷",
    layout="wide"
)

st.title("📷 MTF & RMS Spots Imaging Quality Analysis")
st.markdown("---")

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("📁 输入")
    uploaded_file = st.file_uploader(
        "上传一张图片进行 MTF 分析",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
    )
    st.markdown("---")
    st.caption("支持格式：PNG, JPG, JPEG, TIFF, BMP")

# ---------- 主界面：两列布局 ----------
col1, col2 = st.columns(2)

# ---------- 左列：上传图片预览 ----------
with col1:
    st.subheader("📸 上传的图片")
    if uploaded_file is not None:
        # 显示图片
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图片", use_container_width=True)
    else:
        st.info("👈 请先在左侧边栏上传一张图片")

# ---------- 右列：MTF 计算按钮和结果 ----------
with col2:
    st.subheader("📊 MTF 分析")

    # 计算按钮
    if st.button("🚀 计算 MTF", type="primary"):
        # ----- 第1关：检查有没有上传图片 -----
        if uploaded_file is None:
            st.error("⚠️ 请先在左侧边栏上传一张图片！")
        else:
            # ----- 第2关：开始计算 -----
            with st.spinner("⏳ 正在计算 MTF，请稍候..."):
                try:
                    # 重要：从上传的文件对象创建 Image（此时 uploaded_file 肯定不是 None）
                    image = Image.open(uploaded_file)
                    
                    # 转为灰度图 NumPy 数组
                    img_array = np.array(image.convert('L'))
                    
                    # 调用你的光学分析函数
                    freqs, mtf = calculate_mtf_from_edge(img_array)
                    
                    # 绘制 MTF 曲线
                    fig, ax = plt.subplots(figsize=(8, 6))
                    half = len(freqs) // 2
                    ax.plot(freqs[half:], mtf[half:])
                    ax.set_xlabel('Spatial Frequency (cycles/pixel)')
                    ax.set_ylabel('MTF')
                    ax.set_title('Modulation Transfer Function')
                    ax.grid(True)
                    ax.set_ylim(0, 1.1)
                    
                    # 显示图表
                    st.pyplot(fig)
                    
                    # 显示 MTF50（可选）
                    # 找到最接近 0.5 的频率值
                    try:
                        half = len(freqs) // 2
                        mtf_half = mtf[:half][::-1]
                        freq_half = freqs[:half][::-1]
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(freq_half, mtf_half, linewidth=2)
                        ax.set_xlabel('Spatial Frequency (cycles/pixel)')
                        ax.set_ylabel('MTF')
                        ax.set_title('Modulation Transfer Function')
                        ax.grid(True)
                        ax.set_ylim(0, 1.1)

                        st.pyplot(fig)
                        # 从高频往低频找第一个小于0.5的点
                        if mtf_half[0] < 0.5:
                            st.warning("MTF 曲线在零频率就已经低于 0.5，无法计算 MTF50")
                            st.info("Reupload a new image with a proper slanted edge for accurate MTF calculation.")
                        idx = np.where(mtf_half < 0.5)[0]
                        if len(idx) > 0:
                            mtf50 = freq_half[idx[0]]
                            if mtf50 < 0.001:
                                st.warning("MTF50 计算结果过低，可能是图像质量不佳或边缘不明显。")
                            st.metric("MTF50", f"{mtf50:.3f} cy/px")
                        else:
                            st.info("MTF 曲线未降到 0.5 以下")
                    except Exception as e:
                        st.error(f"❌ 计算过程中出现错误：{e}")
                    
                    st.success("✅ MTF 计算完成！")
                    
                except Exception as e:
                    st.error(f"❌ 计算过程中出现错误：{e}")
                    st.code(f"错误详情：{type(e).__name__}: {e}")

# ---------- 底部信息 ----------
st.markdown("---")
# ---------- RMS 光斑分析（新增部分） ----------
st.markdown("---")
st.header("🎯 RMS 光斑分析 (Spot Diagram)")

# 创建两列：左边放控制钮，右边放图表
col3, col4 = st.columns([1, 2])

with col3:
    st.subheader("⚙️ 像差控制")
    
    # 滑块控制离焦和彗差
    defocus_val = st.slider(
        "离焦量 (Defocus)", 
        min_value=-1.0, 
        max_value=1.0, 
        value=0.0, 
        step=0.05,
        help="正值表示焦点靠后，负值表示焦点靠前"
    )
    
    coma_val = st.slider(
        "彗差量 (Coma)", 
        min_value=0.0, 
        max_value=0.5, 
        value=0.0, 
        step=0.02,
        help="模拟镜头倾斜或不对称引起的彗星状光斑"
    )
    
    num_rays = st.slider(
        "光线数量", 
        min_value=100, 
        max_value=2000, 
        value=500, 
        step=100
    )
    
    # 生成按钮
    generate_btn = st.button("🔘 生成光斑图", type="primary", use_container_width=True)

with col4:
    st.subheader("📈 光斑分布图")
    
    if generate_btn or 'last_spot_x' not in st.session_state:
        # 调用生成函数
        x_vals, y_vals = optical_analysis.generate_synthetic_spot(
            defocus=defocus_val,
            coma=coma_val,
            num_rays=num_rays
        )
        
        # 计算 RMS 半径（调用我们已有的函数）
        spots = np.column_stack((x_vals, y_vals))
        rms_radius = optical_analysis.calculate_rms_spot_size(spots)
        centroid = np.mean(spots, axis=0)
        
        # ---- 使用 Plotly 绘制交互式散点图 ----
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # 绘制光线点
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers',
            marker=dict(size=3, color='royalblue', opacity=0.7),
            name='光线交点'
        ))
        
        # 绘制质心（红叉）
        fig.add_trace(go.Scatter(
            x=[centroid[0]], y=[centroid[1]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='x', line_width=2),
            name=f'质心'
        ))
        
        # 绘制 RMS 圆圈
        theta_circle = np.linspace(0, 2*np.pi, 100)
        circle_x = centroid[0] + rms_radius * np.cos(theta_circle)
        circle_y = centroid[1] + rms_radius * np.sin(theta_circle)
        
        fig.add_trace(go.Scatter(
            x=circle_x, y=circle_y,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'RMS 半径 = {rms_radius:.4f}'
        ))
        
        # 设置等比例坐标轴（让圆看起来是圆的）
        fig.update_layout(
            xaxis_title="X 位置 (mm)",
            yaxis_title="Y 位置 (mm)",
            xaxis=dict(scaleanchor="y", scaleratio=1),  # 等比例
            height=500,
            showlegend=True,
            hovermode='closest'
        )
        
        # 显示图表
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示 RMS 数值指标
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("RMS 半径 (微米)", f"{rms_radius * 1000:.2f} µm")
        with col_metric2:
            st.metric("光线数量", f"{num_rays}")
        with col_metric3:
            status = "✅ 聚焦良好" if rms_radius < 0.02 else "⚠️ 像差较大"
            st.metric("聚焦状态", status)
            
    else:
        # 首次加载占位
        st.info("👆 点击左侧的「生成光斑图」按钮来查看光斑分布。")

st.caption("🔬 基于 ISO 12233 斜边法计算的 MTF | 使用 OpenCV + NumPy + SciPy")

