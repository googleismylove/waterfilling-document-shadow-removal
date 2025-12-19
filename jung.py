import cv2
import numpy as np
from numba import njit, prange
import time
from scipy.ndimage import rank_filter
import os
import argparse

# ==================== Numba 安全核心（无损加速）====================

@njit(fastmath=False, cache=True)
def _water_filling_core_numba(h_small, max_iter, eta, exp_decay):
    """使用水填充算法估计光照层"""
    H, W = h_small.shape
    w_curr = np.zeros((H, W), dtype=np.float32)
    w_next = np.zeros((H, W), dtype=np.float32)

    for t in range(max_iter):
        # 计算当前最大值 G_peak = max(w_curr + h_small)
        G_peak = -1e9
        for i in range(H):
            for j in range(W):
                val = w_curr[i, j] + h_small[i, j]
                if val > G_peak:
                    G_peak = val

        # 更新内部像素 (1:-1, 1:-1)
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                center = w_curr[i, j] + h_small[i, j]
                pouring = exp_decay[t] * (G_peak - center)

                total_diff = 0.0
                # 上
                nb = w_curr[i-1, j] + h_small[i-1, j]
                if nb - center < 0:
                    total_diff += nb - center
                # 下
                nb = w_curr[i+1, j] + h_small[i+1, j]
                if nb - center < 0:
                    total_diff += nb - center
                # 左
                nb = w_curr[i, j-1] + h_small[i, j-1]
                if nb - center < 0:
                    total_diff += nb - center
                # 右
                nb = w_curr[i, j+1] + h_small[i, j+1]
                if nb - center < 0:
                    total_diff += nb - center

                del_w = eta * total_diff
                w_new_val = w_curr[i, j] + del_w + pouring
                w_next[i, j] = w_new_val if w_new_val > 0 else 0.0

        # 边界保持不变
        for j in range(W):
            w_next[0, j] = w_curr[0, j]
            w_next[H-1, j] = w_curr[H-1, j]
        for i in range(H):
            w_next[i, 0] = w_curr[i, 0]
            w_next[i, W-1] = w_curr[i, W-1]

        # 交换缓冲区
        w_curr, w_next = w_next, w_curr

    # 最终输出：G_final = w_curr + h_small
    G_final = np.empty((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            G_final[i, j] = w_curr[i, j] + h_small[i, j]
    return G_final


@njit(fastmath=False, cache=True)
def _refine_core_numba(h, orig, max_iter, eta, brightness):
    """基于光照层重构增强图像"""
    H, W = h.shape
    w_curr = np.zeros((H, W), dtype=np.float32)
    w_next = np.zeros((H, W), dtype=np.float32)

    for t in range(max_iter):
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                center = w_curr[i, j] + h[i, j]
                total_diff = (
                    (w_curr[i-1, j] + h[i-1, j] - center) +
                    (w_curr[i+1, j] + h[i+1, j] - center) +
                    (w_curr[i, j-1] + h[i, j-1] - center) +
                    (w_curr[i, j+1] + h[i, j+1] - center)
                )
                del_w = eta * total_diff
                w_new_val = w_curr[i, j] + del_w
                w_next[i, j] = w_new_val if w_new_val > 0 else 0.0

        # 边界保持不变
        for j in range(W):
            w_next[0, j] = w_curr[0, j]
            w_next[H-1, j] = w_curr[H-1, j]
        for i in range(H):
            w_next[i, 0] = w_curr[i, 0]
            w_next[i, W-1] = w_curr[i, W-1]

        w_curr, w_next = w_next, w_curr

    # 重建增强图像
    enhanced = np.empty((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            G_val = w_curr[i, j] + h[i, j]
            if G_val < 1e-6:
                G_val = 1e-6
            enhanced[i, j] = brightness * orig[i, j] / G_val * 255.0
    return enhanced


# ==================== 辅助函数 ====================

def nlm_denoise(image, h=12, template_window_size=7, search_window_size=21):
    """非局部均值去噪"""
    return cv2.fastNlMeansDenoising(image, h=h, templateWindowSize=template_window_size, searchWindowSize=search_window_size)


def adaptive_sharpen(image, amount=1.2):
    """自适应锐化"""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def get_background_color(image, size=50, min_peak_ratio=0.02):
    """估算背景色（用于白边去除）"""
    small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([small], [0], None, [256], [0, 256])
    hist = hist.flatten()
    total = hist.sum()
    if total == 0:
        return 255
    threshold = min_peak_ratio * total
    bg_candidate = 255
    for i in range(255, -1, -1):
        if hist[i] >= threshold:
            bg_candidate = i
            break
    return bg_candidate


def remove_white_edges_based_on_bg_color(enhanced_y, bg_color):
    """基于背景色去除白边"""
    mask = enhanced_y > bg_color
    result = enhanced_y.copy()
    result[mask] = bg_color
    return result


def fast_illumination_estimation(gray, downscale=8):
    """快速光照估计（用于手写模式）"""
    h, w = gray.shape
    small = cv2.resize(gray, (w // downscale, h // downscale), interpolation=cv2.INTER_LINEAR)
    ksize_small = max(15, (small.shape[1] // 5) | 1)
    ksize_small = min(ksize_small, 101)
    blurred_small = cv2.GaussianBlur(small, (ksize_small, ksize_small), sigmaX=10)
    illumination = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return illumination


def fix_contrast_fast_from_image(gray_image):
    """手写专用后处理流程（不保存中间图）"""
    gray = gray_image.copy()

    # 步骤1：自适应阈值生成文本掩膜
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )

    # 跳过形态学清理（保持原始掩膜）
    text_mask = thresh

    # 步骤2：修复背景
    mask_uint8 = text_mask.astype(np.uint8)
    background = cv2.inpaint(gray, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # 步骤3：估计光照
    illumination = fast_illumination_estimation(background)
    illumination = np.maximum(illumination, 1e-6)

    # 步骤4：归一化
    normalized = gray.astype(np.float32) / illumination * 255
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # 步骤5：非锐化掩膜
    blurred = cv2.GaussianBlur(normalized, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(normalized, 1.1, blurred, -0.1, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # 步骤6：仅对文本区域做直方图均衡
    equalized = cv2.equalizeHist(sharpened)
    result = sharpened.copy()
    result[mask_uint8 > 0] = equalized[mask_uint8 > 0]

    return result


# ==================== 主增强函数 ====================
def jung_enhance_paper_correct(image_path, output_path=None, keep_color=False, handwriting=False):
    start_total = time.perf_counter()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")

    print(f"[INFO] 已加载图像: {img.shape[1]}x{img.shape[0]}")

    is_color = len(img.shape) == 3 and img.shape[2] == 3

    if is_color:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        gray_for_jung = y
    else:
        gray_for_jung = img
        keep_color = False

    # === 第一步：Jung 算法增强亮度通道 ===
    t0 = time.perf_counter()
    shading = water_filling_luminance(gray_for_jung)
    t1 = time.perf_counter()
    enhanced_y = refine_and_reconstruct(gray_for_jung, shading)
    t2 = time.perf_counter()
    print(f"[Timing] 水填充: {t1 - t0:.3f}s | 精炼: {t2 - t1:.3f}s")

    # 初始白边去除（仅灰度）
    white_edge_time = 0.0
    if not keep_color:
        t_white_start = time.perf_counter()
        bg_color = get_background_color(enhanced_y)
        enhanced_y = remove_white_edges_based_on_bg_color(enhanced_y, bg_color)
        t_white_end = time.perf_counter()
        white_edge_time = t_white_end - t_white_start
        print(f"[Timing] 初始白边去除: {white_edge_time:.3f}s (背景色={bg_color})")
    else:
        print("[SKIP] 彩色模式跳过初始白边去除")

    # ====== 第一阶段：去噪 + 锐化 ======
    t3 = time.perf_counter()
    enhanced_y = nlm_denoise(enhanced_y, h=12)
    t4 = time.perf_counter()
    print(f"[Timing] 第一阶段 NLM 去噪: {t4 - t3:.3f}s")

    enhanced_y = adaptive_sharpen(enhanced_y, amount=1.2)
    t5 = time.perf_counter()
    print(f"[Timing] 第一阶段锐化: {t5 - t4:.3f}s")

    # ====== 第二阶段：轻量去噪 + 轻量锐化 ======
    t6 = time.perf_counter()
    enhanced_y = nlm_denoise(enhanced_y, h=4)
    t7 = time.perf_counter()
    print(f"[Timing] 第二阶段 NLM 去噪: {t7 - t6:.3f}s")

    enhanced_y = adaptive_sharpen(enhanced_y, amount=1.0)
    t8 = time.perf_counter()
    print(f"[Timing] 第二阶段锐化: {t8 - t7:.3f}s")

    # ====== 手写专用后处理 ======
    handwriting_time = 0.0
    if handwriting and not keep_color:
        t11 = time.perf_counter()
        enhanced_y = fix_contrast_fast_from_image(enhanced_y)
        t12 = time.perf_counter()
        handwriting_time = t12 - t11
        print(f"[Timing] 手写后处理: {handwriting_time:.3f}s")
    else:
        if handwriting and keep_color:
            print("[WARNING] 手写后处理在彩色模式下被跳过")
        else:
            print("[SKIP] 手写后处理未启用")

    # ====== 后处理2：上采样 + 秩滤波 + 对比度增强 ======
    postproc2_time = 0.0
    is_handwriting_mode = handwriting
    if not keep_color:
        t13 = time.perf_counter()
        h_orig, w_orig = enhanced_y.shape
        scale = 3

        # 上采样
        gray_up = cv2.resize(enhanced_y, (w_orig * scale, h_orig * scale), interpolation=cv2.INTER_CUBIC)

        # 秩滤波
        filtered_up = rank_filter(gray_up, rank=2, size=3)

        # 下采样回原始尺寸
        downsampled = cv2.resize(filtered_up, (w_orig, h_orig), interpolation=cv2.INTER_AREA)

        # 非手写模式下增强对比度
        if not is_handwriting_mode:
            enhanced_y = cv2.convertScaleAbs(downsampled, alpha=1.2, beta=20)
            print("[INFO] 应用了最终对比度/亮度增强")
        else:
            enhanced_y = downsampled.astype(np.uint8)
            print("[INFO] 手写模式下跳过最终对比度增强")

        t14 = time.perf_counter()
        postproc2_time = t14 - t13
        print(f"[Timing] 后处理2: {postproc2_time:.3f}s")
    else:
        print("[SKIP] 彩色模式跳过后处理2")

    # 重建彩色图像
    t15 = time.perf_counter()
    if keep_color and is_color:
        enhanced_ycrcb = cv2.merge([enhanced_y, cr, cb])
        result = cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        result = enhanced_y
    t16 = time.perf_counter()
    color_recon_time = t16 - t15

    total_time = time.perf_counter() - start_total
    print("\n" + "="*60)
    print(f"[SUMMARY] 总耗时: {total_time:.3f} 秒")
    print(f"  - 水填充 + 精炼: {t2 - t0:.3f}s")
    if not keep_color:
        print(f"  - 初始白边去除: {white_edge_time:.3f}s")
    print(f"  - 第一阶段去噪/锐化: {t5 - t3:.3f}s")
    print(f"  - 第二阶段去噪/锐化: {t8 - t6:.3f}s")
    if handwriting and not keep_color:
        print(f"  - 手写后处理: {handwriting_time:.3f}s")
    if not keep_color:
        print(f"  - 后处理2: {postproc2_time:.3f}s")
    print(f"  - 彩色重建: {color_recon_time:.3f}s")
    print("="*60)

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"[INFO] 已保存至 {output_path}")
    return result


# ==================== Jung 核心函数 ====================

def water_filling_luminance(gray_img, max_iter=2500, downsample_ratio=0.2, eta=0.2):
    """水填充算法估计光照层"""
    assert len(gray_img.shape) == 2, "输入必须是灰度图"
    h = gray_img.astype(np.float32)
    
    h_small = cv2.resize(h, None, fx=downsample_ratio, fy=downsample_ratio, interpolation=cv2.INTER_LINEAR)
    H, W = h_small.shape
    
    if H < 3 or W < 3:
        G_full = cv2.resize(h_small, (gray_img.shape[1], gray_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        return G_full

    exp_decay = np.exp(-np.arange(max_iter, dtype=np.float32))
    print(f"[Water-Filling] 处理亮度通道 | 尺寸={W}x{H}, η={eta}")
    
    G_final = _water_filling_core_numba(h_small, max_iter, eta, exp_decay)
    G_full = cv2.resize(G_final, (gray_img.shape[1], gray_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return G_full


def refine_and_reconstruct(gray_img, shading_layer, max_iter=100, eta=0.2, brightness=0.85):
    """基于光照层精炼并重建增强图像"""
    h = shading_layer.astype(np.float32)
    orig = gray_img.astype(np.float32)
    H, W = h.shape
    
    if H < 3 or W < 3:
        enhanced = brightness * orig / np.maximum(h, 1e-6) * 255.0
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    print(f"[Refinement] 尺寸={W}x{H}, η={eta}")
    enhanced_float = _refine_core_numba(h, orig, max_iter, eta, brightness)
    return np.clip(enhanced_float, 0, 255).astype(np.uint8)


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jung 文档增强算法，支持手写模式")
    parser.add_argument("input", help="输入图像路径")
    parser.add_argument("output", nargs='?', default="jung_output_correct.jpg", help="输出图像路径")
    parser.add_argument("--color", action="store_true", help="保留彩色（禁用白边去除、手写模式和后处理2）")
    parser.add_argument("--handwriting", action="store_true", help="启用手写专用后处理（仅灰度）")

    args = parser.parse_args()

    result = jung_enhance_paper_correct(
        args.input,
        args.output,
        keep_color=args.color,
        handwriting=args.handwriting
    )

    try:
        orig = cv2.imread(args.input, cv2.IMREAD_COLOR if args.color else cv2.IMREAD_GRAYSCALE)
        cv2.imshow("Original", orig)
        cv2.imshow("Jung Enhanced", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[Warning] 无法显示图像: {e}")