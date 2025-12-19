

```markdown
[English](README.md) | 简体中文

# 基于 Jung 注水法（Water-Filling）的文档图像增强工具

本脚本实现了基于 Jung 注水法（Water-Filling）的文档图像增强算法，支持普通打印文档与手写笔记两种模式，并可选择保留原始色彩。整个流程完全在 CPU 上运行，适用于 Windows 笔记本等无 GPU 环境。

---

## ✅ 环境要求

- **Python 版本**：3.8 及以上（已在 Python 3.12 下验证通过）
- **依赖库**（可通过 pip 安装）：
  ```bash
  pip install opencv-python numpy scipy numba
  ```

> 💡 注意：`numba` 首次运行时会编译 JIT 函数，可能稍慢；后续调用将显著加速。

---

## 📌 基本用法

```bash
# 增强一张灰度/彩色文档图（默认输出为 jung_output_correct.jpg）
python jung_enhance.py input.jpg

# 指定输出路径
python jung_enhance.py input.jpg output_enhanced.jpg

# 保留原始颜色（适用于彩色扫描件或照片）
python jung_enhance.py input.jpg --color

# 启用手写专用后处理（仅对灰度图有效，自动忽略 --color）
python jung_enhance.py input_handwriting.jpg --handwriting

# 同时指定输出并启用手写模式
python jung_enhance.py note.jpg enhanced_note.jpg --handwriting
```

---

## ⚙️ 默认参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 输出路径 | `jung_output_correct.jpg` | 若未指定第二位置参数，则使用此默认文件名 |
| `--color` | 禁用（默认处理为灰度） | 启用后保留 YCrCb 色彩空间中的 Cr/Cb 通道，输出彩色图像 |
| `--handwriting` | 禁用 | 启用后激活六步手写优化流程（自适应掩膜、背景修复、局部均衡等） |
| 水填充迭代次数 | 2500 | 在降采样图像上执行，平衡精度与速度 |
| 精炼迭代次数 | 100 | 用于光照层后优化 |
| 降采样比例 | 0.2（即 1/5） | 加速核心计算，同时保留全局光照结构 |
| 亮度缩放因子 | 0.85 | 控制最终图像明暗程度，避免过曝 |

> 📝 所有上述参数目前为硬编码，如需调整，可直接修改 `water_filling_luminance()` 和 `refine_and_reconstruct()` 函数中的默认值。

---

## ⚠️ 注意事项

- 若同时使用 `--color` 和 `--handwriting`，手写后处理将被跳过（因该模块仅支持灰度图像）。
- 输入图像建议为**手机拍摄的文档照片**（含阴影、不均匀光照），纯黑底白字扫描件无需增强。
- 处理时间通常在 **1.5–2.5 秒/图**（取决于分辨率和是否启用后处理），在普通 Windows 笔记本上即可流畅运行。
- 运行成功后，程序会自动保存结果，并尝试弹出原图与增强图供对比（若环境支持 GUI，如本地 Windows 桌面）。

---
仓库的example文件夹提供测试用例用于测试本算法。
> © 本文基于 CSDN 博主「小天在线学理工」原创内容整理  
> 原文链接：https://blog.csdn.net/sillybsillyb/article/details/156002897  
> 遵循 CC 4.0 BY-SA 版权协议
```
