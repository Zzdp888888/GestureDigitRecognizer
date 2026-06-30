# 手势数字识别实训项目

本项目基于《神经网络与深度学习综合实训》指导书中的手势数字识别任务完成。项目保留指导书初始代码结构，在 `model.py`、`train.py`、`test.py`、`inference.py` 的基础上完成模型训练、测试和推理，并扩展了 Web 端图片上传与摄像头实时识别功能。

## 功能概览

- 支持 0-9 手势数字识别。
- 保留指导书原始 CNN 训练、测试、推理流程。
- 引入中国数字手势数据集，并构建分组 train/test 划分。
- 对比增强 CNN、EfficientNet-B0、ResNet34 等模型。
- Web 端支持上传图片、摄像头实时识别、深色/浅色主题切换。
- 默认使用 `GestureDigitRecognizer_model.pt` 手势数字识别模型包。

## 项目结构

```text
gesture_rebuild_from_gitee/
├─ images/                  # 指导书原始数据集及扩展数据清单
├─ models/                  # 当前运行模型，权重文件建议通过 Git LFS 管理
├─ web/                     # Web 前端页面
├─ dataset.py               # 数据集读取
├─ model.py                 # CNN 模型定义
├─ train.py                 # 训练入口
├─ test.py                  # 测试入口
├─ inference.py             # 单张图片推理入口
├─ hand_detector.py         # Web 端 ROI 候选与背景弱化
├─ app_server.py            # Web 服务与推理接口
├─ report/                  # 指导书、报告模板和实训报告
└─ 实验过程归档/             # 实验过程说明、训练脚本和评估脚本归档
```

## 运行环境

测试计算机 Miniconda 的 `CV` 环境：
- Python 3.9.25
- PyTorch 2.8.0+cu126
- TorchVision 0.23.0+cu126
- NumPy 2.0.2
- Pillow 11.3.0
- OpenCV 4.12.0.88
- MediaPipe 0.10.35
- Matplotlib 3.9.4
- scikit-learn 1.6.1
- tqdm 4.67.1


```powershell
cd D:\CV\深度学习\实训\gesture_rebuild_from_gitee
& C:\Users\Alne\AppData\Local\miniconda3\envs\CV\python.exe app_server.py
```

打开 Web 页面：

- <http://127.0.0.1:8000/web/>
- 调试对比模式：<http://127.0.0.1:8000/web/?debug=1>

## 训练与测试

指导书基础流程：

```powershell
& C:\Users\Alne\AppData\Local\miniconda3\envs\CV\python.exe train.py
& C:\Users\Alne\AppData\Local\miniconda3\envs\CV\python.exe test.py
& C:\Users\Alne\AppData\Local\miniconda3\envs\CV\python.exe inference.py
```

扩展模型训练、分组测试和数据集构建脚本保存在 `实验过程归档/` 中，便于复现实验过程和撰写报告。


本项目所使用的数据集来源于开源仓库：[Chinese-number-gestures-recognition](https://github.com/tz28/Chinese-number-gestures-recognition)
数据集版权归原作者所有，使用请遵循原仓库开源协议。

## 当前模型

- 主模型包：`models/GestureDigitRecognizer_model.pt`
- 配置文件：`models/GestureDigitRecognizer_model.json`
- 展示名称：手势数字识别模型
- 识别范围：0-9

模型包内部集成指导书 CNN、中国手势增强 CNN、EfficientNet-B0、ResNet34 及逐数字协同评分规则。Web 端默认只展示最终识别结果；在 `?debug=1` 模式下可以查看各模型对比结果。



## 实验报告资料

- 实训报告：`report/17组神经网络与深度学习实训报告.docx`、`report/17组神经网络与深度学习实训报告.pdf`
- 报告模板与指导书：`report/实训项目报告模板.docx`、`report/实训项目指导书.pdf`



