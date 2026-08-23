# EAG-PT-Fork 项目协作规范

## 1. 项目定位

本仓库是 EAG-PT 的个人 fork，用于：

- 在本地 Windows 环境阅读论文、分析源码和修改实验代码；
- 通过 Git 同步代码、配置和研究笔记；
- 在具备 NVIDIA GPU、CUDA 和 OptiX 的 Linux 环境拉取提交并训练；
- 保存可复现实验所需的参数、日志、指标和结论。

仓库远程地址：`https://github.com/NiKuliCat/EAG-PT-Fork.git`

项目原始论文：`2601.23065v2.pdf`  
项目分析笔记：`EAG-PT_实现框架与Baseline改进分析.md`

除非用户明确要求，不要把本仓库当作通用软件项目重构；修改应优先服务于 EAG-PT 重光照 baseline 的复现、分析和改进。

## 2. 仓库结构

当前仓库根目录为 `EAG-PT-Fork/`：

```text
EAG-PT-Fork/
├── AGENTS.md
├── README.md
├── 2601.23065v2.pdf
├── EAG-PT_实现框架与Baseline改进分析.md
└── EAG-PT/
    ├── 0-radiant-scene-reconstruction.py
    ├── 1-diffuse-material-recovery.py
    ├── editing-and-rendering.py
    ├── light-baking.py
    ├── libraries/
    ├── _scripts/
    ├── _data/       # 数据目录，默认不提交
    ├── _output/     # 训练输出，默认不提交
    └── _docs/
```

`EAG-PT/` 是源码运行目录。执行脚本前必须进入该目录：

```bash
cd EAG-PT
```

所有相对路径（`_data/`、`_output/`、PLY 路径和脚本参数）都以 `EAG-PT/` 为基准，不以 Git 仓库根目录为基准。

## 3. 数据、模型和输出边界

### 3.1 禁止默认提交的内容

以下内容通常体积很大、包含生成结果或属于外部源数据，不应提交到 Git：

- `EAG-PT/_data/` 下的完整数据集、EXR、PNG、ZIP 和原始采集数据；
- `EAG-PT/_output/` 下的训练结果、PLY、EXR、PNG、TSDF、缓存和日志副本；
- `EAG-PT-tracer/` 的编译目录、PTX、CUDA build、`*.egg-info/`；
- Python/Conda 缓存、模型权重、临时下载文件和编辑器缓存；
- 任何密钥、token、私有路径、SSH 配置或机器专属凭据。

如果实验必须依赖某个小型示例文件，应放入 `examples/` 或 `configs/`，并在 README 中说明来源、大小和生成方式。不要为了方便把完整数据集加入仓库。

### 3.2 允许提交的内容

- Python、Shell、CUDA/OptiX 接口和配置代码；
- 小型、可公开分发的测试样例；
- 实验配置文件、命令记录、指标汇总和简短日志；
- 论文、中文分析笔记和方法说明；
- 数据下载说明、目录校验信息和复现脚本。

### 3.3 数据路径规则

代码中数据路径应通过命令行参数或配置传入，不要写死本机绝对路径。推荐：

```bash
python 0-radiant-scene-reconstruction.py \
  --NVS_DATASET_PATH _data/Blender/kitchen \
  --DATASET_IS_SYNTHETIC true
```

Linux 训练机的数据可以放在仓库外部，再通过软链接映射到 `EAG-PT/_data`；不要为适配个人目录修改源码：

```bash
ln -s /data/datasets/EAG-PT EAG-PT/_data
```

## 4. 本地修改到 Linux 训练的标准流程

### 4.1 本地开发前

```bash
git status
git branch --show-current
git remote -v
```

确认当前目录是 `EAG-PT-Fork/`，确认没有误把数据或训练输出加入暂存区。

### 4.2 本地修改和静态检查

修改前先阅读相关入口和配置，至少确认：

- 参数是否位于 `EAG-PT/libraries/configs.py`；
- 数据加载是否位于 `EAG-PT/libraries/classes.py`；
- Stage 0、Stage 1、编辑/路径追踪、light baking 的调用链是否受到影响；
- 新增字段是否需要同步 PLY 读取、保存、激活函数和 tracer 接口。

本地至少执行：

```bash
python -m compileall EAG-PT
git diff --check
git status --short
```

如果本机没有 CUDA/OptiX，不要声称完成了训练验证；应在提交说明中写明只完成了静态检查或 CPU 可执行性检查。

### 4.3 提交代码

提交前检查：

```bash
git diff --stat
git diff -- EAG-PT
git status --short
```

推荐提交粒度：一个提交完成一个逻辑变化，例如：

```text
feat: add roughness parameter to Gaussian material
fix: correct relighted dataset path
exp: add kitchen baseline config
docs: update Linux reproduction steps
```

推荐同步流程：

```bash
git add AGENTS.md README.md EAG-PT/*.py EAG-PT/libraries EAG-PT/_scripts
git diff --cached --check
git commit -m "feat: describe the change"
git push origin main
```

如果当前工作在功能分支，推送分支并在合并前保留清晰的实验基线：

```bash
git push -u origin feature/<short-name>
```

除非用户明确要求，不要使用 `git reset --hard`、强制推送或覆盖他人提交。

### 4.4 Linux 训练机拉取

首次部署：

```bash
git clone https://github.com/NiKuliCat/EAG-PT-Fork.git
cd EAG-PT-Fork
```

已有工作区：

```bash
git status
git fetch origin
git switch main
git pull --ff-only origin main
git rev-parse HEAD
```

训练前必须记录实际 commit：

```bash
git rev-parse HEAD > EAG-PT/_output/code-commit.txt
```

如果 Linux 工作区有本地修改，先保存为独立提交或 `git stash push`，不要直接 `git pull` 覆盖。训练机不应修改源码后悄悄训练；需要修改时回到本地开发流程。

## 5. Linux 实验环境基线

官方 README 给出的参考环境：

```text
Ubuntu 22.04
NVIDIA RTX 4090
Driver 570.153.02
CUDA/nvcc 12.8
OptiX 7.7
GCC/G++ 11.4
Python 3.11
PyTorch 2.9.0 + torchvision 0.24.0, cu128
```

主要 Python 依赖版本见 `EAG-PT/README.md`。独立 tracer 仍需从原仓库安装：

```bash
git clone https://github.com/InternRobotics/EAG-PT-tracer.git
cd EAG-PT-tracer
TORCH_CUDA_ARCH_LIST=8.9 pip install . -v --no-build-isolation
```

RTX 4090 使用 compute capability `8.9`。其他 GPU 必须同步检查并修改 tracer 的 `setup.py` 和 PTX 生成脚本；不能只修改环境变量。

训练前检查：

```bash
nvidia-smi
nvcc --version
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import eag_pt_tracer_optix; print('tracer import ok')"
```

## 6. 实验阶段和输出规范

### Stage 0：辐射场/发光感知重建

输入：`transforms.json`、线性 `Radiance-exr`、`points3d.ply`、法线和发光监督；synthetic 场景可用 depth/albedo。  
默认：30,000 iterations；由 `_scripts/0-and-1-reconstruction.sh` 设置 normal supervision 和 normal consistency。  
输出：`_output/<scene>_0-radiant/iter30000-plys/optimized-2d-gaussians.ply`。

### Stage 1：漫反射材质恢复

输入：Stage 0 PLY 和原始场景多视角 radiance；默认 256 SPP、400 iterations。  
输出：`_output/<scene>_1-diffuse/iter400-plys/optimized-2d-gaussians_iter400.ply`。

### Stage 2：场景编辑与路径追踪

输入：Stage 1 PLY、编辑场景编号和对应 dataset；默认单反弹/路径追踪 1024 SPP，路径追踪 bounce limit 为 7。  
输出：`0-nobounce/`、`1-singlebounce-spp1024/`、`2-pathtracing-spp1024/`，包含 EXR、去噪结果、指标和耗时。

### Stage 3：Light baking

输入：编辑后的 `edited.ply` 和训练视角 path-traced EXR。  
输出：固定几何、仅优化 radiance 的 baked Gaussian，以及可实时 0-bounce 预览结果。

每次实验至少保存：

- 实验名、场景、数据路径和编辑场景编号；
- Git commit hash；
- GPU、CUDA、OptiX、Python 和依赖版本；
- 完整命令行参数；
- 随机种子、Gaussian 数量、SPP、bounce limit 和 iteration；
- PSNR、LPIPS、FLIP、渲染时间、显存和必要的 forward/backward 时间；
- 失败日志和异常说明。

推荐在 `_output/<experiment-name>/` 保存 `config.txt`、`command.sh`、`code-commit.txt` 和 `metrics.json`；这些文本文件可以提交，完整渲染图和 PLY 默认不提交。

## 7. 结果与代码的可比性

比较不同方法时必须固定：

- 同一数据集版本和 train/test split；
- 同一相机位姿和输入分辨率；
- 同一 Gaussian 数量或明确报告数量变化；
- 同一 SPP、bounce limit 和 denoiser；
- 同一颜色空间和指标计算方式；
- 同一编辑场景和光源位置。

禁止只比较原始 capture-time 视图来宣称重光照改进。至少同时报告：

```text
original / capture-time
relighted / edited-time
path tracing
light baking
```

涉及 roughness、metallic、specular 或 emission 的修改，必须同时更新 Stage 1 材质模型和 Stage 2 path tracer，避免训练 BRDF 与最终渲染 BRDF 不一致。

## 8. 代码修改规则

- 优先复用现有 `TracerConfig`、`EmissionAwareGaussians`、`LearnableEmissionAwareGaussians` 和 `UTILITIES_*` 接口。
- 新增 Gaussian 属性时，必须同步：初始化、激活/反激活、PLY 读取、PLY 保存、优化器参数组、tracer 前向接口和反向接口。
- 不要在 `editing-and-rendering.py` 中继续堆积无法复用的场景编号；新增场景应提取为清晰的配置或独立函数。
- 不要把线性 EXR、sRGB PNG、PQ 训练空间混用；每个转换必须在代码和实验说明中写清楚。
- CUDA/OptiX 修改必须提供至少一个小规模 smoke test，检查 import、一次 forward、一次 backward 和 NaN。
- 论文或笔记中的数字必须能追溯到代码参数、日志或论文页码；不确定的内容标记为“未验证”或“合理推断”。
- Windows 本地无法运行 CUDA/OptiX 时，只做静态检查、配置检查和数据格式检查，不改写成假定 CPU 可训练。

## 9. Git 与备份策略

- `main` 保持可拉取、可训练或至少可静态检查的状态。
- 较大的功能改动使用 `feature/<name>` 分支；实验结果稳定后再合并。
- 每次开始训练前先拉取并记录 commit；训练期间不要让代码工作区漂移。
- 训练结果应通过 commit 记录“配置和指标”，而不是提交所有图像和模型文件。
- 重要 checkpoint 放在外部存储、对象存储或 Git LFS；使用前先确认仓库配额和备份策略。
- 提交前执行 `git diff --check`，提交后执行 `git status`，确认没有遗漏或误加入大文件。
- 如果误将数据或模型加入暂存区，先用 `git restore --staged <path>` 取消暂存，不要删除源数据。

## 10. 提交说明模板

每个影响训练的提交建议在 commit body 或配套 Markdown 中说明：

```text
变更：
- 修改了什么模块和参数

原因：
- 要解决什么问题

验证：
- 执行了哪些命令
- 是否实际使用 CUDA/OptiX 训练

实验：
- 数据集/场景
- commit hash
- 主要指标和输出目录

兼容性：
- 是否需要重新生成 PLY
- 是否需要重新安装 tracer
- 是否改变 PLY schema 或命令行参数
```

## 11. 安全和源数据规则

- 不修改或清理外部同步维护的 Zotero 数据目录。
- 不覆盖已有论文笔记，除非用户明确要求重生成。
- 不把本机绝对路径、凭据或隐私信息提交到公共仓库；如需提交配置模板，使用 `*.example` 并替换为相对路径。
- 不在日志、截图或 commit 中泄露 token、内网地址、用户名和机器序列号。
- 删除训练输出前先确认它不是唯一 checkpoint；优先移动到外部备份，而不是递归删除。

## 12. 完成标准

一项代码改动只有在以下条件满足后才算完成：

1. 代码、配置、脚本和文档保持一致；
2. 已执行与风险匹配的静态检查或 Linux smoke test；
3. 实验使用的 commit、命令、数据和环境已记录；
4. 结果和失败情况可追溯；
5. Git 工作区干净，且没有误提交数据、模型、缓存或凭据。

