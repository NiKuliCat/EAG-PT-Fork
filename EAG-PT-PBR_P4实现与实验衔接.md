# EAG-PT-PBR P4 实现与实验衔接

## 实现状态

本次代码在 `d8c6e6a` 基线上实现 P4 renderer，不需要重新训练 Stage 0、Stage 1 或 Stage 2。

已完成：

- Disney diffuse/GGX 混合采样；
- power-weighted emitter CDF；
- 截断到 3 sigma 的二维 Gaussian emitter 采样；
- emitter next-event estimation；
- Gaussian shadow transmittance；
- light/BSDF power-heuristic MIS；
- 第 4 个 bounce 起的 Russian roulette；
- emission/direct/indirect/shadow 分解；
- camera 0/200 原始帧号选择和命名；
- 原灯、关闭 emitter、修改 emitter 颜色/强度三种场景；
- CPU BRDF/PDF/MIS/CDF 单元测试。

Windows 仅完成 Python 测试和静态检查。OptiX/CUDA 编译与数值结果必须在 Linux 验证。

## 固定输入

```text
dataset:
data/dataset-kitchen

Stage 2 PLY:
_output/pbr-stage2-full/optimized-2d-gaussians_pbr_arm.ply
```

Linux 上已修正的 `data/dataset-kitchen/transforms.json` 不要被原始文件覆盖。

## Linux 重新编译

从仓库根目录执行：

```bash
conda activate EAG-PT

python -m unittest discover -s EAG-PT/tests -v

cd EAG-PT-tracer
TORCH_CUDA_ARCH_LIST=8.9 \
  pip install . -v --no-build-isolation --force-reinstall --no-deps

cd ../EAG-PT
python - <<'PY'
import eag_pt_tracer_optix

renderer = eag_pt_tracer_optix.SampleRenderer()
for name in ("nobounce", "materialpass", "singlebounce", "pathtracing"):
    print(name, hasattr(renderer, name))
PY
```

四项均应为 `True`。

## 分级验证

以下命令均从 `EAG-PT/` 目录执行。参数依次为：dataset、PLY、SPP、bounce、场景编号、输出名。

### 1 SPP / 1 bounce

```bash
bash _scripts/pbr-p4-validation.sh \
  data/dataset-kitchen \
  _output/pbr-stage2-full/optimized-2d-gaussians_pbr_arm.ply \
  1 1 0 pbr-p4-smoke
```

检查两个原始帧的 total、direct、indirect、emission、shadow、BaseColor、Roughness、Metallic 均已输出。

### 16 SPP / 2 bounce

```bash
bash _scripts/pbr-p4-validation.sh \
  data/dataset-kitchen \
  _output/pbr-stage2-full/optimized-2d-gaussians_pbr_arm.ply \
  16 2 0 pbr-p4-medium
```

`_records.py` 中每个 `decomposition_errors` 必须小于 `1e-4`。

### 64 SPP / 7 bounce

```bash
bash _scripts/pbr-p4-validation.sh \
  data/dataset-kitchen \
  _output/pbr-stage2-full/optimized-2d-gaussians_pbr_arm.ply \
  64 7 0 pbr-p4-bounce7
```

与 P3 camera 0 基线对照：noisy PSNR 14.90 dB、denoised PSNR 36.22 dB、15.01 秒。

### emitter 场景

场景 `900` 仅将 `emissives` 置零；场景 `901` 仅修改 emitter radiance，不修改 PBR 材质。

```bash
bash _scripts/pbr-p4-validation.sh \
  data/dataset-kitchen \
  _output/pbr-stage2-full/optimized-2d-gaussians_pbr_arm.ply \
  16 2 900 pbr-p4-emitter-off

bash _scripts/pbr-p4-validation.sh \
  data/dataset-kitchen \
  _output/pbr-stage2-full/optimized-2d-gaussians_pbr_arm.ply \
  16 2 901 pbr-p4-emitter-warm
```

关闭 emitter 后，无环境光场景应接近黑色；三个 material pass 在场景 0/900/901 之间应逐像素一致。

### 1024 SPP / 7 bounce

仅在 64 SPP 验收后执行：

```bash
bash _scripts/pbr-p4-validation.sh \
  data/dataset-kitchen \
  _output/pbr-stage2-full/optimized-2d-gaussians_pbr_arm.ply \
  1024 7 0 pbr-p4-final
```

## 验收条件

- 所有输出无 NaN/Inf；
- `decomposition_errors < 1e-4`；
- shadow 全部位于 `[0,1]`；
- 64 SPP camera 0 noisy PSNR 至少达到 16.90 dB；
- 64 SPP camera 0 denoised PSNR不低于 35.7 dB；
- 64 SPP / 7 bounce 单视角耗时不超过 37.5 秒；
- 1024 SPP camera 0/200 平均 PSNR 不低于 35.78 dB；
- emitter 修改不改变 BaseColor、Roughness、Metallic pass；
- `nobounce` 回归结果与 P3 基线一致。

P4 通过后再实施材质邻域平滑、singlebounce roughness/metallic backward 和 PBR render loss。
