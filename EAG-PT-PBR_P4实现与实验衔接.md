# EAG-PT-PBR P4 实现与实验衔接

## 实现状态

本次代码在 `d8c6e6a` 基线上实现 P4 renderer，不需要重新训练 Stage 0、Stage 1 或 Stage 2。

当前 Linux/Windows 代码基线为：

```text
功能基线：2ec8b45
当前 HEAD：bfde8f5
```

`bfde8f5` 是对实验性提交 `0966990` 的显式回退，代码功能等价于 `2ec8b45`。

已完成：

- Disney diffuse/GGX 混合采样；
- power-weighted emitter CDF；
- emissive 大于 `0.1` 的 Gaussian 才进入 emitter CDF；
- 截断到 3 sigma 的二维 Gaussian emitter 采样；
- emitter next-event estimation；
- Gaussian shadow transmittance；
- light/BSDF power-heuristic MIS；
- 第 4 个 bounce 起的 Russian roulette；
- emission/direct/indirect/shadow 分解；
- camera 0/200 原始帧号选择和命名；
- 原灯、关闭 emitter、修改 emitter 颜色/强度三种场景；
- CPU BRDF/PDF/MIS/CDF 单元测试。

## 已完成的 P4 诊断

使用 Stage 2 PLY、camera 0/200、16 SPP/7 bounce 得到：

| 配置 | 平均去噪 PSNR | 平均 LPIPS | 结论 |
|---|---:|---:|---|
| Disney，NEE/MIS/RR 全开 | 27.16 dB | 0.3026 | 当前 P4 基线，数值稳定但质量不足 |
| Disney，NEE 开、MIS/RR 关闭 | 17.84 dB | 0.2231 | NEE 估计与 BSDF 命中 emitter 存在重复/不匹配，仅作诊断 |
| Disney，NEE/MIS 关闭 | 33.88 dB | 0.0629 | 说明 BRDF/sampler 基本可用，回归来自 NEE 路径 |

64 SPP/7 bounce 当前 P4 基线：

```text
camera 0/200 平均 PSNR：26.26 dB
平均 LPIPS：0.3199
平均 FLIP：0.2032
平均耗时：14.99 s/view
decomposition error：约 6e-8
```

分量统计显示 camera 0/200 的首跳 `shadow` 均值约为 `0.045`，而 `direct` 无明显火点；`indirect` 曾出现极端值（16 SPP camera 0 最大约 `505.59`）。因此当前主要问题是 Gaussian 阴影可见性和 BSDF-hit-emitter PDF 配对，而不是 BRDF 本身。

实验性提交 `0966990` 将 shadow ray 两端 margin 改为接近 `0.2`，结果 16 SPP/7 bounce 去噪 PSNR 降至 `5.70 dB`，已由 `bfde8f5` 回退。后续禁止再次采用全局 `0.2` shadow margin。

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

## 下一次实现任务

在 `bfde8f5`/`2ec8b45` 基线上继续，不重新训练材质：

1. 保持 shadow ray 的全局 epsilon 不变，增加 source Gaussian 排除，避免从当前表面发出的阴影射线立即自遮挡。
2. 对被采样的 target emitter 增加 target surfel 排除，不能用统一的大端点 margin 代替。
3. 在 shadow payload 中传递 source/target id，或在 any-hit 阶段按 ray-local 排除这两个 id；必须保留其他 Gaussian 的 transmittance。
4. 重新核对 emitter `Le`、Gaussian opacity、截断 Gaussian area PDF 的乘法关系，确保 NEE 只包含一次发光权重。
5. 对 BSDF 命中 emitter：当前 `trace_forth_with_material` 聚合多个 Gaussian，但 MIS 使用 dominant surfel 的单一 PDF。应改为与实际命中/聚合语义一致的 light PDF，或在验证阶段暂时关闭该 MIS 分支做对照。

每次只改一个因素，固定执行：

```text
16 SPP / 7 bounce / NEE on / MIS on / RR off
```

验收顺序：先看 `shadow mean` 和 `direct mean`，再看 PSNR；不要因为 decomposition error 很小就判定采样正确。

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
