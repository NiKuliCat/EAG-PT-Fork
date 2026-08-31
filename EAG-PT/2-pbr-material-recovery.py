from __future__ import annotations

import argparse
import pathlib
import time

from libraries.configs import TracerConfig
from libraries.utilities import ExLog, setup_torch_and_random
from libraries.classes import (
    EAGNvsDataset,
    EmissionAwareGaussians,
    LearnableEmissionAwareGaussians,
)


def main(tracer_config: TracerConfig) -> None:
    if not tracer_config.PBR_ENABLED:
        raise ValueError("Run with --PBR_ENABLED true.")
    gaussians = EmissionAwareGaussians.LoadPly(tracer_config.EAG_PLY_PATH)
    dataset = EAGNvsDataset.LoadBlenderTransformsSingle(
        tracer_config=tracer_config,
        transforms_json_path=pathlib.Path(tracer_config.NVS_DATASET_TRANSFORMS_JSON_PATH),
    )
    learnable = LearnableEmissionAwareGaussians(gaussians, dataset)
    learnable.optimizePbrMaterials(tracer_config)


if __name__ == "__main__":
    ExLog("PBR MATERIAL RECOVERY START")
    setup_torch_and_random()
    parser = argparse.ArgumentParser()
    config = TracerConfig(parser=parser)
    config.extract(parser.parse_args())
    config.process()
    start = time.perf_counter()
    main(config)
    elapsed = time.perf_counter() - start
    with open(config.OUTPUT_FOLDER_PATH / "main-duration.py", "w") as f:
        f.write(f"main_duration_seconds={elapsed}\n")
    ExLog("PBR MATERIAL RECOVERY END")

