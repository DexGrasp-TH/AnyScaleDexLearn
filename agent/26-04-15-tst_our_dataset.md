# Work

Test training the generative model using our collected dataset in `/data/dataset/AnyScaleGrasp/OurHumanGraspFormat`.


## General Requests

* Write concise and necessary comments to your generated code for better user understanding.

## Run Log

Record what you have done here after every job.

- 2026-04-16: Updated `dexlearn/scripts/check_human_dataloader.py` to extract `dataset_name` from `config.data.grasp_path` by parsing the path segment before `grasp`, and removed the previous default fallback value.
- 2026-04-16: Updated `dexlearn/scripts/check_human_dataloader.py` trimesh visualization to set an initial Z-up friendly camera view (`scene.set_camera(...)`) for better tabletop grasp inspection.
- 2026-04-16: Fixed left-to-right MANO mirroring in `dexlearn/dataset/human_multidex.py` by mirroring axis-angle joint pose vectors (45D MANO pose) instead of directly copying left-hand pose to right-hand pose.

## What I want you to do next



## TODO (in the future)
