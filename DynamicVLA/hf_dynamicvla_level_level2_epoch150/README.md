---
license: other
license_name: slab-license
license_link: LICENSE
datasets:
- hzxie/DOM
base_model:
- HuggingFaceTB/SmolLM2-360M
pipeline_tag: robotics
tags:
- robotics
- lerobot
- dynamicvla
---



# Model Card for DynamicVLA

[DynamicVLA](https://arxiv.org/abs/2601.22153) is a vision-language-action model for dynamic object manipulation. 
It is designed to handle dynamic scenes that require fast perception, temporal anticipation, and continuous control.

This model is trained and evaluated using the official [DynamicVLA codebase](https://github.com/hzxie/DynamicVLA). 
For full setup, training, and benchmarking instructions, please refer to the repository README.

* * *

## How to Get Started with the Model

For a complete walkthrough, see the official DynamicVLA repository. Below is the short version for training and running inference/evaluation.

### Train from scratch

From the `PROJECT_ROOT/dynamic-vla` directory, run:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --standalone run.py \
  -c configs/dynamicvla.yaml \
  -d hzxie/DOM
```

### Evaluate the policy / run inference

```bash
# 1. start evaluation server
python3 simulations/evaluate.py \
  --scene_dir ../scenes \
  --output_dir ../output/evaluation \
  --env_cfg ../test-envs.txt \
  --enable_cameras --headless -n 20 --save

# 2. run policy inference
python3 scripts/inference.py \
  -p /path/to/vla-checkpoint \
  -r euler -d -s
```
