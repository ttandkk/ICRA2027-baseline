# SmolVLA 训练—推理输入输出一致性记录

## 1. 目的与结论

本文记录 `SmolVLA-FC-80000` checkpoint 在 MotionForge FC001–FC009 评测中的训练—推理一致性审计结果，覆盖图像、机器人状态、语言指令、动作表示、归一化和时间语义。

审计日期：2026-08-25。

结论：当前主链路未发现会造成训推错位的输入输出处理错误。checkpoint 配置、序列化 processor、MotionForge observation/action contract 和 SmolVLA bridge 在 key、shape、dtype、通道顺序、状态排列、动作坐标系及归一化方式上相互一致。

该结论不是对原始训练数据的逐帧证明。训练配置记录的数据目录目前不在本机，因此无法独立复核训练集的实际 MP4 像素、`tasks.jsonl` 和 FPS 元数据；相关内容在第 10 节单独标记为未确认项。

## 2. 审计对象

- 推理 bridge：[`motionforge_smolvla_bridge_client.py`](motionforge_smolvla_bridge_client.py)
- 评测 runner：[`run_fc001_fc009_evaluation.sh`](run_fc001_fc009_evaluation.sh)
- checkpoint：`../../ckpts/MotionforgeGroup/SmolVLA/FC-80000/pretrained_model/`
- MotionForge observation builder：`../../MotionForge/source/motionforge/motionforge/benchmark/runtime/observation_builder.py`
- MotionForge action adapter：`../../MotionForge/source/motionforge/motionforge/benchmark/control/action_adapter.py`
- HDF5 → LeRobot exporter：`../../MotionForge/scripts/data_genertaion/exporters/export_hdf5_to_lerobot.py`
- LeRobot SmolVLA processor：`../../lerobot/src/lerobot/policies/smolvla/processor_smolvla.py`
- LeRobot SmolVLA policy：`../../lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`

checkpoint 训练记录：

- step：`80000`
- dataset repo id：`local/factory_conveyor_level2_seeded`
- 原数据路径：`/projects/hdd/ssd/ICLR2027/dataset/factory_conveyor_level2_seeded`
- policy：`smolvla`
- LeRobot runtime：本次推理日志为 `0.6.2`

## 3. 数据流

```text
训练：
MotionForge camera/state/oracle action
  → HDF5
  → LeRobot Parquet + MP4 + task metadata
  → checkpoint preprocessor
  → SmolVLA

推理：
MotionForge ObservationPacket
  → motionforge_smolvla_bridge_client.py
  → checkpoint preprocessor
  → SmolVLA.predict_action_chunk
  → checkpoint postprocessor
  → MotionForge ActionPacket
  → ActionAdapter
  → Isaac 8D absolute pose action
```

## 4. 总体 contract 对照

| 项目 | 训练侧 | 推理侧 | 结论 |
| --- | --- | --- | --- |
| Overview 图像 | RGB，`(3,240,320)` | RGB uint8 HWC → float32 CHW | 一致 |
| Front 图像 | RGB，`(3,240,320)` | RGB uint8 HWC → float32 CHW | 一致 |
| Wrist 图像 | RGB，`(3,160,160)` | RGB uint8 HWC → float32 CHW | 一致 |
| State | 9D joint position + 1D gripper width | 相同顺序和 10D shape | 一致 |
| Task | `language_instruction` → `task` | 优先 `task`，回退 `language_instruction` | 一致 |
| Action | robot-frame absolute xyz + rot6d + gripper | 相同 10D 表示 | 一致 |
| State normalization | `MEAN_STD` | checkpoint preprocessor | 一致 |
| Action normalization | `MEAN_STD` | checkpoint postprocessor 反归一化 | 一致 |
| 图像 normalization | `IDENTITY`，模型内部转 `[-1,1]` | 相同 policy 代码路径 | 一致 |
| Chunk | 50 steps | 预测 50，发送前 16 | 模型 contract 一致；发送范围按评测协议裁剪 |

## 5. Checkpoint 与 processor

`config.json` 与 `train_config.json["policy"]` 已做完整 JSON 对比，结果完全相同。关键配置如下：

```text
n_obs_steps=1
chunk_size=50
n_action_steps=50
state_dim=10
action_dim=10
resize_imgs_with_padding=(512,512)
tokenizer_max_length=48
normalization_mapping:
  VISUAL=IDENTITY
  STATE=MEAN_STD
  ACTION=MEAN_STD
adapt_to_pi_aloha=false
use_delta_joint_actions_aloha=false
```

推理直接从 checkpoint 加载：

- `policy_preprocessor.json`
- `policy_preprocessor_step_5_normalizer_processor.safetensors`
- `policy_postprocessor.json`
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors`

preprocessor 顺序：

```text
rename observations
→ add batch dimension
→ task append newline
→ tokenize
→ move to device
→ normalize
```

postprocessor 顺序：

```text
unnormalize action
→ move to CPU
```

全部 action 统计张量在 preprocessor 与 postprocessor 中逐元素相同。动态验证中，零归一化动作经过 postprocessor 后与训练 action mean 的最大绝对误差为 `0`。

三路图像的 `count` 元数据在两个 safetensors 中形状分别为 `(1,1,1)` 和 `(1,)`，数值相同。图像 normalization 为 `IDENTITY`，且 postprocessor 只处理 action，因此该元数据形状差异不影响推理行为。

## 6. 图像处理一致性

训练生成器和推理 observation builder 都调用 `motionforge.tools.as_rgb_uint8` 读取以下相机：

| LeRobot key | MotionForge camera | 原始 shape |
| --- | --- | --- |
| `observation.images.overview` | `overview_camera` | `(240,320,3)` |
| `observation.images.front` | `front_camera` | `(240,320,3)` |
| `observation.images.wrist` | `wrist_camera` | `(160,160,3)` |

训练导出器使用 OpenCV 写视频前执行 `RGB2BGR`；LeRobot video decoder 输出 RGB tensor。推理 bridge 直接接收 RGB uint8，并执行：

```text
HWC uint8
→ 去除可选 alpha
→ HWC 转 CHW
→ float32 / 255
```

通道常量测试输入 `R=10, G=20, B=30` 后，processor 输出保持为：

```text
[10/255, 20/255, 30/255]
```

说明没有 RGB/BGR 交换。

processor 保持图像为 `[0,1]`。SmolVLA policy 内部在训练和推理时都使用相同的 `resize_with_pad(..., 512, 512)`，随后执行 `image * 2 - 1` 转到 `[-1,1]`。

本次还旁路只读抓取过一帧真实 FC006 observation：

```text
overview: (240,320,3), uint8, range [3,255]
front:    (240,320,3), uint8, range [2,255]
wrist:    (160,160,3), uint8, range [33,248]
```

## 7. State 与语言处理一致性

训练 exporter：

```text
observation.state = concat(joint_pos, gripper_width)
```

推理 observation builder：

```text
joint_pos = robot.data.joint_pos
gripper_width = panda_finger_joint1 + panda_finger_joint2
observation.state = concat(joint_pos, gripper_width)
```

最终排列为：

```text
[joint_pos_0, ..., joint_pos_8, gripper_width]
```

checkpoint 统计也支持该解释：第 8、9 个 joint 分量范围约为 `[0,0.04]`，第 10 个 gripper width 分量范围约为 `[0,0.08]`。

真实 observation 的 state 为有限 `float32 (10,)`。使用 checkpoint mean/std 手工计算后，与 processor 输出逐元素一致，最大绝对误差为 `0`。

语言处理：

- 训练数据使用 HDF5 `meta/language_instruction` 生成 LeRobot task metadata。
- 推理包同时携带 `language_instruction` 和 `task`，二者由同一个 scenario instruction 生成。
- bridge 会拒绝空字符串并去除首尾空白。
- checkpoint processor 自动追加换行，随后截断/补齐为 48 tokens。

FC006 真实 observation 中抓取到的 task 与当前 scenario 文件完全一致。

## 8. Action 表示与时间语义

训练 action：

```text
[x, y, z,
 rot6d_col0_x, rot6d_col0_y, rot6d_col0_z,
 rot6d_col1_x, rot6d_col1_y, rot6d_col1_z,
 gripper]
```

属性：

```text
action_key=eef_xyz_rot6d_gripper
action_frame=robot
action_representation=ABSOLUTE
```

推理 bridge 使用相同 key、frame 和 representation。MotionForge `ActionAdapter` 对 rot6d 的两列执行 Gram–Schmidt 正交化，再转为 `wxyz` quaternion；gripper 被限制到 `[-1,1]`。

1000 组随机 quaternion 的 encode/decode 往返测试：

```text
position_max_abs_error=0
rotation_matrix_max_abs_error=5.960464477539062e-07
gripper_max_abs_error=0
```

时间语义：

- policy 训练和输出 50 个连续 action steps。
- 统一评测协议为 30 Hz。
- bridge 发送前 16 步，并声明 execution horizon 8。
- MotionForge 将 30 Hz action 插值到 120 Hz control clock。
- `observation_aligned` 模式会根据端到端推理延迟跳过已过期的动作前缀。

注意：在当前 `wall_clock_strict` 模式下，`execution_horizon=8` 是 packet metadata，不是服务端的硬截断上限；服务端使用 latency-aware request/queue replacement。只有 synchronous scheduler 会按该字段硬裁剪。因此不能把当前评测解释为原生 SmolVLA 连续执行全部 50 步，也不能简单解释为每次严格执行 8 步。

## 9. 模型加载一致性

训练配置中的 `load_vlm_weights=true` 表示训练初始化来源。推理 bridge 在构建架构前临时设置：

```text
load_vlm_weights=false
compile_model=false
```

这样可以避免再次下载基础 SmolVLM 权重。随后 bridge 使用：

```text
SmolVLAPolicy.from_pretrained(
    checkpoint,
    local_files_only=True,
    strict=True,
)
```

本地 checkpoint 包含完整 VLM 和 action expert state；严格加载及真实前向均已通过。因此该 override 只改变初始化/下载路径，不改变最终模型参数。

## 10. 已确认项、未确认项与风险

### 已确认

- `config.json` 与训练 policy 配置完全一致。
- checkpoint state/action 统计与运行时 processor 一致。
- 三路相机 key、顺序、shape、dtype 和 RGB 通道一致。
- state 的 10 个分量排列一致。
- task 进入相同 newline/tokenizer 处理链。
- action 的维度、rot6d convention、坐标系及 absolute 语义一致。
- 30 Hz action 到 120 Hz control 的 resampling 路径有现有单元测试覆盖。
- checkpoint 严格加载及真实动作 chunk 前向通过。

### 未确认

- 原始训练数据的实际 `meta/info.json`、`tasks.jsonl`、MP4 和 Parquet 当前不可访问。
- 无法逐帧证明训练视频像素与当前 renderer 输出完全相同。
- 无法仅根据 checkpoint 证明训练数据实际 FPS；30 Hz 依据当前 exporter 默认值、项目协议和推理配置。

### 分布差异，但不是 processor 错位

- 训练数据名称为 `factory_conveyor_level2_seeded`。
- 当前统一评测协议使用 `initial_position_mode=fixed`。
- 二者具有初始位置分布差异，可能影响成功率，但不改变输入输出字段语义。
- GPU 并发导致的 realtime deadline miss 也可能影响成功率，不应误判为 normalization 或 action-format 错误。

## 11. 验证记录

已执行：

```text
checkpoint config/train policy JSON equality: passed
checkpoint processor dynamic state normalization: passed, max error 0
checkpoint action unnormalization: passed, max error 0
RGB channel preservation test: passed
rot6d/quaternion 1000-sample round trip: passed
MotionForge benchmark core/realtime tests: 40 passed
real MotionForge observation read-only capture: passed
```

MotionForge 测试命令：

```bash
cd /project/mohan_ws/MotionForge
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /project/mohan_ws/miniconda3/envs/motionforge/bin/python \
  -m pytest -q -p no:cacheprovider \
  source/motionforge/tests/test_benchmark_core.py \
  source/motionforge/tests/test_benchmark_realtime.py
```

结果：`40 passed in 1.50s`。

LeRobot 环境当前未安装 `pytest`，因此没有直接运行完整的 `tests/processor/test_smolvla_processor.py`；本次使用 checkpoint 自带 processor 完成了等价的关键动态路径验证，没有为审计额外安装依赖。

## 12. 后续加固建议

如果需要将人工审计固化为每次启动的强制检查，可在 runner/bridge 中增加：

1. `config.json` 与 `train_config.policy` 一致性检查；
2. state/action 统计 key、shape 和有限值检查；
3. pre/post action 统计逐元素一致性检查；
4. RGB、state、task 和 action contract 的轻量 smoke test；
5. 如果重新挂载训练数据，校验 `meta/info.json`、`tasks.jsonl`、视频 FPS 和随机抽帧像素。

上述建议尚未实现；本文只记录当前审计结果。
