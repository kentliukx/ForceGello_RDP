# ForceGello 统一数据与右臂 RDP 训练

由 i2rt 转换的 Zarr 对称保存左右臂实测状态、实际提交给从臂的同帧位置命令、同帧 TCP 位移和三路 RGB。
没有提交命令的 idle 帧仍保留实测状态，命令列用当前实测位置占位、位移六维为零并标记为无效。
默认网络使用顶部、右腕图像及右臂状态，输出右臂 TCP 位移和夹爪目标。

## 动作定义

源 Zarr 中每臂的 `robot_command[t]` 是 **7 维**实际提交的从臂位置命令：

```text
[q1,q2,q3,q4,q5,q6,gripper]
```

前六维单位 rad，第七维是归一化夹爪开度。源 Zarr 同时存该关节命令经
`grasp_site` 正运动学得到的 `robot_tcp_command[t]`，以及
`robot_command_valid[t]`。每臂 `robot_tcp_command_delta[t]` 对称记录同帧 TCP 目标
相对同帧实测 TCP 的 `[Δx,Δy,Δz,roll,pitch,yaw,gripper_target]`；idle 时前六维严格为零，
末维为当前夹爪开度。全局 `command` 按 `[left7,right7]` 拼成 14 维。
主臂的关节位置和速度另存为 `leader_joint_pos/vel`，不直接充当训练目标。

右臂训练标签为 **7 维**：

```text
action[t] = [p_cmd(t+50ms) - p_follower_obs(t),
             Euler_xyz(R_cmd(t+50ms) R_follower_obs(t)ᵀ),
             gripper_cmd(t+50ms)]
```

平移单位 m，旋转单位 rad，两者都按右臂基座系表达；最后一维是绝对夹爪目标。
旋转顺序固定为 XYZ 固定轴角，即 `R_delta = Rz(yaw) Ry(pitch) Rx(roll)`；
使用这三个分量整体重建相对旋转，不能逐个相加为世界姿态。
该未来标签由训练加载器按需计算，源 Zarr 的位移字段仍是**同帧命令相对同帧观测**。

`t` 是对齐后的 60 Hz 采样序号。相邻帧必须各自对应左右机器人流中连续的原始记录，
重复取到同一条记录或跨过原始记录时会拆段，避免把重复状态当作新的观测帧。
每段所有有效对齐帧都写入 Zarr；训练加载器会丢弃没有未来标签或未来命令无效的帧，
不重复填末帧、不跨轨迹取标签。`monotonic_ns` 和 `robot_source_monotonic_ns`
分别保存当前对齐时间和左右原始记录时间。

## 数据转换

在 i2rt 的采集环境中运行，复用 YAM 模型与依赖，无需启动机械臂、ROS 或 ImplicitRDP：

```bash
cd /home/noematrix/ForceGello/i2rt
uv run --extra collection --with 'zarr>=2.16,<3' --with 'numcodecs<0.16' \
  python examples/bimanual_data_collection/post_process_forcegello.py \
  --input-dir data/demonstrations
```

输入可以是一条轨迹目录或包含多条轨迹的目录；只处理 `status=complete` 的完整轨迹。
每条轨迹需要 `robot_left.pkl`、`robot_right.pkl` 及三路相机记录。
默认输出到 i2rt 的 `data/forcegello_zarr/replay_buffer.zarr`，已有输出不会覆盖；
旧的 `forcegello.bimanual_joint.v1` 数据需要从原始轨迹重新转换。
也可在安装了 MuJoCo、OpenCV、Tyro、Zarr 2 的环境中运行，用 `--robot-xml` 指定包含
`joint1..joint6` 与 `grasp_site` 的模型，避免导入 i2rt。

默认以每条轨迹开始时间为原点建立 60 Hz 时间轴，观测只取不晚于当前时刻的最新状态和图像。
原始 20 Hz 图像可对应多个机器人样本，不插值出新图像；训练时再按 5 Hz 取样并保持。
重新生成的 60 Hz Zarr 可直接用于训练：加载时按每段的 0、1、2 三种起点分别每 3 帧取一帧，
形成三条 20 Hz 本体时间轴；它们共享原始录像，因此按原始轨迹一起划入训练集或验证集。
图像在每条时间轴上每 4 帧更新一次，模拟 5 Hz 保持。训练时按索引从原始 Zarr 读取，
不会把三相图像复制到内存，也不会修改源 Zarr。
训练动作使用当前从臂 TCP 观测与下一个选中时刻（50 ms 后）的实际从臂命令计算。
无效命令和段尾被丢弃；原始 Zarr 的命令列本身没有未来偏移。
二阶段训练设置 `slow_obs_latency_steps: 1`：慢策略输入是 t 的图像和本体状态，
动作段与快策略本体状态从 t+1 对齐。向量最早在 t+1（50 ms 后）进入快解码，
该次快解码预测的命令在 t+2（100 ms 后）到达。图像和慢策略本体状态仍来自同一时刻；
一阶段 AT 的动作与本体状态保持原有逐帧对齐。新 checkpoint 会保存这个延迟设置，
部署按设置调整动作段起点；旧 checkpoint 沿用原来的零延迟对齐方式。
三相增加了可训练的时间起点，但相邻相位共用图像和相近的动作，不等于三倍独立示范。
机器人/图像最大年龄默认为 0.05/0.10 秒，缺失和过期处拆段，训练窗口不会跨越缺口。

## Jacobian 末端力估计

左右臂都按各自从臂的当前关节位置，每帧重新计算正运动学及 `grasp_site` 的几何 Jacobian。
不使用旧的 `cartesian.tcp_wrench_estimate` 缓存。

```text
tau_residual = 保存的 joint_eff_external[:6]
             = 实测关节力矩 - 最近一次提交的重力补偿项
J = J(q)，6×6，当前机械臂基座系，行序 [线速度,角速度]
F_base = sign × solve(J Jᵀ + damping² I, J tau_residual)
F_tcp  = [Rᵀ F_base[:3], Rᵀ F_base[3:]]
```

`side_robot_tcp_wrench` 为局部 TCP 系的 `[Fx,Fy,Fz,Tx,Ty,Tz]`，单位 N、Nm。
默认 `--damping 0.01 --wrench-sign 1`；估计仍包含摩擦、惯性、跟踪及模型误差，不是六维力传感器实测值。

## 最终 Zarr 格式

`N` 为源 Zarr 的总对齐帧数，`E` 为轨迹段数。下列 `{side}` 字段分别存在完整的 `left`、`right` 两套，形状一致。

```text
replay_buffer.zarr/
  data/
    external_img                     (N,240,320,3) uint8 RGB
    left_wrist_img                   (N,240,320,3) uint8 RGB
    right_wrist_img                  (N,240,320,3) uint8 RGB
    {side}_robot_joint_pos           (N,7)         float32，从臂
    {side}_robot_joint_vel           (N,7)         float32，从臂
    {side}_robot_joint_effort        (N,6)         float32，Nm
    {side}_robot_joint_effort_external(N,6)         float32，Nm
    {side}_robot_tcp_pose            (N,9)         float32
    {side}_robot_tcp_vel             (N,6)         float32
    {side}_robot_tcp_wrench          (N,6)         float32，Jacobian 估计
    {side}_robot_gripper_width       (N,1)         float32，m
    {side}_robot_jacobian            (N,6,6)       float32
    {side}_leader_joint_pos          (N,7)         float32，主臂 + 手柄夹爪
    {side}_leader_joint_vel          (N,7)         float32，末维 0
    {side}_robot_command             (N,7)         float32，同帧提交的从臂位置命令
    {side}_robot_command_valid       (N,)          uint8，1 表示真实命令
    {side}_robot_tcp_command         (N,9)         float32，同帧命令的 grasp_site FK
    {side}_robot_tcp_command_delta   (N,7)         float32，同帧 TCP 位移、XYZ 固定轴角和夹爪目标
    command                         (N,14)        float32，left7 + right7
    timestamp                       (N,)          float64，Unix 秒
    monotonic_ns                    (N,)          int64，当前帧对齐时间
    robot_source_monotonic_ns        (N,2)         int64，当前左右记录时间
    camera_source_monotonic_ns       (N,3)         int64，原始图像时间
  meta/
    episode_ends                    (E,)          int64，累计末尾位置（不含）
```

`pose9 = [x,y,z,R[:,0],R[:,1]]`，位置单位 m；旋转使用矩阵前两列的 6D 表示。
TCP 位姿和速度表达在各自机械臂的基座系。`tcp_vel` 为 `[vx,vy,vz,wx,wy,wz]`，单位 m/s、rad/s。
从臂关节数组末维是归一化夹爪开度及其变化率；主臂 `leader_joint_vel` 末维是控制接口约定的 0。
idle 时 `robot_command` 与当前从臂关节位置相同，`robot_tcp_command_delta[:6]=0`，
`robot_command_valid=0`，不能当成有动作的示范目标。
夹爪宽度用归一化开度乘名义行程 `--gripper-max-width-m`（默认 0.095 m）转换。

两列机器人来源时间按 `[left,right]` 排列；图像来源时间按 `[external,left_wrist,right_wrist]` 排列。
根 attributes 保存 schema `forcegello.bimanual_command.v1`、坐标系、力估计参数、命令布局和来源。
图像分批解码和写入，转换完成前目录名为 `replay_buffer.zarr.partial`。

## 右臂网络配置

默认 task 的 `shape_meta` 为：

```yaml
obs:
  external_img:
    shape: [3, 240, 320]
    type: rgb
    n_obs_steps: 2
    temporal_downsample_ratio: 4
    hold_steps: 4
  right_wrist_img:
    shape: [3, 240, 320]
    type: rgb
    n_obs_steps: 2
    temporal_downsample_ratio: 4
    hold_steps: 4
  right_robot_joint_pos:
    shape: [7]
    type: low_dim
    n_obs_steps: 2
    temporal_downsample_ratio: 1
  right_robot_joint_vel:
    shape: [7]
    type: low_dim
    n_obs_steps: 2
    temporal_downsample_ratio: 1
  right_robot_tcp_pose:
    shape: [9]
    type: low_dim
    n_obs_steps: 2
    temporal_downsample_ratio: 1
  right_robot_tcp_wrench:
    shape: [6]
    type: low_dim
    n_obs_steps: 2
    temporal_downsample_ratio: 1
extended_obs:
  right_robot_joint_pos: {shape: [7], type: low_dim}
  right_robot_joint_vel: {shape: [7], type: low_dim}
  right_robot_tcp_pose: {shape: [9], type: low_dim}
  right_robot_tcp_wrench: {shape: [6], type: low_dim}
action:
  key: right_robot_tcp_delta
  space: tcp_delta
  shape: [7]
```

`obs` 只加载网络需要的字段；完整 Zarr 不随网络输入配置改变。
这批数据的左臂始终 idle，`left_robot_command_valid=0`，不能直接拿来训练左臂控制策略。
7 维 TCP 标签按每维范围归一化；`delta_action=False`、`relative_action=False`，
因为标签已经按当前从臂观测构造成位移。

- 训练与部署均为：动作、关节位置、速度和 wrench **20 Hz**；视觉和慢扩散推理 **5 Hz**。
- 慢策略每次取两帧视觉，间隔 4 个机器人周期（0.2 秒）；本体状态取当前与上一周期，间隔 50 ms。
- 图像在每段轨迹的 5 Hz 时钟上取样保持，不使用未来帧。
- 快 GRU 每个 20 Hz 控制周期接收位置7、速度7、TCP pose9、wrench6，合计 **29 维**反馈。
- Chunk 恢复 **32 步**（20 Hz，1.6 秒的序列）；原 RDP 公式使用 8 步观测前缀，返回 25 步可执行动作。
- AT 编码器输入 `(B,32,7)`，GRU 训练条件 `(B,32,29)`，完整解码 `(B,32,7)`，可执行部分 `(B,25,7)`。
- 卷积 AT 将 32 个动作编码成长度 8、每项 16 维的 latent。
- 默认 task 为 `real_forcegello_right_wrench_at_20fps` / `real_forcegello_right_wrench_ldp_20fps`；
  原 `_60fps` 配置名作为兼容别名指向这两份新配置。

需要重新训练 AT 和 LDP；旧的关节命令 checkpoint 不能用于新的 TCP 动作空间。
源 Zarr 的命令与从臂观测同帧，均为 60 Hz。训练样本每 3 帧取一帧形成 20 Hz，
标签用源帧 `t` 的从臂观测和 `t+3` 的从臂命令（50 ms 后）计算。
不插值观测，也不让网络预测速度。
在 RDP 训练环境中：

```bash
cd /home/noematrix/ForceGello/RDP
DATASET_PATH=../i2rt/data/forcegello_zarr bash train_rdp.sh
```

脚本先用 `train_forcegello_right_at` 训练 AT，再将 checkpoint 传给 `train_forcegello_right_rdp`。
默认 `GPU_ID=0`、`LOGGING_MODE=offline`，结果位于 `data/outputs/forcegello_right_时间/{at,rdp}`。
AT 和 RDP 每隔 20 个 epoch 保留一个 `epoch=...-val_loss=...-deploy.ckpt`，其中 AT 只保存
`model`，RDP 只保存部署使用的 `ema_model`；这些文件不能用来续训。
各阶段的 `checkpoints/latest.ckpt` 仍保存完整训练状态，可用于断点恢复。
YAM 的无 ROS 在线执行入口为 [i2rt/examples/rdp_policy/run.py](../../i2rt/examples/rdp_policy/README.md)。
每个 20 Hz 周期用当前从臂状态和预测位移解 IK，60 Hz 插值关节位置；
关节速度为相邻关节目标差除以 0.05 秒，不由网络预测。慢扩散与实时 GRU 解码分开运行。

原 RDP 的 `initial_marker`、`marker_offset` 是 GelSight/McTac 触觉胶面标记点的位置和位移，
`marker_offset_emb` 是其 PCA 特征；此数据中不包含这些触觉字段或伪造的夹爪力。

验证：`python -m pytest tests/test_forcegello_right.py -q`。
测试使用不同的主臂/从臂、左右臂数据检查来源选择、下一帧标签、末帧与缺口边界，
并验证两阶段网络前向、反向及 7 维解码。网络冒烟测试使用较小的 U-Net 隐藏通道数。
