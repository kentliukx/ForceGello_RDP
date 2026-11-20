# ForceGello 统一数据与右臂 RDP 训练

由 i2rt 转换的 Zarr 为左右臂保存对称的实测状态、主臂状态和同帧命令，并保留三路 RGB。
即使只启用右臂遥操作，左臂仍保存真实记录；不按黄色按钮状态筛选，不填充伪造的左臂或触觉数据。
默认网络从统一数据中选择顶部、右腕图像及右臂状态，输出右臂关节命令。

## 动作定义

每个机械臂的网络动作为 **7 维绝对目标位置**：

```text
[q1,q2,q3,q4,q5,q6,gripper]
```

前六个目标角来自主臂，单位 rad；第七维是手柄的归一化夹爪输入。
部署时六个关节的速度目标由相邻 20 Hz 位置目标差分得到，单位 rad/s；夹爪速度固定为 0。

**新生成的 Zarr 中 `side_robot_command[t] = side_leader_pos7[t]`**，
与从臂观测使用同一条对齐后的 60 Hz 记录；不是从臂的 TCP 或关节状态。
预测未来的动作标签由训练加载器按训练频率取样。
全局 `command` 按 `[left7,right7]` 拼接成 14 维。

`t` 是对齐后的 60 Hz 采样序号。相邻帧必须各自对应左右机器人流中连续的原始记录，
重复取到同一条记录或跨过原始记录时会拆段，避免把重复状态当作新的观测帧。
每段所有有效对齐帧都写入 Zarr；训练加载器会丢弃没有未来标签的段尾帧，
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
每条轨迹需要 `robot_left.pkl`、`robot_right.pkl` 及三路相机记录，主臂信息必须存在，不能用从臂信息代替。
默认输出到 i2rt 的 `data/forcegello_zarr/replay_buffer.zarr`，已有输出不会覆盖；
当前已有的 `forcegello.bimanual_command.v1` Zarr 的每臂命令已是 7 维提交给从臂的位置目标，可直接训练本配置。
如果使用旧的 `forcegello.bimanual_joint.v1` 14 维命令 Zarr，加载器只取前七维位置目标。
重新转换时输出 `forcegello.bimanual_joint.v2`，不会覆盖已有目录。
也可在安装了 MuJoCo、OpenCV、Tyro、Zarr 2 的环境中运行，用 `--robot-xml` 指定包含
`joint1..joint6` 与 `grasp_site` 的模型，避免导入 i2rt。

默认以每条轨迹开始时间为原点建立 60 Hz 时间轴，观测只取不晚于当前时刻的最新状态和图像。
原始 20 Hz 图像可对应多个机器人样本，不插值出新图像；训练时再按 5 Hz 取样并保持。
重新生成的 60 Hz Zarr 可直接用于训练：加载时按每段的 0、1、2 三种起点分别每 3 帧取一帧，
形成三条 20 Hz 本体时间轴；它们共享原始录像，因此按原始轨迹一起划入训练集或验证集。
图像在每条时间轴上每 4 帧更新一次，模拟 5 Hz 保持。训练时按索引从原始 Zarr 读取，
不会把三相图像复制到内存，也不会修改源 Zarr。
训练动作从原始 `command` 取下一个选中时刻（50 ms 后）的主臂命令；
每段没有有效下一选中帧的末尾被丢弃。原始 Zarr 的命令列本身没有未来偏移。
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

`N` 为总训练帧数，`E` 为轨迹段数。下列 `{side}` 字段分别存在完整的 `left`、`right` 两套，形状一致。

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
    {side}_robot_command             (N,7)         float32，当前主臂 pos7
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
从臂关节数组末维是归一化夹爪开度及其变化率；主臂 `leader_joint_vel` 末维则是控制接口约定的 0。
夹爪宽度用归一化开度乘名义行程 `--gripper-max-width-m`（默认 0.095 m）转换。

两列机器人来源时间按 `[left,right]` 排列；图像来源时间按 `[external,left_wrist,right_wrist]` 排列。
根 attributes 保存 schema `forcegello.bimanual_joint.v2`、坐标系、力估计参数、命令布局和来源。
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
  right_robot_tcp_wrench:
    shape: [6]
    type: low_dim
    n_obs_steps: 2
    temporal_downsample_ratio: 1
extended_obs:
  right_robot_joint_pos: {shape: [7], type: low_dim}
  right_robot_joint_vel: {shape: [7], type: low_dim}
action:
  key: right_robot_command
  space: joint
  shape: [7]
```

`obs` 只加载网络需要的字段；完整 Zarr 不随网络输入配置改变。
训练左臂时将 obs 中的右臂字段换成左臂，并选择 `left_robot_command`；双臂训练同时选择两侧 obs，并选择全局 `command`、维度 14。
关节命令按每维范围归一化，不使用 TCP 的相对位姿变换；`relative_action=False`。

- 训练与部署均为：动作、关节位置、速度和 wrench **20 Hz**；视觉和慢扩散推理 **5 Hz**。
- 慢策略每次取两帧视觉，间隔 4 个机器人周期（0.2 秒）；本体状态取当前与上一周期，间隔 50 ms。
- 图像在每段轨迹的 5 Hz 时钟上取样保持，不使用未来帧。
- 快 GRU 每个 20 Hz 控制周期接收位置7、速度7，合计 **14 维**反馈；慢策略仍使用 wrench。
- Chunk 恢复 **32 步**（20 Hz，1.6 秒的序列）；原 RDP 公式使用 8 步观测前缀，返回 25 步可执行动作。
- AT 编码器输入 `(B,32,7)`，GRU 训练条件 `(B,32,14)`，完整解码 `(B,32,7)`，可执行部分 `(B,25,7)`。
- 卷积 AT 将 32 个动作编码成长度 8、每项 16 维的 latent。
- 默认 task 为 `real_forcegello_right_wrench_at_20fps` / `real_forcegello_right_wrench_ldp_20fps`；
  原 `_60fps` 配置名作为兼容别名指向这两份新配置。

需要重新训练 AT 和 LDP；旧 60 Hz checkpoint 的时间定义与网络序列长度不适用于当前部署默认值。
源 Zarr 的 `command[t] = 主臂[t]`，`t` 为 60 Hz；训练样本每 3 帧取一帧形成 20 Hz，
标签取该样本之后的第 3 个源帧，即 `action_train[t] = command_zarr[t+3]`（50 ms 后）。
观测与主臂动作均按实测帧取值，不插值；主臂速度仍保存在独立的诊断字段中，不作为动作标签。
在 RDP 训练环境中：

```bash
cd /home/noematrix/ForceGello/RDP
DATASET_PATH=../i2rt/data/forcegello_zarr bash train_rdp.sh
```

脚本先用 `train_forcegello_right_at` 训练 AT，再将 checkpoint 传给 `train_forcegello_right_rdp`。
默认 `GPU_ID=0`、`LOGGING_MODE=offline`，结果位于 `data/outputs/forcegello_right_时间/{at,rdp}`。
YAM 的无 ROS 在线执行入口为 [i2rt/examples/rdp_policy/run.py](../../i2rt/examples/rdp_policy/README.md)，
反归一化后的七维作为位置目标；部署端以相邻目标位置差分生成前六个关节的速度目标，夹爪速度为 0。

原 RDP 的 `initial_marker`、`marker_offset` 是 GelSight/McTac 触觉胶面标记点的位置和位移，
`marker_offset_emb` 是其 PCA 特征；此数据中不包含这些触觉字段或伪造的夹爪力。

验证：`python -m pytest tests/test_forcegello_right.py -q`。
测试使用不同的主臂/从臂、左右臂数据检查来源选择、下一帧标签、末帧与缺口边界，
并验证两阶段网络前向、反向及 7 维解码。网络冒烟测试使用较小的 U-Net 隐藏通道数。
