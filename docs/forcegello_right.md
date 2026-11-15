# ForceGello 统一数据与右臂 RDP 训练

转换后的 Zarr 为左右臂保存对称的实测状态、主臂状态和动作标签，并保留三路 RGB。
即使只启用右臂遥操作，左臂仍保存真实记录；不按黄色按钮状态筛选，不填充伪造的左臂或触觉数据。
默认网络从统一数据中选择顶部、右腕图像及右臂状态，输出右臂关节命令。

## 动作定义

每个机械臂的命令为 **14 维**，与采集器给从臂的 `pos7/vel7` 参数布局一致：

```text
[q1,q2,q3,q4,q5,q6,gripper, dq1,dq2,dq3,dq4,dq5,dq6,0]
 └──────── pos7 ───────┘  └──────── vel7 ────────┘
```

前六个位置/速度来自主臂实测值，单位 rad、rad/s；第七个位置是手柄的归一化夹爪输入。
夹爪速度目标固定为 0，与采集器一致，不用手柄开度差分伪造速度。

**`side_robot_action[t] = side_leader_pos7_vel7[t+1]`**，不是从臂的下一帧 TCP 或关节状态。
`side_robot_target[t]` 则保存当前帧主臂的同格式命令，便于核对。
全局 `action`、`target` 按 `[left14,right14]` 拼接成 28 维。

`t` 是对齐后的 60 Hz 采样序号。相邻帧必须各自对应左右机器人流中连续的原始记录，
重复取到同一条记录或跨过原始记录时会拆段，避免把重复状态当作新的主臂下一帧。
每段最后一帧只提供上一帧的未来标签，自身不写入 Zarr；不重复填末帧、不跨轨迹取标签。
`action_monotonic_ns` 和 `action_source_monotonic_ns` 保存下一帧对齐时间和左右原始来源时间。

## 数据转换

在 i2rt 的采集环境中运行，复用 YAM 模型与依赖，无需启动机械臂、ROS 或 ImplicitRDP：

```bash
cd /home/kentliukx/ForceGello/i2rt
uv run --extra collection --with 'zarr>=2.16,<3' --with 'numcodecs<0.16' \
  python ../RDP/post_process_forcegello.py \
  --input-dir data/demonstrations \
  --output-dir ../RDP/data/forcegello_zarr
```

输入可以是一条轨迹目录或包含多条轨迹的目录；只处理 `status=complete` 的完整轨迹。
每条轨迹需要 `robot_left.pkl`、`robot_right.pkl` 及三路相机记录，主臂信息必须存在，不能用从臂信息代替。
已有输出不会覆盖；旧的 `forcegello.rdp.right.v1` 数据需要从原始轨迹重新转换。
也可在安装了 MuJoCo、OpenCV、Tyro、Zarr 2 的环境中运行，用 `--robot-xml` 指定包含
`joint1..joint6` 与 `grasp_site` 的模型，避免导入 i2rt。

默认以每条轨迹开始时间为原点建立 60 Hz 时间轴，观测只取不晚于当前时刻的最新状态和图像。
20 Hz 图像可对应多个机器人样本，不插值出新图像。
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
    {side}_robot_target              (N,14)        float32，当前主臂 pos7/vel7
    {side}_robot_action              (N,14)        float32，下一帧主臂 pos7/vel7
    target                          (N,28)        float32，left14 + right14
    action                          (N,28)        float32，left14 + right14
    timestamp                       (N,)          float64，Unix 秒
    monotonic_ns                    (N,)          int64，当前帧对齐时间
    robot_source_monotonic_ns        (N,2)         int64，当前左右记录时间
    action_monotonic_ns              (N,)          int64，下一帧对齐时间
    action_source_monotonic_ns       (N,2)         int64，下一帧左右记录时间
    camera_source_monotonic_ns       (N,3)         int64，原始图像时间
  meta/
    episode_ends                    (E,)          int64，累计末尾位置（不含）
```

`pose9 = [x,y,z,R[:,0],R[:,1]]`，位置单位 m；旋转使用矩阵前两列的 6D 表示。
TCP 位姿和速度表达在各自机械臂的基座系。`tcp_vel` 为 `[vx,vy,vz,wx,wy,wz]`，单位 m/s、rad/s。
从臂关节数组末维是归一化夹爪开度及其变化率；主臂 `leader_joint_vel` 末维则是控制接口约定的 0。
夹爪宽度用归一化开度乘名义行程 `--gripper-max-width-m`（默认 0.095 m）转换。

两列机器人来源时间按 `[left,right]` 排列；图像来源时间按 `[external,left_wrist,right_wrist]` 排列。
根 attributes 保存 schema `forcegello.rdp.bimanual_joint.v2`、坐标系、力估计参数、动作布局和来源。
图像分批解码和写入，转换完成前目录名为 `replay_buffer.zarr.partial`。

## 右臂网络配置

默认 task 的 `shape_meta` 为：

```yaml
obs:
  external_img:          {shape: [3, 240, 320], type: rgb}
  right_wrist_img:       {shape: [3, 240, 320], type: rgb}
  right_robot_joint_pos: {shape: [7], type: low_dim}
  right_robot_joint_vel: {shape: [7], type: low_dim}
  right_robot_tcp_wrench: {shape: [6], type: low_dim}
extended_obs:
  right_robot_tcp_wrench: {shape: [6], type: low_dim}
action:
  key: right_robot_action
  space: joint
  shape: [14]
```

`obs` 只加载网络需要的字段；完整 Zarr 不随网络输入配置改变。
训练左臂时将 obs 中的右臂字段换成左臂，并选择 `left_robot_action`；双臂训练同时选择两侧 obs，并选择全局 `action`、维度 28。
关节命令按每维范围归一化，不使用 TCP 的相对位姿变换；`relative_action=False`。

- AT 编码器输入 `(B,32,14)`，快 GRU 使用 `(B,32,6)` wrench，重建 14 维动作序列。
- 慢策略输入两帧视觉/关节状态历史，低维状态每帧为 `7+7+6=20` 维。
- 视觉历史间隔 1/20 秒，动作和快解码器反馈间隔 1/60 秒。
- 当前卷积 AT 将 32 个动作编码成长度 8、每项 16 维的 latent。
- 解码完整输出 `(B,32,14)`，返回可执行部分为 `(B,27,14)`。

在 RDP 训练环境中：

```bash
cd /home/kentliukx/ForceGello/RDP
DATASET_PATH=data/forcegello_zarr bash train_rdp.sh
```

脚本先用 `train_forcegello_right_at` 训练 AT，再将 checkpoint 传给 `train_forcegello_right_rdp`。
默认 `GPU_ID=0`、`LOGGING_MODE=offline`，结果位于 `data/outputs/forcegello_right_时间/{at,rdp}`。
训练配置不包含 YAM 在线执行接口；部署时应把反归一化后的前七维、后七维分别交给从臂 pos/vel 控制接口。

原 RDP 的 `initial_marker`、`marker_offset` 是 GelSight/McTac 触觉胶面标记点的位置和位移，
`marker_offset_emb` 是其 PCA 特征；此数据中不包含这些触觉字段或伪造的夹爪力。

验证：`python -m pytest tests/test_forcegello_right.py -q`。
测试使用不同的主臂/从臂、左右臂数据检查来源选择、下一帧标签、末帧与缺口边界，
并验证两阶段网络前向、反向及 14 维解码。网络冒烟测试使用较小的 U-Net 隐藏通道数。
