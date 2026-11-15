import json
import pickle
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
import pytest
import torch
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from post_process_forcegello import (
    CAMERAS,
    I2RT_ROOT,
    ArmKinematics,
    leader_target,
    load_model,
    read_robot_stream,
)
from post_process_forcegello import (
    main as convert,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "reactive_diffusion_policy/config"
OmegaConf.register_new_resolver("eval", eval, replace=True)


@pytest.fixture
def robot_xml() -> Path:
    return load_model(
        {"settings": {"arm": "yam", "follower_gripper": "linear_4310"}}, I2RT_ROOT, None
    )


def make_episode(root: Path, number: int, xml: Path, gap: bool = False) -> Path:
    episode = root / str(number)
    episode.mkdir(parents=True)
    origin = number * 10_000_000_000
    metadata = {
        "status": "complete",
        "start_monotonic_ns": origin,
        "end_monotonic_ns": origin + 1_000_000_000,
        "settings": {"arm": "yam", "follower_gripper": "linear_4310", "control_hz": 60},
    }
    (episode / "metadata.json").write_text(json.dumps(metadata))
    kinematics = ArmKinematics(xml, 1e-8, 1, 0.095, "right")
    for side, offset in (("left", 0.4), ("right", -0.3)):
        with (episode / f"robot_{side}.pkl").open("wb") as stream:
            for i in range(60):
                if gap and side == "right" and 15 <= i < 30:
                    continue
                obs = {
                    "joint_pos": np.array(
                        [0.2 + number * 0.01 + i * 0.001, 1.0, 1.2, -0.3, 0.2, 0.4, 0.5]
                    ),
                    "joint_vel": np.arange(7) * 0.01,
                    "joint_eff_measured": np.zeros(7),
                    "joint_eff_external": np.zeros(7),
                }
                projected = kinematics.project(obs)
                obs["joint_eff_external"][:6] = projected[
                    "right_robot_jacobian"
                ].T @ np.array([1, -2, 3, 0.1, -0.2, 0.3])
                obs["joint_eff_measured"][:6] = obs["joint_eff_external"][:6] + 1
                q = (
                    obs["joint_pos"].copy() + offset
                )  # leader deliberately differs from the follower
                q[-1] = 0.3 + i * 0.001
                dq = np.arange(6) * 0.03 + i * 0.002 + offset
                ns = origin + round(i * 1e9 / 60)
                pickle.dump(
                    {
                        "monotonic_ns": ns,
                        "timestamp_ns": 1_700_000_000_000_000_000 + ns,
                        "follower": obs,
                        "leader": {"joint_pos": q[:6], "joint_vel": dq},
                        "leader_joint_pos_with_gripper": q,
                        "teleoperation_mode": "idle",
                        "cartesian": {"tcp_wrench_estimate": np.ones(6) * 999},
                    },
                    stream,
                )
    with (episode / "cameras.jsonl").open("w") as stream:
        for name in CAMERAS:
            (episode / name).mkdir()
            for i in range(0, 60, 3):
                filename = f"{name}/{i}.jpg"
                image = np.zeros((24, 32, 3), dtype=np.uint8)
                image[..., 2] = 255
                assert cv2.imwrite(str(episode / filename), image)
                stream.write(
                    json.dumps(
                        {
                            "camera": name,
                            "path": filename,
                            "monotonic_ns": origin + round(i * 1e9 / 60),
                        }
                    )
                    + "\n"
                )
    return episode


@pytest.fixture
def converted(tmp_path: Path, robot_xml: Path) -> Path:
    raw = tmp_path / "raw"
    for number in (1, 2):
        make_episode(raw, number, robot_xml)
    output = tmp_path / "dataset"
    convert(
        raw,
        output,
        robot_xml=robot_xml,
        damping=1e-8,
        image_width=320,
        image_height=240,
    )
    return output


def test_conversion_is_symmetric_and_labels_next_leader_not_follower(
    converted: Path,
) -> None:
    root = zarr.open_group(str(converted / "replay_buffer.zarr"), mode="r")
    data = root["data"]
    np.testing.assert_array_equal(root["meta/episode_ends"], [59, 118])
    assert data["action"].shape == (118, 28)
    assert data["right_robot_tcp_pose"].shape == (118, 9)
    assert data["right_robot_tcp_wrench"].shape == (118, 6)
    assert data["right_robot_jacobian"].shape == (118, 6, 6)
    assert not any("marker" in key for key in data)
    left_keys = {key.removeprefix("left_") for key in data if key.startswith("left_")}
    right_keys = {
        key.removeprefix("right_") for key in data if key.startswith("right_")
    }
    assert left_keys == right_keys
    for key in left_keys:
        assert data[f"left_{key}"].shape == data[f"right_{key}"].shape
    assert "right_robot_gripper_force" not in data
    for name in CAMERAS:
        image = data[f"{name}_img"]
        assert image.shape == (118, 240, 320, 3) and image.dtype == np.uint8
        assert image[0, 0, 0, 0] > 250 and image[0, 0, 0, 2] < 5
    for number, start, end in ((1, 0, 59), (2, 59, 118)):
        for side in ("left", "right"):
            raw = read_robot_stream(
                converted.parent / "raw" / str(number) / f"robot_{side}.pkl"
            )
            expected = np.stack([leader_target(row) for row in raw[1:]])
            np.testing.assert_allclose(
                data[f"{side}_robot_action"][start:end], expected
            )
            np.testing.assert_allclose(
                data[f"{side}_robot_target"][start:end],
                np.stack([leader_target(row) for row in raw[:-1]]),
            )
            assert data[f"{side}_robot_action"].shape == (118, 14)
            np.testing.assert_allclose(data[f"{side}_robot_action"][start:end, -1], 0)
    np.testing.assert_allclose(data["action"][:, :14], data["left_robot_action"][:])
    np.testing.assert_allclose(data["action"][:, 14:], data["right_robot_action"][:])
    assert not np.allclose(data["action"][58], data["target"][59])
    assert data["action_monotonic_ns"].shape == (118,)
    assert data["timestamp"].dtype == np.float64
    ns = data["monotonic_ns"][:]
    assert np.all(data["robot_source_monotonic_ns"][:] <= ns[:, None])
    assert np.all(data["camera_source_monotonic_ns"][:] <= ns[:, None])
    assert np.all(data["action_source_monotonic_ns"][:] > ns[:, None])
    np.testing.assert_allclose(np.diff(data["timestamp"][:59]), 1 / 60, atol=1e-6)
    with pytest.raises(FileExistsError):
        convert(converted.parent / "raw", converted)


def test_wrench_reconstructs_effort_and_rotates_to_tcp(robot_xml: Path) -> None:
    kinematics = ArmKinematics(robot_xml, 1e-8, 1, 0.095, "right")
    obs = {
        "joint_pos": np.array([0.2, 1, 1.2, -0.3, 0.2, 0.4, 0.5]),
        "joint_vel": np.ones(7),
        "joint_eff_measured": np.zeros(7),
        "joint_eff_external": np.zeros(7),
    }
    first = kinematics.project(obs)
    rotation = kinematics.data.site_xmat[kinematics.site].reshape(3, 3).copy()
    base_wrench = np.array([1, -2, 3, 0.1, -0.2, 0.3])
    obs["joint_eff_external"][:6] = first["right_robot_jacobian"].T @ base_wrench
    second = kinematics.project(obs)
    expected = np.concatenate(
        [rotation.T @ base_wrench[:3], rotation.T @ base_wrench[3:]]
    )
    np.testing.assert_allclose(second["right_robot_tcp_wrench"], expected, atol=2e-5)
    obs["joint_pos"][1] += 0.2
    third = kinematics.project(obs)
    assert not np.allclose(third["right_robot_jacobian"], first["right_robot_jacobian"])


def test_missing_samples_split_sequences_without_cross_gap_actions(
    tmp_path: Path, robot_xml: Path
) -> None:
    episode = make_episode(tmp_path / "raw", 1, robot_xml, gap=True)
    output = tmp_path / "dataset"
    convert(episode, output, robot_xml=robot_xml, max_robot_age_s=0.02)
    root = zarr.open_group(str(output / "replay_buffer.zarr"), mode="r")
    ends = root["meta/episode_ends"][:]
    assert len(ends) == 2
    start = 0
    for end in ends:
        ns = root["data/monotonic_ns"][start:end]
        np.testing.assert_allclose(np.diff(ns) / 1e9, 1 / 60, atol=1e-8)
        np.testing.assert_allclose(
            root["data/action_monotonic_ns"][start:end] - ns, round(1e9 / 60), atol=1
        )
        assert np.all(root["data/action_source_monotonic_ns"][start:end] > ns[:, None])
        start = end


def config(name: str, overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


@pytest.mark.parametrize(
    "key,dim", [("left_robot_action", 14), ("right_robot_action", 14), ("action", 28)]
)
def test_action_selection_and_joint_normalization(
    converted: Path, key: str, dim: int
) -> None:
    cfg = config(
        "train_forcegello_right_at",
        [
            f"task.dataset_path={converted}",
            "training.device=cpu",
            "task.dataset.val_ratio=0",
            f"task.shape_meta.action.key={key}",
            f"task.shape_meta.action.shape=[{dim}]",
        ],
    )
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert dataset.action_key == key
    expected = zarr.open_group(str(converted / "replay_buffer.zarr"), mode="r")[
        f"data/{key.removesuffix('action')}target"
    ][np.concatenate([np.arange(3, 58, 3), np.arange(62, 117, 3)])]
    np.testing.assert_allclose(dataset.get_all_actions().numpy(), expected)
    assert dataset[0]["action"].shape == (32, dim)
    normalizer = dataset.get_normalizer()["action"]
    values = torch.from_numpy(expected)
    normalized = normalizer.normalize(values)
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalizer.unnormalize(normalized), values)


def test_right_arm_dataset_and_both_network_stages(
    converted: Path, tmp_path: Path
) -> None:
    torch.set_num_threads(1)
    cfg = config(
        "train_forcegello_right_at",
        [
            f"task.dataset_path={converted}",
            "training.device=cpu",
            "task.dataset.val_ratio=0",
        ],
    )
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["action"].shape == (2, 32, 14)
    assert batch["obs"]["right_robot_joint_pos"].shape == (2, 2, 7)
    assert batch["extended_obs"]["right_robot_tcp_wrench"].shape == (2, 32, 6)
    assert batch["extended_obs"]["right_robot_joint_pos"].shape == (2, 32, 7)
    assert batch["extended_obs"]["right_robot_joint_vel"].shape == (2, 32, 7)
    assert not any(key.startswith("left") for key in batch["obs"])
    assert dataset.action_key == "right_robot_action"
    expected = zarr.open_group(str(converted / "replay_buffer.zarr"), mode="r")[
        "data/right_robot_target"
    ][np.concatenate([np.arange(3, 58, 3), np.arange(62, 117, 3)])]
    np.testing.assert_allclose(dataset.get_all_actions().numpy(), expected)
    normalizer = dataset.get_normalizer()
    tokenizer = hydra.utils.instantiate(cfg.policy)
    tokenizer.set_normalizer(normalizer)
    loss = tokenizer.compute_loss_and_metric(batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert tokenizer.decoder.rnn.weight_ih_l0.grad is not None
    ckpt = tmp_path / "at.ckpt"
    # Workspace checkpoint layout, including Hydra metadata, as used by actual stage 1.
    torch.save({"state_dicts": {"model": tokenizer.state_dict()}, "cfg": cfg}, ckpt)

    cfg = config(
        "train_forcegello_right_rdp",
        [
            f"task.dataset_path={converted}",
            "training.device=cpu",
            "task.dataset.val_ratio=0",
            f"at_load_dir={ckpt}",
            "policy.down_dims=[32,64,128]",
            "policy.num_inference_steps=2",
        ],
    )
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["obs"]["right_wrist_img"].shape == (2, 2, 3, 240, 320)
    assert batch["obs"]["external_img"].shape == (2, 2, 3, 240, 320)
    assert "left_wrist_img" not in batch["obs"]
    policy = hydra.utils.instantiate(cfg.policy).to(torch.device("cpu"))
    policy.set_normalizer(dataset.get_normalizer())
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.model.final_conv[-1].weight.grad is not None
    with torch.no_grad():
        result = policy.predict_action(
            batch["obs"],
            dataset_obs_temporal_downsample_ratio=4,
            extended_obs_dict=batch["extended_obs"],
        )
    assert result["action_pred"].shape == (2, 32, 14)
    assert result["action"].shape == (2, 25, 14)
    assert torch.isfinite(result["action"]).all()


@pytest.mark.parametrize("anchor", [10, 11, 12, 13, 29, 30, 31, 32])
def test_rgb_5hz_and_proprio_20hz_are_sampled_independently(
    converted: Path, anchor: int
) -> None:
    from reactive_diffusion_policy.dataset.real_image_tactile_dataset import (
        RealImageTactileDataset,
    )

    root = zarr.open_group(str(converted / "replay_buffer.zarr"), mode="a")
    # Frame IDs reveal the precise source time, including between visual updates.
    for key in ("external_img", "right_wrist_img"):
        root[f"data/{key}"][:, 0, 0, 0] = np.arange(118, dtype=np.uint8)
    cfg = config("train_forcegello_right_rdp", [])
    dataset = RealImageTactileDataset(
        shape_meta=cfg.task.shape_meta,
        dataset_path=str(converted),
        horizon=32,
        pad_before=7,
        pad_after=24,
        n_obs_steps=8,
        obs_temporal_downsample_ratio=4,
        sample_hz=20,
    )
    np.testing.assert_array_equal(dataset.replay_buffer.episode_ends, [19, 38])
    index = next(
        i
        for i, (start, _, padding, _) in enumerate(dataset.sampler.indices)
        if start + 7 - padding == anchor
    )
    sample = dataset[index]
    episode_start = 0 if anchor < 19 else 19
    raw_start = 0 if anchor < 19 else 59
    local = anchor - episode_start
    expected_rgb = raw_start + (np.array([local - 4, local]) // 4 * 4) * 3
    for key in ("external_img", "right_wrist_img"):
        np.testing.assert_allclose(
            sample["obs"][key][:, 0, 0, 0].numpy() * 255, expected_rgb, atol=1e-5
        )
    for key in (
        "right_robot_joint_pos",
        "right_robot_joint_vel",
        "right_robot_tcp_wrench",
    ):
        np.testing.assert_allclose(
            sample["obs"][key],
            root[f"data/{key}"][
                raw_start + (local - 1) * 3 : raw_start + local * 3 + 1 : 3
            ],
        )
        np.testing.assert_allclose(
            sample["extended_obs"][key][:8],
            root[f"data/{key}"][
                raw_start + (local - 7) * 3 : raw_start + local * 3 + 1 : 3
            ],
        )
    np.testing.assert_allclose(
        sample["action"][7],
        root["data/right_robot_target"][raw_start + (local + 1) * 3],
    )
    assert not np.allclose(
        sample["action"][7], root["data/right_robot_action"][raw_start + local * 3]
    )  # The label must be 50 ms ahead, not the old 16.7 ms target.
    assert root["data/action"].shape == (118, 28)  # Source data is unchanged.


def test_resampling_validates_rates_and_preserves_episode_boundaries(
    converted: Path,
) -> None:
    from reactive_diffusion_policy.dataset.forcegello_sampling import load_joint_replay

    path = str(converted / "replay_buffer.zarr")
    keys = ["right_robot_joint_pos", "right_robot_action"]
    for rate in (0, -20, float("nan"), 24, 120):
        with pytest.raises(ValueError):
            load_joint_replay(path, keys, "right_robot_action", rate)
    replay = load_joint_replay(path, keys, "right_robot_action", 60)
    source = zarr.open_group(path, mode="r")
    np.testing.assert_array_equal(replay.episode_ends, [59, 118])
    np.testing.assert_array_equal(
        replay["right_robot_action"], source["data/right_robot_action"]
    )
    # Too-short segments disappear; one episode's last target never comes from the next.
    source = zarr.open_group(path, mode="a")
    source["meta/episode_ends"][:] = [2, 118]
    replay = load_joint_replay(path, keys, "right_robot_action", 20)
    np.testing.assert_array_equal(replay.episode_ends, [38])
    np.testing.assert_array_equal(
        replay["right_robot_action"][0], source["data/right_robot_target"][5]
    )
