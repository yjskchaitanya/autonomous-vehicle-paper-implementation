from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.kitti_dataset import KITTIMultiSensorDataset
from .models.segmentation import MultiOrientationSegmenter
from .pipeline import PaperPipeline


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_segmentation(config: dict) -> None:
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    dataset_cfg = config["dataset"]
    seg_cfg = config["segmentation"]
    pre_cfg = config["preprocessing"]

    train_set = KITTIMultiSensorDataset(dataset_cfg, split="train")
    val_set = KITTIMultiSensorDataset(dataset_cfg, split="val")
    train_loader = DataLoader(train_set, batch_size=int(seg_cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=int(seg_cfg["batch_size"]), shuffle=False)

    model = MultiOrientationSegmenter(
        in_channels=4,
        base_channels=int(seg_cfg["base_channels"]),
        num_classes=int(seg_cfg["num_classes"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(seg_cfg["learning_rate"]))
    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(seg_cfg["epochs"])):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"seg epoch {epoch + 1} train"):
            camera = batch["camera"].to(device)
            threshold = _batch_threshold(camera, pre_cfg).to(device)
            x = torch.cat([camera, threshold], dim=1)
            y = batch["segmentation_mask"].to(device)
            logits = model(x, list(seg_cfg["orientations"]))
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"seg epoch {epoch + 1} val"):
                camera = batch["camera"].to(device)
                threshold = _batch_threshold(camera, pre_cfg).to(device)
                x = torch.cat([camera, threshold], dim=1)
                y = batch["segmentation_mask"].to(device)
                logits = model(x, list(seg_cfg["orientations"]))
                val_losses.append(F.cross_entropy(logits, y).item())

        torch.save(model.state_dict(), output_dir / "segmenter_latest.pt")
        print(
            f"[segmentation] epoch={epoch + 1} "
            f"train_loss={np.mean(train_losses):.4f} val_loss={np.mean(val_losses):.4f}"
        )


def train_dqn(config: dict) -> None:
    from .models.dqn import DQNAgent

    device = config["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    dataset_cfg = config["dataset"]
    dqn_cfg = config["dqn"]
    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = KITTIMultiSensorDataset(dataset_cfg, split="train")
    pipeline = PaperPipeline(config, device=device)
    agent = DQNAgent(dqn_cfg, device=device)

    for epoch in range(int(dqn_cfg["epochs"])):
        losses = []
        rewards = []
        for sample in tqdm(dataset, desc=f"dqn epoch {epoch + 1}"):
            camera = (sample["camera"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            lidar = (sample["lidar"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            weather = (sample["weather"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            output = pipeline.run(camera, lidar, weather)
            state = output.state_vector
            action = agent.select_action(state)
            reward = _reward_from_output(output, action)
            next_state = state.copy()
            done = False
            agent.push(state, action, reward, next_state, done)
            loss = agent.train_step()

            rewards.append(reward)
            if loss is not None:
                losses.append(loss)

        torch.save(agent.policy_net.state_dict(), output_dir / "dqn_latest.pt")
        print(
            f"[dqn] epoch={epoch + 1} "
            f"avg_reward={np.mean(rewards):.4f} "
            f"avg_loss={(np.mean(losses) if losses else 0.0):.4f} "
            f"epsilon={agent.epsilon:.4f}"
        )


def _batch_threshold(camera: torch.Tensor, pre_cfg: dict) -> torch.Tensor:
    images = (camera.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    threshold_batch = []
    from .models.preprocessing import iawmf_threshold

    for image in images:
        threshold = iawmf_threshold(
            image,
            kernel_size=int(pre_cfg["weighted_mean_kernel"]),
            c=int(pre_cfg["weighted_mean_c"]),
        )
        threshold_batch.append(threshold[None, ...] / 255.0)
    threshold = torch.from_numpy(np.stack(threshold_batch, axis=0)).float()
    return threshold


def _reward_from_output(output, action: int) -> float:
    best_score = float(output.evo_result["best_score"])
    obstacle_count = sum(1 for d in output.detections if d.label in {"car", "truck", "bus", "pedestrian"})
    safe_bonus = 1.0 / (1.0 + obstacle_count)
    action_penalty = 0.05 if action in {3, 4} else 0.0
    return best_score + safe_bonus - action_penalty
