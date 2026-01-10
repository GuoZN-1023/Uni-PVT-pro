# train.py
import os
import argparse
import yaml
import torch

from models.fusion_model import FusionModel
from utils.dataset import get_dataloaders
from utils.physics_loss import PhysicsLoss
from utils.trainer import train_model
from utils.logger import get_file_logger


def _bridge_loss_cfg_to_training(cfg: dict):

    cfg.setdefault("training", {})
    loss_cfg = cfg.get("loss", {}) or {}
    tr = cfg["training"]

    mapping = {
        "lambda_nonneg": "lambda_nonneg",
        "lambda_smooth": "lambda_smooth",
        "lambda_entropy": "lambda_entropy",
        "loss_type": "loss_type",
        "huber_delta": "huber_delta",
        "lambda_extreme": "lambda_extreme",
        "lambda_relative": "lambda_relative",
        "extreme_alpha": "extreme_alpha",
    }

    for k_loss, k_tr in mapping.items():
        if k_tr not in tr and k_loss in loss_cfg:
            tr[k_tr] = loss_cfg[k_loss]

    if "region_weights" not in tr and "region_weights" in loss_cfg:
        tr["region_weights"] = loss_cfg["region_weights"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="可选：覆盖 config 中的 data 路径，直接指定某个 csv 文件",
    )
    args = parser.parse_args()

  
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)


    if args.data is not None:
        cfg.setdefault("paths", {})
        cfg["paths"]["data"] = args.data


    _bridge_loss_cfg_to_training(cfg)


    save_dir = cfg["paths"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)

    cfg["paths"]["scaler"] = cfg["paths"].get("scaler", os.path.join(save_dir, "scaler.pkl"))


    log_file = os.path.join(save_dir, "logs", "training.log")
    logger = get_file_logger(log_file, name="train")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    dataloaders = get_dataloaders(cfg)

    train_loader = dataloaders["train"]
    if hasattr(train_loader.dataset, "dataset"):
        full_dataset = train_loader.dataset.dataset
    else:
        full_dataset = train_loader.dataset

    real_input_dim = int(full_dataset.input_dim)
    cfg.setdefault("model", {})
    cfg["model"]["input_dim"] = real_input_dim
    logger.info(f"Detected input_dim={real_input_dim}, set cfg['model']['input_dim'] accordingly.")

    with open(os.path.join(save_dir, "config_used.yaml"), "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    model = FusionModel(cfg).to(device)

    criterion = PhysicsLoss(cfg)

    lr = float(cfg["training"]["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_model(model, dataloaders, criterion, optimizer, cfg, device, logger)


if __name__ == "__main__":
    main()
