#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke-test the actual package modules used by training and augmentation."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = [
    ("model.data.loader", ["MVTecADDataset", "create_dataloader"]),
    ("model.models.focus_stylegan", ["FocusStyleGAN"]),
    ("model.training.trainer", ["FocusStyleGANTrainer"]),
    ("model.evaluation.evaluator", ["Evaluator"]),
    ("model.augmentation.augmentor", ["Augmentor"]),
    ("model.utils.config", ["Config"]),
    ("model.utils.logger", ["setup_logger"]),
]


def test_imports():
    missing = []
    for module_path, names in MODULES:
        module = importlib.import_module(module_path)
        for name in names:
            if not hasattr(module, name):
                missing.append(f"{module_path}.{name}")
    assert not missing, f"missing exports: {missing}"


def test_config_loading():
    from model.utils.config import Config

    config = Config(Path(__file__).resolve().parents[1] / "core" / "integrated_config.yaml")
    assert config.data.image_size > 0
    assert config.training.epochs > 0


def test_model_creation():
    import yaml
    from model.models.focus_stylegan import FocusStyleGAN

    config_path = Path(__file__).resolve().parents[1] / "core" / "integrated_config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = FocusStyleGAN(config["model"])
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
