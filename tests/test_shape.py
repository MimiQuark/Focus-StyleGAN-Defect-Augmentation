import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from model.models.focus_stylegan import FocusStyleGAN  # noqa: E402
from model.utils.config import Config  # noqa: E402


def test_forward_shapes():
    """前向传播必须成功，且输出形状/数值有效（失败即测试失败）。"""
    config = Config(str(Path(__file__).resolve().parent.parent
                        / "core" / "integrated_config.yaml"))
    model = FocusStyleGAN(config=config.model)

    batch_size = 2
    real_images = torch.randn(batch_size, 3, 256, 256)
    z_defect = torch.randn(batch_size, config.model.generator.latent_dim)
    z_background = torch.randn(batch_size, config.model.generator.latent_dim)

    outputs = model(real_images, z_defect, z_background)

    assert outputs["fused_images"].shape == (batch_size, 3, 256, 256)
    assert outputs["d_real"].shape[0] == batch_size
    assert outputs["d_fake"].shape[0] == batch_size
    assert torch.isfinite(outputs["fused_images"]).all()
    assert outputs["fused_images"].min() >= -1.0 and \
        outputs["fused_images"].max() <= 1.0


if __name__ == "__main__":
    test_forward_shapes()
    print("前向传播形状/数值断言通过")
