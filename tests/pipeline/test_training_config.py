"""Regression tests for MyTrainingConfig and Accelerate shared state."""

from accelerate import Accelerator

from mlkit.pipeline.dist_pipeline_impl import MyTrainingConfig


def test_my_training_config_does_not_reset_gradient_accumulation() -> None:
    """Building MyTrainingConfig must not construct Accelerator and clobber GradientState."""
    acc = Accelerator(gradient_accumulation_steps=4)
    assert acc.gradient_accumulation_steps == 4

    config = MyTrainingConfig(
        train_strategy="epochs",
        n_epochs=1,
        batch_size=2,
        real_batch_size=4,
        mixed_precision="no",
    )
    assert config.gradient_accumulation_steps == 2
    assert acc.gradient_accumulation_steps == 4