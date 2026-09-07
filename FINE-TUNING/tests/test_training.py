"""Tests for training pipeline (mock mode — no GPU required)."""
import pytest
from src.training.train import LoRAConfig, TrainingConfig, train


class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16
        assert 0 < cfg.dropout < 1
        assert "q_proj" in cfg.target_modules

    def test_rank_32(self):
        cfg = LoRAConfig(rank=32)
        assert cfg.rank == 32


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.num_epochs > 0
        assert cfg.per_device_batch_size > 0
        assert 0 < cfg.learning_rate < 1

    def test_lora_config_embedded(self):
        cfg = TrainingConfig()
        assert isinstance(cfg.lora, LoRAConfig)


class TestTrain:
    def _make_records(self, n=10):
        return [
            {
                "id": f"r{i}",
                "instruction": "Extract entities.",
                "input": "NVIDIA reported $5B revenue.",
                "output": '{"entities": []}',
            }
            for i in range(n)
        ]

    def test_mock_training_returns_dict(self):
        cfg = TrainingConfig(output_dir="/tmp/test_checkpoint")
        records = self._make_records()
        result = train(cfg, records, records[:2])
        assert isinstance(result, dict)
        assert "status" in result

    def test_mock_training_has_params(self):
        cfg = TrainingConfig(output_dir="/tmp/test_checkpoint")
        records = self._make_records()
        result = train(cfg, records, records[:2])
        assert "trainable_params" in result
        assert result["trainable_params"] > 0

    def test_rank_8_fewer_trainable_than_32(self):
        records = self._make_records()
        r8_cfg = TrainingConfig(lora=LoRAConfig(rank=8), output_dir="/tmp/r8")
        r32_cfg = TrainingConfig(lora=LoRAConfig(rank=32), output_dir="/tmp/r32")
        r8 = train(r8_cfg, records, records[:2])
        r32 = train(r32_cfg, records, records[:2])
        assert r8["trainable_params"] < r32["trainable_params"]

    def test_trainable_pct_under_5(self):
        cfg = TrainingConfig(lora=LoRAConfig(rank=8), output_dir="/tmp/test")
        result = train(cfg, self._make_records(), [])
        assert result["trainable_pct"] < 5.0  # LoRA is always < 5% of params

    def test_lora_rank_in_result(self):
        cfg = TrainingConfig(lora=LoRAConfig(rank=16), output_dir="/tmp/test")
        result = train(cfg, self._make_records(), [])
        assert result["lora_rank"] == 16
