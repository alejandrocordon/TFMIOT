"""Tests for Sprint 8 features: scaffolding, server, HF, detection, mixed precision."""

import tempfile
from pathlib import Path

import pytest


class TestProjectScaffolding:
    def test_create_classification_project(self):
        from mlforge.project_templates import create_project

        with tempfile.TemporaryDirectory() as tmp:
            project = create_project("my_clf", tmp, "classification")
            assert (project / "config.yaml").exists()
            assert (project / "train.py").exists()
            assert (project / "README.md").exists()
            assert (project / "data").is_dir()
            config = (project / "config.yaml").read_text()
            assert "my_clf" in config
            assert "framework: pytorch" in config

    def test_create_detection_project(self):
        from mlforge.project_templates import create_project

        with tempfile.TemporaryDirectory() as tmp:
            project = create_project("my_det", tmp, "detection")
            config = (project / "config.yaml").read_text()
            assert "yolov8n" in config
            assert "detection" in config

    def test_create_tf_project(self):
        from mlforge.project_templates import create_project

        with tempfile.TemporaryDirectory() as tmp:
            project = create_project("my_tf", tmp, "classification", "tensorflow")
            config = (project / "config.yaml").read_text()
            assert "framework: tensorflow" in config

    def test_unknown_task_raises(self):
        from mlforge.project_templates import create_project

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="Unknown task"):
                create_project("bad", tmp, "segmentation")


class TestInferenceServer:
    def test_server_module_imports(self):
        from mlforge.deploy.server import ONNXPredictor, PyTorchPredictor, create_app
        assert ONNXPredictor is not None
        assert create_app is not None

    def test_softmax(self):
        import numpy as np
        from mlforge.deploy.server import _softmax

        logits = np.array([1.0, 2.0, 3.0])
        probs = _softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert probs[2] > probs[1] > probs[0]

    def test_preprocess_image(self):
        import numpy as np
        from PIL import Image
        from mlforge.deploy.server import _preprocess_image

        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = _preprocess_image(img, 224)
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32


class TestDetection:
    def test_list_detection_models(self):
        from mlforge.models.detector import list_detection_models

        models = list_detection_models()
        assert len(models) == 5
        names = [m["name"] for m in models]
        assert "yolov8n" in names
        assert "yolov8x" in names

    def test_yolo_variants_info(self):
        from mlforge.models.detector import YOLO_VARIANTS

        for name, info in YOLO_VARIANTS.items():
            assert "description" in info
            assert "size" in info


class TestHuggingFace:
    def test_dataset_wrapper(self):
        from mlforge.data.huggingface import HuggingFaceImageDataset
        from PIL import Image
        import numpy as np

        # Mock HF dataset with list of dicts
        mock_data = [
            {"image": Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)), "label": 0},
            {"image": Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)), "label": 1},
        ]

        class MockHFDataset:
            def __len__(self):
                return len(mock_data)
            def __getitem__(self, idx):
                return mock_data[idx]

        ds = HuggingFaceImageDataset(MockHFDataset())
        assert len(ds) == 2
        img, label = ds[0]
        assert isinstance(img, Image.Image)
        assert label == 0


class TestMixedPrecision:
    def test_train_one_epoch_no_amp(self):
        """Training works with AMP disabled (default on CPU)."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from mlforge.training.trainer_pytorch import _train_one_epoch

        model = nn.Linear(10, 3)
        dataset = TensorDataset(torch.randn(20, 10), torch.randint(0, 3, (20,)))
        loader = DataLoader(dataset, batch_size=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        loss, acc = _train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        assert loss > 0
        assert 0 <= acc <= 1

    def test_train_one_epoch_with_amp_cpu(self):
        """Training works with AMP on CPU (bfloat16)."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from mlforge.training.trainer_pytorch import _train_one_epoch

        model = nn.Linear(10, 3)
        dataset = TensorDataset(torch.randn(20, 10), torch.randint(0, 3, (20,)))
        loader = DataLoader(dataset, batch_size=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.amp.GradScaler(enabled=False)

        loss, acc = _train_one_epoch(
            model, loader, criterion, optimizer, torch.device("cpu"),
            scaler=scaler, use_amp=True,
        )
        assert loss > 0
        assert 0 <= acc <= 1
