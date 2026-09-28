import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi_sensenova import model_setup


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        modules = types.ModuleType("modules")
        modules.paths = types.SimpleNamespace(models_path=self.temp.name)
        modules.sd_models = types.SimpleNamespace(list_models=MagicMock())
        self.modules = patch.dict(sys.modules, {"modules": modules})
        self.modules.start()

    def tearDown(self):
        self.modules.stop()
        self.temp.cleanup()

    def test_unknown_model_never_downloads(self):
        with self.assertRaises(ValueError):
            list(model_setup.download("unknown"))

    def test_manual_instructions_use_forge_models_path(self):
        text = model_setup.instructions("SenseNova U1.5 8B MoT Q8 GGUF")
        self.assertIn(str(Path(self.temp.name) / "Stable-diffusion"), text)
        self.assertIn(model_setup.RESOURCE_TARGET, text)

    def test_official_download_is_pinned_atomic_and_no_python(self):
        hub = types.ModuleType("huggingface_hub")
        api = MagicMock()
        api.model_info.return_value.sha = "a" * 40
        hub.HfApi = MagicMock(return_value=api)
        hub.hf_hub_download = MagicMock()

        def snapshot(**kwargs):
            folder = Path(kwargs["local_dir"])
            folder.mkdir(parents=True)
            (folder / "config.json").write_text("{}", encoding="utf-8")
            (folder / "model.safetensors").write_bytes(b"weights")
        hub.snapshot_download = MagicMock(side_effect=snapshot)
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            statuses = list(model_setup.download("Official SenseNova U1.5 8B MoT"))
        call = hub.snapshot_download.call_args.kwargs
        self.assertEqual(call["revision"], "a" * 40)
        self.assertNotIn("*.py", call["allow_patterns"])
        self.assertTrue((Path(self.temp.name) / "Stable-diffusion" /
                         "SenseNova-U1.5-8B-MoT" / "config.json").is_file())
        self.assertTrue(statuses[-1].startswith("Ready:"))

    def test_gguf_download_adds_official_resources(self):
        hub = types.ModuleType("huggingface_hub")
        api = MagicMock()
        api.model_info.side_effect = [types.SimpleNamespace(sha="b" * 40),
                                      types.SimpleNamespace(sha="c" * 40)]
        hub.HfApi = MagicMock(return_value=api)
        def file_download(**kwargs):
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.write_bytes(b"GGUF")
            return str(target)
        hub.hf_hub_download = MagicMock(side_effect=file_download)

        def snapshot(**kwargs):
            folder = Path(kwargs["local_dir"])
            folder.mkdir(parents=True)
            (folder / "config.json").write_text("{}", encoding="utf-8")
            (folder / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (folder / "tokenizer.json").write_text("{}", encoding="utf-8")
        hub.snapshot_download = MagicMock(side_effect=snapshot)
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            list(model_setup.download("SenseNova U1.5 8B MoT Q8 GGUF"))
        root = Path(self.temp.name) / "Stable-diffusion"
        self.assertTrue((root / "SenseNova-U1.5-8B-MoT-Q8_0.gguf").is_file())
        self.assertTrue((root / model_setup.RESOURCE_TARGET / "tokenizer.json").is_file())
        self.assertEqual(hub.hf_hub_download.call_args.kwargs["revision"], "b" * 40)
        self.assertIn("local_dir", hub.hf_hub_download.call_args.kwargs)

    def test_model_index_rejects_missing_shard(self):
        folder = Path(self.temp.name) / "model"
        folder.mkdir()
        (folder / "config.json").write_text("{}", encoding="utf-8")
        (folder / "model.safetensors").write_bytes(b"part")
        (folder / "model.safetensors.index.json").write_text(
            '{"weight_map":{"layer":"missing.safetensors"}}', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Missing safetensors shard"):
            model_setup._validate_model(folder)

    def test_existing_invalid_resources_are_rejected(self):
        root = Path(self.temp.name) / "Stable-diffusion"
        resources = root / model_setup.RESOURCE_TARGET
        resources.mkdir(parents=True)
        (resources / "config.json").write_text("{}", encoding="utf-8")
        hub = types.ModuleType("huggingface_hub")
        api = MagicMock()
        api.model_info.return_value.sha = "d" * 40
        hub.HfApi = MagicMock(return_value=api)
        hub.hf_hub_download = MagicMock()
        hub.snapshot_download = MagicMock()
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            with self.assertRaisesRegex(RuntimeError, "tokenizer_config"):
                list(model_setup.download("SenseNova U1.5 8B MoT Q8 GGUF"))
        hub.hf_hub_download.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
