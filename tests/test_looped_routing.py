import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pi_sensenova import detect, selection, run, model_setup
from pi_sensenova.settings import Settings, DEFAULTS


class LoopedRoutingTests(unittest.TestCase):
    def test_looped_download_is_local_pinned_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hub = types.ModuleType("huggingface_hub")
            api = MagicMock()
            api.model_info.return_value.sha = "c" * 40
            hub.HfApi = MagicMock(return_value=api)
            def snapshot(**kwargs):
                folder = Path(kwargs["local_dir"])
                folder.mkdir()
                (folder / "config.json").write_text('{"model_type":"t5","d_model":1024}')
                for name in ("tokenizer_config.json", "tokenizer.json", "model.safetensors"):
                    (folder / name).write_text("{}")
            hub.snapshot_download = MagicMock(side_effect=snapshot)
            def checkpoint(**kwargs):
                file = Path(kwargs["local_dir"]) / kwargs["filename"]
                weights = {k: torch.zeros(1) for k in (
                    "img_embed.proj1.weight", "txt_embed.weight", "mask_token", "final.weight")}
                torch.save({"config":{"loop_split":[6,5,6]},"ema":weights}, file)
                return str(file)
            hub.hf_hub_download = MagicMock(side_effect=checkpoint)
            with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.object(model_setup, "_models_dir", return_value=root), patch.object(model_setup, "_refresh"):
                list(model_setup.download("Looped-DiT B32 · fewer image tokens"))
            self.assertTrue((root / "looped-dit-b32.pt").is_file())
            self.assertTrue((root / "flan-t5-large" / "config.json").is_file())
            self.assertEqual(hub.hf_hub_download.call_args.kwargs["revision"], "c" * 40)
            self.assertNotIn("*.py", hub.snapshot_download.call_args.kwargs["allow_patterns"])
            self.assertFalse(list(root.glob(".sensenova-download-*")))

    def test_signature_not_filename_and_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renamed.pt"
            weights = {k: torch.zeros(1) for k in (
                "img_embed.proj1.weight", "txt_embed.weight", "mask_token", "final.weight")}
            torch.save({"config": {"image_size": 512, "patch_size": 16, "loop_split": [6, 5, 6]}, "ema": weights}, path)
            self.assertTrue(detect.is_looped(path))
            self.assertTrue(detect.is_ours(path))
            torch.save({"unrelated": torch.zeros(2)}, path)
            self.assertFalse(detect.is_ours(path))

    def test_old_saved_controls_keep_new_default(self):
        self.assertEqual(Settings.from_ui(DEFAULTS[:8]).loop_depth, 4)
        self.assertEqual(Settings.from_ui(DEFAULTS[:7]).loop_depth, 4)
        for value in (0, 17, 1.5, float("nan"), True):
            with self.assertRaises(ValueError):
                Settings.from_ui((*DEFAULTS[:8], value))

    def test_native_unload_releases_both_loaded_engines(self):
        first, second = MagicMock(), MagicMock()
        with patch.dict(sys.modules, {
            "pi_sensenova.engine": types.SimpleNamespace(ENGINE=first),
            "pi_sensenova.looped": types.SimpleNamespace(LOOPED_ENGINE=second),
        }):
            run.release()
        first.unload.assert_called_once()
        second.unload.assert_called_once()

    def test_registration_of_pt_does_not_modify_other_checkpoints(self):
        registered = []
        class Checkpoint:
            def __init__(self, filename):
                self.filename = filename
                self.ids = []
            def register(self):
                registered.append(self)
        host = types.ModuleType("modules")
        host.sd_models = types.SimpleNamespace(checkpoints_list={}, CheckpointInfo=Checkpoint)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "looped.pt"
            path.write_bytes(b"dummy")
            with patch.dict(sys.modules, {"modules": host}), patch.object(selection, "model_roots", return_value=[Path(directory)]), patch.object(detect, "is_looped", return_value=True):
                selection.register()
            self.assertEqual(registered[0].filename, str(path.resolve()))
            self.assertIn("Looped-DiT", registered[0].title)


if __name__ == "__main__":
    unittest.main()
