"""Regression records for discarded host passes, adapter leakage, size and routing errors."""
import ast
import contextlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor" / "deps")]

import torch
from pi_sensenova.settings import DEFAULTS, UI_KEYS, Settings, compose, dimensions, memory_mode
from pi_sensenova.adapters import RuntimeAdapter
from pi_sensenova.engine import Cancelled, progress, to_images
from pi_sensenova import run, detect


class DetectionTests(unittest.TestCase):
    def test_gguf_identity_cache_invalidates_when_file_changes(self):
        calls = []
        def reader(path, mode):
            calls.append(path)
            names = ("fm_modules.test", "language_model.test") if len(calls) == 1 else ("unrelated.weight",)
            return types.SimpleNamespace(tensors=[types.SimpleNamespace(name=n) for n in names])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "renamed.gguf"
            path.write_bytes(b"one")
            with patch.dict(sys.modules, {"gguf": types.SimpleNamespace(GGUFReader=reader)}):
                self.assertTrue(detect.is_ours(path))
                self.assertTrue(detect.is_ours(path))
                self.assertEqual(len(calls), 1)
                path.write_bytes(b"different checkpoint")
                self.assertFalse(detect.is_ours(path))
                self.assertEqual(len(calls), 2)
        detect._gguf_identity.cache_clear()


class SettingsTests(unittest.TestCase):
    def test_actual_ui_order(self):
        tree = ast.parse((ROOT / "scripts" / "engine.py").read_text(encoding="utf-8"))
        assigned = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "controls" for t in n.targets))
        self.assertEqual(tuple(n.id for n in assigned.elts), UI_KEYS)
        self.assertEqual(Settings.from_ui(DEFAULTS), Settings())
        with self.assertRaises(ValueError):
            Settings.from_ui(DEFAULTS[:6])

    def test_resolution_not_upscaled(self):
        self.assertEqual(dimensions(1024, 768), (1024, 768))
        self.assertEqual(dimensions(1536, 1536), (1536, 1536))
        with self.assertRaises(ValueError):
            dimensions(1001, 1024)

    def test_natural_is_explicit_and_preserves_prompt(self):
        self.assertEqual(compose("A ceramic cup", "", False), "A ceramic cup")
        result = compose("A ceramic cup", "text", True)
        self.assertTrue(result.startswith("A ceramic cup"))
        self.assertIn("No beauty retouching", result)
        self.assertTrue(result.endswith("Avoid: text"))

    def test_memory_uses_weight_size(self):
        gb = 2**30
        self.assertEqual(memory_mode(30*gb, 20*gb), "full")
        self.assertEqual(memory_mode(30*gb, 35*gb), "fast")
        self.assertEqual(memory_mode(8*gb, 20*gb), "low")


class AdapterTests(unittest.TestCase):
    def test_exact_delta_and_removal(self):
        torch.manual_seed(1)
        model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False))
        x = torch.randn(2, 4)
        before = model[0].weight.detach().clone()
        baseline = model(x)
        down, up = torch.randn(2, 4), torch.randn(3, 2)
        adapter = RuntimeAdapter()
        self.assertEqual(adapter.attach(model, {"diffusion_model.0.lora_down.weight": down, "diffusion_model.0.lora_up.weight": up}), 1)
        torch.testing.assert_close(model(x), baseline + x @ down.T @ up.T)
        torch.testing.assert_close(model[0].weight, before, rtol=0, atol=0)
        adapter.remove()
        torch.testing.assert_close(model(x), baseline, rtol=0, atol=0)

    def test_malformed_adapter_is_atomic(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 3))
        adapter = RuntimeAdapter()
        with self.assertRaises(ValueError):
            adapter.attach(model, {"0.lora_down.weight": torch.ones(2, 4)})
        self.assertFalse(model[0]._forward_hooks)


class ProgressTests(unittest.TestCase):
    def test_streaming_restores_parameters_after_failure(self):
        from pi_sensenova.streaming import BlockStream
        model = torch.nn.Module()
        model.language_model = torch.nn.Module()
        model.language_model.model = torch.nn.Module()
        model.language_model.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        layer = model.language_model.model.layers[0]
        original = layer.weight.detach().clone()
        with self.assertRaises(RuntimeError):
            with BlockStream(model, target="cpu"):
                layer(torch.ones(1, 2))
                raise RuntimeError("interrupted")
        torch.testing.assert_close(layer.weight, original, rtol=0, atol=0)
        self.assertFalse(layer._forward_hooks)
        self.assertFalse(layer._forward_pre_hooks)

    def test_stop_restores_instance(self):
        class Model:
            def unpatchify(self, value):
                return value
        model = Model()
        with self.assertRaises(Cancelled):
            with progress(model, 3, lambda step, total: step == 2):
                model.unpatchify(1)
                model.unpatchify(2)
        self.assertNotIn("unpatchify", model.__dict__)
        self.assertEqual(model.unpatchify(7), 7)

    def test_tensor_conversion(self):
        images = to_images(torch.tensor([[[[-1., 1.]], [[-1., 1.]], [[-1., 1.]]]]))
        self.assertEqual(images[0].getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(images[0].getpixel((1, 0)), (255, 255, 255))


class RoutingTests(unittest.TestCase):
    def test_native_unload_preserves_delegate(self):
        calls = []
        sd = types.SimpleNamespace(unload_model_weights=lambda *a, **k: calls.append((a,k)) or "unloaded")
        modules = types.ModuleType("modules")
        modules.sd_models = sd
        with patch.dict(sys.modules, {"modules":modules}), patch.object(run,"release") as release:
            run.install_unload()
            self.assertEqual(sd.unload_model_weights(5, future=True), "unloaded")
            release.assert_called_once()
        self.assertEqual(calls,[((5,),{"future":True})])

    def host(self):
        calls = []
        class Runner:
            def run(self, p, *args, **kwargs):
                calls.append((p, args, kwargs))
                return "original runner"
        processing = types.ModuleType("modules.processing")
        def original(p, *args, **kwargs):
            calls.append((p, args, kwargs))
            return "original API"
        processing.process_images = original
        scripts = types.ModuleType("modules.scripts")
        scripts.ScriptRunner = Runner
        modules = types.ModuleType("modules")
        modules.processing = processing
        api = types.ModuleType("modules.api.api")
        api.process_images = original
        return {"modules": modules, "modules.scripts": scripts, "modules.processing": processing, "modules.api.api": api}, Runner, calls

    def test_passthrough_future_arguments_and_return(self):
        host, Runner, calls = self.host()
        p = object()
        with patch.dict(sys.modules, host), patch.object(run.selection, "selected", return_value=None), patch.object(run, "release"):
            self.assertTrue(run.install())
            self.assertEqual(Runner().run(p, 3, future_option=7), "original runner")
            self.assertEqual(host["modules.api.api"].process_images(p, 9, future_option=11), "original API")
        self.assertEqual(calls, [(p, (3,), {"future_option": 7}), (p, (9,), {"future_option": 11})])

    def test_selected_checkpoint_has_zero_host_passes(self):
        host, Runner, calls = self.host()
        with patch.dict(sys.modules, host), patch.object(run.selection, "selected", return_value="ours"), patch.object(run, "takeover", return_value="image"):
            run.install()
            self.assertEqual(Runner().run(object()), "image")
            self.assertEqual(host["modules.api.api"].process_images(object()), "image")
            self.assertFalse(calls)

    def test_incompatible_update_leaves_originals(self):
        host, Runner, _ = self.host()
        def changed(self, p, required_new_argument):
            pass
        Runner.run = changed
        original = host["modules.processing"].process_images
        with patch.dict(sys.modules, host):
            self.assertFalse(run.install())
        self.assertIs(Runner.run, changed)
        self.assertIs(host["modules.processing"].process_images, original)

    def test_api_alias_imported_before_another_extension(self):
        host, Runner, calls = self.host()
        older = host["modules.api.api"].process_images
        host["modules.processing"].process_images = lambda p: "other extension"
        with patch.dict(sys.modules, host), patch.object(run.selection, "selected", return_value="ours"), patch.object(run, "takeover", return_value="image"):
            run.install()
            self.assertEqual(host["modules.api.api"].process_images(object()), "image")
            self.assertFalse(calls)
            with patch.object(run.selection, "selected", return_value=None), patch.object(run, "release"):
                self.assertEqual(host["modules.api.api"].process_images(object()), "original API")


if __name__ == "__main__":
    unittest.main(verbosity=2)
