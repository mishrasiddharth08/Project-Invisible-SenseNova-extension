"""Real entry orchestration with fake GPU: saved batches, seeds, Stop, no retries."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pi_sensenova.generate import generate
from pi_sensenova.settings import Settings
from pi_sensenova.engine import Cancelled


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.saved = []
        self.state = types.SimpleNamespace(interrupted=False, skipped=False, nextjob=lambda: None)
        self.p = types.SimpleNamespace(width=1024, height=1024, batch_size=2, n_iter=2, seed=42,
                                     steps=20, cfg_scale=3, prompt="A cup", negative_prompt="", comments=[],
                                     do_not_save_samples=False, outpath_samples="output")
        shared = types.SimpleNamespace(state=self.state, opts=types.SimpleNamespace(samples_save=True, samples_format="png", live_previews_enable=False))
        processing = types.SimpleNamespace(fix_seed=lambda p: None, Processed=lambda p, imgs, **kw: types.SimpleNamespace(images=imgs, **kw))
        images = types.SimpleNamespace(save_image=lambda *a, **k: self.saved.append((a, k)))
        modules = types.ModuleType("modules")
        modules.shared, modules.processing, modules.images = shared, processing, images
        backend = types.ModuleType("backend")
        backend.memory_management = types.SimpleNamespace(unload_all_models=lambda: None)
        self.host = patch.dict(sys.modules, {"modules": modules, "backend": backend})
        self.host.start()
        self.engine = patch("pi_sensenova.generate.ENGINE")
        self.fake = self.engine.start()
        self.fake.mode = "full"
        self.fake.generate.return_value = [Image.new("RGB", (1024, 1024))]

    def tearDown(self):
        self.engine.stop()
        self.host.stop()

    def test_batch_save_and_seeds(self):
        result = generate(self.p, "model.gguf", Settings())
        self.assertEqual(len(result.images), 4)
        self.assertEqual(len(self.saved), 4)
        self.assertEqual(result.all_seeds, [42, 43, 44, 45])
        self.assertEqual(self.p.batch_size, 2)

    def test_failure_preserves_complete_images_without_retry(self):
        self.fake.generate.side_effect = [[Image.new("RGB", (1024, 1024))], RuntimeError("OOM")]
        result = generate(self.p, "model.gguf", Settings())
        self.assertEqual(len(result.images), 1)
        self.assertEqual(len(self.saved), 1)
        self.assertEqual(self.fake.generate.call_count, 2)

    def test_stop_keeps_completed_images(self):
        self.fake.generate.side_effect = [[Image.new("RGB", (1024, 1024))], Cancelled("Stopped")]
        result = generate(self.p, "model.gguf", Settings())
        self.assertEqual(len(result.images), 1)
        self.assertEqual(len(self.saved), 1)

    def test_save_preference(self):
        self.p.do_not_save_samples = True
        result = generate(self.p, "model.gguf", Settings())
        self.assertEqual(len(result.images), 4)
        self.assertFalse(self.saved)

    def test_txt2img_forwards_no_source(self):
        generate(self.p, "model.gguf", Settings())
        self.assertTrue(all(call.kwargs["source"] is None for call in self.fake.generate.call_args_list))

    def test_img2img_forwards_source_image(self):
        source = Image.new("RGB", (512, 512), "red")
        self.p.init_images = [source]
        generate(self.p, "model.gguf", Settings())
        self.assertTrue(all(call.kwargs["source"] is source for call in self.fake.generate.call_args_list))

    def test_img2img_rejects_multiple_sources(self):
        self.p.init_images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
        with self.assertRaisesRegex(ValueError, "one source image"):
            generate(self.p, "model.gguf", Settings())
        self.fake.load.assert_not_called()

    def test_img2img_requires_image(self):
        class ImgRequest(types.SimpleNamespace):
            pass
        sys.modules["modules"].processing.StableDiffusionProcessingImg2Img = ImgRequest
        request = ImgRequest(**vars(self.p), init_images=[])
        with self.assertRaisesRegex(ValueError, "Add a source image"):
            generate(request, "model.gguf", Settings())
        self.fake.load.assert_not_called()

    def test_txt2img_rejects_editing_image(self):
        class TxtRequest(types.SimpleNamespace):
            pass
        sys.modules["modules"].processing.StableDiffusionProcessingTxt2Img = TxtRequest
        request = TxtRequest(**vars(self.p), init_images=[Image.new("RGB", (64, 64))])
        with self.assertRaisesRegex(ValueError, "img2img tab"):
            generate(request, "model.gguf", Settings())
        self.fake.load.assert_not_called()

    def test_fast_requires_visible_recipe(self):
        with self.assertRaises(ValueError):
            generate(self.p, "model.gguf", Settings(fast=True))
        self.fake.load.assert_not_called()

    def test_degrid_saved_and_displayed_for_both_modes(self):
        cleaned = Image.new("RGB", (1024, 1024), "blue")
        self.p.batch_size = self.p.n_iter = 1
        for sources in ([], [Image.new("RGB", (64, 64))]):
            self.p.init_images = sources
            with patch("pi_sensenova.degrid.apply", return_value=cleaned) as cleanup:
                result = generate(self.p, "model.gguf", Settings(degrid=True))
            cleanup.assert_called_once()
            self.assertTrue(cleanup.call_args.kwargs["enabled"])
            self.assertIs(result.images[0], cleaned)
            self.assertIs(self.saved[-1][0][0], cleaned)
            self.assertEqual(self.state.sampling_step, self.p.steps)
            self.assertIn("DeGrid: True", result.infotexts[0])

    def test_degrid_disabled_bypasses_filter(self):
        with patch("pi_sensenova.degrid.apply") as cleanup:
            generate(self.p, "model.gguf", Settings())
        cleanup.assert_not_called()

    def test_old_settings_remain_compatible(self):
        from pi_sensenova.settings import DEFAULTS
        self.assertFalse(Settings.from_ui(DEFAULTS[:7]).degrid)
        self.assertTrue(Settings.from_ui((*DEFAULTS[:7], True)).degrid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
