import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pi_sensenova.looped import sample, upstream, LoopedEngine
from pi_sensenova.streaming import BlockStream
from pi_sensenova.engine import Cancelled
from pi_sensenova.settings import Settings


class LoopedRuntimeTests(unittest.TestCase):
    def setUp(self):
        _, Model, _ = upstream()
        torch.manual_seed(17)
        self.model = Model(image_size=32, patch_size=8, hidden_size=32, num_heads=4,
                           head_dim=8, pca_channels=8, text_dim=16, text_preamble_depth=1,
                           loop_split=(1, 1, 1), use_xsa=True).eval()
        torch.nn.init.normal_(self.model.final.weight, std=0.02)
        self.text = torch.randn(1, 4, 16)
        self.mask = torch.ones(1, 4, dtype=torch.long)

    def test_official_euler_exact_parity_and_final_frame(self):
        from looped_dit.diffusion import euler_sample
        for loops in (1, 2, 4):
            for cfg in (1.0, 6.0):
                torch.manual_seed(42)
                expected = euler_sample(self.model, self.text, self.mask, 32, 3, cfg, 2, loops)
                frames, progress = [], []
                torch.manual_seed(42)
                actual = sample(self.model, self.text, self.mask, 32, 3, cfg, loops,
                                callback=lambda n, total: progress.append(n) or False,
                                preview_callback=lambda frame, n, total: frames.append(frame().clone()))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(frames[-1], actual, rtol=0, atol=0)
                self.assertEqual(len(frames), 3)
                self.assertEqual(progress[-1], 3)

    def test_cancel_restores_streaming_hooks(self):
        with self.assertRaises(Cancelled):
            with BlockStream(self.model, target="cpu", layers_attr="blocks"):
                sample(self.model, self.text, self.mask, 32, 3, 6, 4,
                       callback=lambda n, total: n == 1)
        self.assertTrue(all(not block._forward_pre_hooks and not block._forward_hooks for block in self.model.blocks))

    def test_direct_backend_rejects_invalid_loop_depth_before_loading(self):
        for depth in (0, 17, 1.5, float("nan"), True):
            with self.assertRaisesRegex(ValueError, "Loop depth"):
                LoopedEngine().generate(prompt="test", width=512, height=512, steps=1, cfg=1,
                                        seed=0, settings=Settings(loop_depth=depth))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_cuda_bf16_official_and_streamed_parity(self):
        from looped_dit.diffusion import euler_sample
        self.model.to(device="cuda", dtype=torch.bfloat16)
        text, mask = self.text.to(device="cuda", dtype=torch.bfloat16), self.mask.to("cuda")
        torch.manual_seed(42)
        expected = euler_sample(self.model, text, mask, 32, 3, 6, 2, 4)
        self.model.to("cpu")
        torch.manual_seed(42)
        with BlockStream(self.model, layers_attr="blocks"):
            actual = sample(self.model, text, mask, 32, 3, 6, 4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(all(t.device.type == "cpu" for t in self.model.parameters()))
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
