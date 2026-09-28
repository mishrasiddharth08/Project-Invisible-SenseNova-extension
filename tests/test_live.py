import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi_sensenova.live import LiveProgress
from pi_sensenova.engine import progress, Cancelled


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.frames = []
        self.state = types.SimpleNamespace(interrupted=False, skipped=False,
            assign_current_image=self.frames.append)
        self.opts = types.SimpleNamespace(live_previews_enable=True, show_progress_every_n_steps=1)

    def test_preview_is_bounded_throttled_and_does_not_change_sampler(self):
        tensor = torch.rand(1, 3, 768, 1024) * 2 - 1
        before = tensor.clone()
        live = LiveProgress(self.state, self.opts, 0, 2, 20)
        with patch('pi_sensenova.live.time.monotonic', side_effect=[10, 10.1, 11.1, 11.2]):
            for step in (1, 2, 3, 20):
                live.preview(tensor, step, 20)
        self.assertEqual(len(self.frames), 3)
        self.assertEqual(self.frames[0].size, (512, 384))
        self.assertTrue(torch.equal(tensor, before))
        self.assertEqual(self.state.current_image_sampling_step, 20)

    def test_preview_throttle_is_one_second_but_final_step_is_forced(self):
        tensor = torch.zeros(1, 3, 2, 2)
        live = LiveProgress(self.state, self.opts, 0, 1, 4)
        with patch('pi_sensenova.live.time.monotonic', side_effect=[20, 20.9, 21.0]):
            live.preview(tensor, 1, 4)
            live.preview(tensor, 2, 4)
            live.preview(tensor, 4, 4)
        self.assertEqual(len(self.frames), 2)
        self.assertEqual(self.state.current_image_sampling_step, 4)

    def test_deferred_final_skips_raw_frame_then_publishes_processed_image(self):
        tensor = torch.zeros(1, 3, 2, 2)
        processed = object()
        live = LiveProgress(self.state, self.opts, 0, 1, 4, defer_final=True)
        live.update(4, 4)
        live.preview(tensor, 4, 4)
        self.assertEqual(self.frames, [])
        self.assertEqual(self.state.sampling_step, 3)
        self.assertIn("Applying DeGrid", self.state.textinfo)
        live.publish_final(processed, 4)
        self.assertIs(self.frames[0], processed)
        self.assertEqual(self.state.sampling_step, 4)
        self.assertEqual(self.state.current_image_sampling_step, 4)

    def test_publish_final_completes_progress_when_previews_are_disabled(self):
        self.opts.live_previews_enable = False
        live = LiveProgress(self.state, self.opts, 0, 1, 4, defer_final=True)
        live.publish_final(object(), 4)
        self.assertEqual(self.frames, [])
        self.assertEqual(self.state.sampling_step, 4)

    def test_preview_ignores_large_forge_step_interval_when_enabled(self):
        tensor = torch.zeros(1, 3, 2, 2)
        self.opts.show_progress_every_n_steps = 100
        live = LiveProgress(self.state, self.opts, 0, 1, 4)
        with patch('pi_sensenova.live.time.monotonic', side_effect=[30, 31.0]):
            live.preview(tensor, 1, 4)
            live.preview(tensor, 2, 4)
        self.assertEqual(len(self.frames), 2)

    def test_preview_respects_disabled_and_batch_only_settings(self):
        live = LiveProgress(self.state, self.opts, 0, 1, 20)
        self.opts.live_previews_enable = False
        live.preview(None, 1, 20)
        self.opts.live_previews_enable = True
        self.opts.show_progress_every_n_steps = -1
        live.preview(None, 1, 20)
        self.assertEqual(self.frames, [])

    def test_progress_and_console_follow_steps_and_close_on_stop(self):
        with patch('pi_sensenova.live.tqdm') as factory:
            factory.return_value.n = 0
            with LiveProgress(self.state, self.opts, 1, 3, 50) as live:
                self.assertFalse(live.update(7, 50))
                self.assertEqual(self.state.sampling_step, 7)
                self.assertIn('image 2/3', self.state.textinfo)
                self.state.interrupted = True
                self.assertTrue(live.update(8, 50))
            factory.return_value.close.assert_called_once()

    def test_real_step_output_delivered_without_changing_return(self):
        output = torch.zeros(1, 3, 2, 2)
        model = types.SimpleNamespace(unpatchify=lambda: output)
        original = model.unpatchify
        seen = []
        with progress(model, 2, lambda *_: False, lambda value, step, total: seen.append((value, step, total))):
            self.assertIs(model.unpatchify(), output)
            self.assertIs(model.unpatchify(), output)
        self.assertEqual([item[1] for item in seen], [1, 2])
        self.assertIs(model.unpatchify, original)

    def test_evolving_sample_and_predictor_restoration(self):
        z = torch.ones(1, 3, 2, 2)
        velocity = torch.full_like(z, 2)
        model = types.SimpleNamespace(unpatchify=lambda value: value,
                                      _t2i_predict_v=lambda *args: velocity)
        original = model._t2i_predict_v
        seen = []
        with progress(model, 2, lambda *_: False, lambda frame, *_: seen.append(frame())):
            self.assertIs(model._t2i_predict_v(None,None,None,None,torch.tensor(.25),z), velocity)
            actual = z + .1 * velocity
            self.assertIs(model.unpatchify(actual), actual)
        self.assertIs(seen[0], actual)
        self.assertTrue(torch.equal(z, torch.ones_like(z)))
        self.assertIs(model._t2i_predict_v, original)
        with self.assertRaises(Cancelled):
            with progress(model, 2, lambda step, _: step == 1, lambda *args: self.fail('Stopped preview')):
                model.unpatchify(z)
        self.assertIs(model._t2i_predict_v, original)


if __name__ == '__main__':
    unittest.main()
