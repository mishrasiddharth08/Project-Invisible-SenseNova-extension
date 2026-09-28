import gc
import sys
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from pi_sensenova.engine import Engine
from pi_sensenova.live import LiveProgress
from pi_sensenova import run


class ProgressReleaseTests(unittest.TestCase):
    def test_current_resets_overall_continues(self):
        state = types.SimpleNamespace(interrupted=False, skipped=False)
        opts = types.SimpleNamespace(live_previews_enable=False)
        first = LiveProgress(state, opts, 0, 4, 20)
        first.update(10, 20)
        self.assertEqual(state.pi_sensenova_progress['current'], .5)
        self.assertEqual(state.pi_sensenova_progress['overall'], .125)
        first.publish_final(None, 20)
        second = LiveProgress(state, opts, 1, 4, 20)
        second.update(0, 20)
        self.assertEqual(state.pi_sensenova_progress['current'], 0)
        self.assertEqual(state.pi_sensenova_progress['overall'], .25)

    def test_idle_release_drops_model_reference(self):
        engine = Engine()
        engine.model = torch.nn.Linear(2, 2)
        ref = weakref.ref(engine.model)
        engine.tokenizer = object()
        with patch('pi_sensenova.engine.torch.cuda.is_available', return_value=True), \
             patch('pi_sensenova.engine.torch.cuda.empty_cache') as empty:
            engine.request_unload()
        gc.collect()
        self.assertIsNone(ref())
        self.assertIsNone(engine.tokenizer)
        self.assertFalse(engine.release_requested.is_set())
        empty.assert_called_once()

    def test_selection_chains_original_once(self):
        calls = []
        info = types.SimpleNamespace(onchange=lambda: calls.append('original'))
        shared = types.SimpleNamespace(opts=types.SimpleNamespace(data_labels={'sd_model_checkpoint': info}))
        modules = types.ModuleType('modules')
        modules.shared = shared
        with patch.dict(sys.modules, {'modules': modules}), \
             patch.object(run.selection, 'selected', return_value=None), \
             patch('pi_sensenova.engine.ENGINE.request_unload') as release:
            run.install_selection_release()
            run.install_selection_release()
            info.onchange()
            release.assert_called_once()
        self.assertEqual(calls, ['original'])
