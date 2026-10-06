"""Native Forge step reporting and bounded pixel-space live previews."""
import time

import torch
from tqdm.auto import tqdm
from PIL import Image, ImageFilter

from .engine import to_images


def reveal(image, step, total):
    """Display-only reveal; the final image bypasses all blending and blur."""
    if step >= total:
        return image
    fraction = max(0.0, step / max(1, total))
    alpha = fraction * fraction * (3 - 2 * fraction)
    soft = image.filter(ImageFilter.GaussianBlur(8 * (1 - alpha)))
    return Image.blend(Image.new("RGB", image.size, (128, 128, 128)), soft, alpha)


class LiveProgress:
    def __init__(self, state, opts, index, count, steps, defer_final=False):
        self.state, self.opts = state, opts
        self.index, self.count, self.steps = index, count, steps
        self.defer_final = bool(defer_final)
        self.bar = None
        self.last_time = float("-inf")
        self.last_step = 0
        self.preview_failed = False

    def __enter__(self):
        self.state.current_latent = None  # Pixel-space model: never invoke a VAE.
        self.state.sampling_step = self.state.preview_step = 0
        self.state.sampling_steps = self.steps
        self.bar = tqdm(total=self.steps, desc=f"SenseNova {self.index + 1}/{self.count}",
                        unit="step", mininterval=0.25)
        self.update(0, self.steps)
        self.preview(torch.zeros(1, 3, 256, 256), 0, self.steps)
        return self

    def __exit__(self, *unused):
        self.bar.close()

    def update(self, step, total):
        self.state.sampling_steps = total
        visible_step = max(0, total - 1) if self.defer_final and step >= total else step
        self.state.sampling_step = visible_step
        current = min(1.0, max(0.0, visible_step / max(1, total)))
        self.state.pi_sensenova_progress = dict(current=current, overall=(self.index + current) / max(1, self.count),
                                               image=self.index + 1, count=self.count)
        self.state.textinfo = (
            f"SenseNova image {self.index + 1}/{self.count} · Finishing image…"
            if self.defer_final and step >= total else
            f"SenseNova image {self.index + 1}/{self.count} · Step {step}/{total}"
        )
        if self.bar is not None and step > self.bar.n:
            self.bar.update(step - self.bar.n)
        return self.state.interrupted or self.state.skipped

    @torch.inference_mode()
    def preview(self, tensor, step, total):
        if self.preview_failed or not getattr(self.opts, "live_previews_enable", True):
            return
        if self.defer_final and step >= total:
            return
        # SenseNova owns its expensive pixel preview cadence. Forge's step
        # interval can disable previews, but must not delay them beyond the
        # one-second UI throttle when previews are enabled.
        interval = int(getattr(self.opts, "show_progress_every_n_steps", 1))
        if interval == -1 or self.state.interrupted or self.state.skipped:
            return
        now = time.monotonic()
        # Keep browser/UI updates to at most one per second. The completed
        # frame is always allowed through so the gallery cannot finish stale.
        if step != total and now - self.last_time < 1.0:
            return
        try:
            # A detached display copy of the actual evolving RGB sample.
            # Downsample before CPU transfer; never change the sampler's tensor.
            frame = (tensor() if callable(tensor) else tensor)[:1].detach()
            height, width = frame.shape[-2:]
            if max(height, width) > 512:
                scale = 512 / max(height, width)
                frame = torch.nn.functional.interpolate(frame.float(),
                    size=(max(1, round(height * scale)), max(1, round(width * scale))), mode="area")
            image = to_images(frame)[0]
            image = reveal(image, step, total)
            self.state.assign_current_image(image)
            self.state.preview_step = self.state.current_image_sampling_step = step
            self.last_time, self.last_step = now, step
        except Exception as error:
            self.preview_failed = True
            print(f"[Invisible-SenseNova] Live preview unavailable; sampling continues: {error}")

    def publish_final(self, image, total):
        """Publish the exact processed output as the final Forge preview."""
        if getattr(self.opts, "live_previews_enable", True):
            self.state.assign_current_image(image)
            self.state.preview_step = self.state.current_image_sampling_step = total
        self.state.sampling_steps = total
        self.state.sampling_step = total
        self.state.pi_sensenova_progress = dict(current=1.0, overall=(self.index + 1) / max(1, self.count),
                                               image=self.index + 1, count=self.count)
        self.state.textinfo = f"SenseNova image {self.index + 1}/{self.count} · Step {total}/{total}"
