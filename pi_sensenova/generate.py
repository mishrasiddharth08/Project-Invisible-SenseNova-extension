"""Resolved settings -> official generation -> immediate Forge saving."""
import json
from pathlib import Path

from .engine import ENGINE, LOCK, Cancelled
from .settings import compose, dimensions
from .live import LiveProgress
from .detect import is_looped


def generate(p, checkpoint, settings):
    from modules import shared, processing, images as host_images
    try:
        from backend import memory_management
    except (ImportError, AttributeError) as error:
        raise RuntimeError("Forge's memory interface is unavailable; SenseNova cannot safely take the GPU.") from error

    with LOCK:
        looped = is_looped(checkpoint)
        if looped:
            from .looped import LOOPED_ENGINE
            engine = LOOPED_ENGINE
        else:
            engine = ENGINE
        model_label = "Looped-DiT" if looped else "SenseNova"
        if getattr(p, "enable_hr", False):
            raise ValueError("SenseNova uses native output resolution. Disable Hires fix and set the desired width/height.")
        if getattr(p, "restore_faces", False):
            raise ValueError("Disable face restoration for SenseNova; it can replace natural skin texture.")
        if getattr(p, "image_mask", None) is not None:
            raise ValueError("SenseNova supports instruction editing; masked inpainting is not implemented.")
        width, height = dimensions(p.width, p.height)
        if looped and (width, height) != (512, 512):
            raise ValueError("Official Looped-DiT requires 512×512. Use Looped-DiT settings.")
        count = int(p.batch_size) * int(p.n_iter)
        if count < 1:
            raise ValueError("Image count must be positive.")
        processing.fix_seed(p)
        base_seed = int(p.seed)
        steps, cfg = int(p.steps), float(p.cfg_scale)
        if looped and (settings.fast or settings.think or settings.adapter):
            raise ValueError("Disable U1.5 fast adapter and Think mode, and clear the adapter path for Looped-DiT.")
        if settings.fast:
            if steps != 8 or not 1 <= cfg <= 3:
                raise ValueError("The official Fast adapter needs Steps 8 and CFG 1–3. Set these in the normal Forge controls.")
        if steps < 1 or not 1 <= cfg <= 20:
            raise ValueError("SenseNova requires at least one step and CFG between 1 and 20.")
        sources = getattr(p, "init_images", None) or []
        if sources and looped:
            raise ValueError("Looped-DiT supports text-to-image only. Select SenseNova U1.5 for editing.")
        if sources and settings.fast:
            raise ValueError("The official 8-step adapter is text-to-image only. Disable Fast for editing and use 50 steps / CFG 4.")
        img_type = getattr(processing, "StableDiffusionProcessingImg2Img", ())
        txt_type = getattr(processing, "StableDiffusionProcessingTxt2Img", ())
        if isinstance(p, img_type) and not sources:
            raise ValueError("Add a source image in the img2img tab before editing.")
        if isinstance(p, txt_type) and sources:
            raise ValueError("Use the img2img tab for image editing.")
        if len(sources) > 1:
            raise ValueError("Use one source image for this SenseNova editing interface.")
        raw_prompts = p.prompt if isinstance(p.prompt, list) else [p.prompt] * count
        negatives = p.negative_prompt if isinstance(p.negative_prompt, list) else [p.negative_prompt] * count
        if len(raw_prompts) != count or len(negatives) != count:
            raise ValueError("Prompt list length must match the requested image count.")
        styles = getattr(p, "styles", [])
        if styles:
            raw_prompts = [shared.prompt_styles.apply_styles_to_prompt(raw, styles) for raw in raw_prompts]
            negatives = [shared.prompt_styles.apply_negative_styles_to_prompt(neg, styles) for neg in negatives]
        prompts = [compose(raw, neg, settings.natural) for raw, neg in zip(raw_prompts, negatives)]
        state = shared.state
        if state.interrupted or state.skipped:
            raise Cancelled("Generation was stopped before model loading.")
        state.job_count = count
        state.sampling_steps = steps
        state.sampling_step = 0
        state.textinfo = f"Loading {model_label} checkpoint…"
        state.pi_sensenova_progress = dict(current=0.0, overall=0.0, image=1, count=count)
        state.pi_sensenova_tab = "img2img" if sources else "txt2img"
        memory_management.unload_all_models()
        if looped:
            ENGINE.unload()
        else:
            import sys
            other = sys.modules.get("pi_sensenova.looped")
            if other is not None:
                other.LOOPED_ENGINE.unload()
        engine.load(checkpoint, settings)
        output, infos, seeds, completed_prompts = [], [], [], []
        state.job_count = count
        p.sd_model_name = Path(checkpoint).parent.name if Path(checkpoint).name == "config.json" else Path(checkpoint).stem
        p.sd_model_hash = p.sd_vae_name = p.sd_vae_hash = None
        p.is_using_inpainting_conditioning = False
        print(f"[Invisible-SenseNova] {count} image(s): batch size {p.batch_size} × count {p.n_iter}; "
              f"serialized for memory efficiency. {width}×{height}, {steps} steps, CFG {cfg}. "
              f"Fast={settings.fast}, Natural photo={settings.natural}. Official Euler flow sampler.")

        for index, prompt in enumerate(prompts):
            if state.interrupted:
                break
            if state.skipped:
                state.skipped = False
                state.nextjob()
                continue
            seed = (base_seed + index) % 4294967294
            state.job = f"SenseNova {index + 1}/{count}"
            try:
                with LiveProgress(state, shared.opts, index, count, steps, defer_final=True) as live:
                    generated = engine.generate(prompt=prompt, width=width, height=height, steps=steps,
                                                cfg=cfg, seed=seed, settings=settings,
                                                source=sources[0] if sources else None, callback=live.update,
                                                preview_callback=live.preview)
            except Cancelled:
                if state.skipped and not state.interrupted:
                    state.skipped = False
                    state.nextjob()
                    if index + 1 < count:
                        engine.load(checkpoint, settings)
                    continue
                break
            except Exception as error:
                # Previously saved complete images survive; never retry behind the user's back.
                if not output:
                    raise
                p.comments.append(f"SenseNova stopped after {len(output)} saved image(s): {error}")
                break
            for image in generated:
                if image.size != (width, height):
                    raise RuntimeError(f"Official output size {image.size} differs from requested {(width, height)}.")
                if settings.degrid:
                    state.textinfo = "Finishing image · DeGrid"
                    from .degrid import apply
                    image = apply(image, enabled=True)
                info = (f"{prompt}\nSteps: {steps}, Sampler: {model_label} Euler, CFG scale: {cfg}, "
                        f"Seed: {seed}, Size: {width}x{height}, Model: {p.sd_model_name}, "
                        f"SenseNova fast: {settings.fast}, Natural photo: {settings.natural}, "
                        f"Timestep shift: {settings.shift}, Think: {settings.think}, Memory: {engine.mode}, "
                        f"DeGrid: {settings.degrid}")
                if looped:
                    info = (f"{prompt}\nSteps: {steps}, Sampler: Looped-DiT Euler, CFG scale: {cfg}, "
                            f"Seed: {seed}, Size: {width}x{height}, Model: {p.sd_model_name}, "
                            f"Loop depth: {int(settings.loop_depth)}, Memory: {engine.mode}, "
                            f"Natural photo: {settings.natural}, DeGrid: {settings.degrid}, "
                            f"Runtime: {engine.metrics.get('upstream_revision', '')}")
                image.info["parameters"] = info
                if shared.opts.samples_save and not p.do_not_save_samples:
                    host_images.save_image(image, p.outpath_samples, "", seed=seed, prompt=prompt,
                                           extension=shared.opts.samples_format, info=info, p=p)
                output.append(image)
                infos.append(info)
                seeds.append(seed)
                completed_prompts.append(prompt)
                live.publish_final(image, steps)
            state.nextjob()
        p.sampler_name = f"{model_label} Euler"
        p.all_prompts = completed_prompts or prompts[:1]
        p.all_seeds = seeds or [base_seed]
        return processing.Processed(p, output, seed=seeds[0] if seeds else base_seed,
                                    info=infos[0] if infos else "SenseNova stopped; no complete image.",
                                    all_prompts=completed_prompts or prompts[:1], all_seeds=seeds or [base_seed],
                                    all_subseeds=[0] * max(1, len(output)), infotexts=infos or ["Stopped."])
