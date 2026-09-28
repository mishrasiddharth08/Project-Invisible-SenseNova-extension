"""Only Forge script entry point; no model loading during UI construction."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "vendor" / "deps"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from modules import scripts, script_callbacks
import gradio as gr
from pi_sensenova import run, selection, preset
from pi_sensenova.settings import DEFAULTS, UI_KEYS
from pi_sensenova import config

NATIVE = {}
QUALITY = {}


def bind_quality():
    for tab, (button, controls) in list(QUALITY.items()):
        if all(f"{tab}_{field}" in NATIVE for field in ("steps", "cfg_scale")):
            button.click(lambda: (50, 4.0, False, True, 3.0),
                         outputs=[NATIVE[f"{tab}_steps"], NATIVE[f"{tab}_cfg_scale"], *controls])
            button.interactive = True
            del QUALITY[tab]


def remember_component(component, **kwargs):
    name = kwargs.get("elem_id") or getattr(component, "elem_id", None)
    if name in {f"{tab}_{field}" for tab in ("txt2img", "img2img") for field in ("steps", "cfg_scale")}:
        NATIVE[name] = component
        bind_quality()


script_callbacks.on_after_component(remember_component)


class Script(scripts.Script):
    pi_sensenova = True

    def title(self):
        return "SenseNova — Project Invisible"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        suffix = "i2i" if is_img2img else "t2i"
        with gr.Accordion("SenseNova · Project Invisible", open=False, elem_id=f"pi_sensenova_{suffix}"):
            gr.Markdown("Upload a picture here, describe the change, then **Generate**. Editing is experimental; "
                        "denoising strength and masks are unsupported." if is_img2img else
                        "Choose a SenseNova checkpoint, write a prompt, then **Generate**. **For editing, use img2img.**")
            natural = gr.Checkbox(label="Natural photo details (adds texture instructions to your prompt)", value=True)
            quality = gr.Button("Use photo quality settings · 50 steps / CFG 4", interactive=False)
            with gr.Accordion("Advanced", open=False):
                degrid = gr.Checkbox(label="DeGrid · reduce tiny grid patterns (optional)", value=False)
                fast = gr.Checkbox(label="8-step speed adapter (requires its matching LoRA)", value=False)
                resources = gr.Textbox(label="Local config/tokenizer folder (optional if beside checkpoint)", value="")
                memory = gr.Dropdown(["Auto", "Full", "Fast offload", "Balanced", "Low VRAM"], value="Auto", label="Memory mode")
                adapter = gr.Textbox(label="Official 8-step adapter file (blank uses models/Lora/SenseNova)", value="")
                think = gr.Checkbox(label="Think before generating (adds latency)", value=False)
                shift = gr.Slider(0.1, 10, value=3.0, step=0.1, label="Timestep shift")
                gr.Markdown("Start at 1024×1024. SenseNova uses its own Euler sampler. Negative prompts become "
                            "‘Avoid’ instructions. Photo details cannot guarantee realism. Hires fix is unsupported.")
                unload = gr.Button("Unload SenseNova")
                status = gr.Textbox(label="Status", interactive=False)
                def release():
                    from pi_sensenova.engine import ENGINE
                    ENGINE.unload()
                    return "SenseNova unloaded."
                unload.click(release, outputs=status)
            with gr.Accordion("Get models · manual or automatic", open=False):
                from pi_sensenova import model_setup
                choices = list(model_setup.CATALOG)
                model = gr.Dropdown(choices, value=choices[0], label="Model")
                method = gr.Radio(["Manual (recommended)", "Automatic"], value="Manual (recommended)", label="Download method")
                guide = gr.Markdown(model_setup.instructions(choices[0]))
                download = gr.Button("Download selected model", visible=False)
                download_status = gr.Textbox(label="Download status", interactive=False)
                model.change(model_setup.instructions, inputs=model, outputs=guide)
                method.change(lambda value: gr.update(visible=value == "Automatic"), inputs=method, outputs=download)
                download.click(model_setup.download, inputs=model, outputs=download_status)
            tab = "img2img" if is_img2img else "txt2img"
            QUALITY[tab] = (quality, [fast, natural, shift])
            bind_quality()
        controls = [resources, memory, fast, adapter, natural, think, shift, degrid]
        assert len(controls) == len(UI_KEYS)
        for control, value in zip(controls, config.read()):
            control.value = value
            control.change(config.save, inputs=controls, outputs=[], queue=True, show_progress="hidden")
        return controls


run.install()
run.install_unload()
selection.install()
preset.install()
script_callbacks.on_app_started(run.bind_aliases)


def app_started(demo, app):
    from pi_sensenova.run import install_selection_release
    install_selection_release()
    if any(getattr(route, "path", "") == "/pi-sensenova/state" for route in app.routes):
        return
    @app.get("/pi-sensenova/state")
    def state(checkpoint: str = ""):
        # Resolve only registered dropdown entries; never accept arbitrary paths.
        from modules import sd_models
        from pi_sensenova.detect import is_ours
        ci = sd_models.checkpoint_aliases.get(checkpoint) or sd_models.checkpoints_list.get(checkpoint)
        from modules import shared
        return {"active": bool(ci and is_ours(ci.filename)),
                "progress": getattr(shared.state, "pi_sensenova_progress", None),
                "tab": getattr(shared.state, "pi_sensenova_tab", "txt2img")}


script_callbacks.on_app_started(app_started)
