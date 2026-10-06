// Display-only integration. Restore the native control whenever SenseNova is inactive.
(() => {
    let signature = null;
    let timer = null;
    let request = 0;
    let pending = false;
    let lastPoll = 0;
    const root = () => typeof gradioApp === "function" ? gradioApp() : document;
    async function refresh() {
        const checkpoint = root().querySelector("#setting_sd_model_checkpoint");
        if (!checkpoint) return;
        const next = checkpoint.querySelector("input")?.value || "";
        if (pending) return;
        if (next === signature && Date.now() - lastPoll < 1000) return;
        lastPoll = Date.now();
        pending = true;
        const changed = next !== signature;
        signature = next;
        const current = ++request;
        if (changed) document.body.classList.remove("pi-sensenova-active", "pi-looped-active");
        try {
            const response = await fetch("/pi-sensenova/state?checkpoint=" + encodeURIComponent(next), {cache: "no-store"});
            if (!response.ok) throw new Error("SenseNova state unavailable");
            const state = await response.json();
            if ((checkpoint.querySelector("input")?.value || "") !== next) { signature = null; return; }
            if (current === request) {
                document.body.classList.toggle("pi-sensenova-active", state.active === true);
                document.body.classList.toggle("pi-looped-active", state.active === true && state.looped === true);
                for (const tab of ["txt2img", "img2img"]) {
                    let box = root().querySelector(`#sn-progress-${tab}`);
                    const gallery = root().querySelector(`#${tab}_gallery_container`);
                    if (!box && gallery) {
                        box = document.createElement("div");
                        box.id = `sn-progress-${tab}`;
                        box.className = "sn-dual-progress";
                        for (const label of ["Current image", "Overall generation"]) {
                            const row = document.createElement("div");
                            const title = document.createElement("span");
                            title.textContent = label;
                            const bar = document.createElement("progress");
                            bar.max = 1;
                            bar.value = 0;
                            bar.setAttribute("aria-label", label);
                            row.append(title, bar);
                            box.append(row);
                        }
                        gallery.before(box);
                    }
                    if (!box) continue;
                    box.hidden = !state.active || !state.progress || state.tab !== tab;
                    if (!box.hidden) {
                        const values = [state.progress.current, state.progress.overall];
                        box.querySelectorAll("progress").forEach((bar, i) => {
                            bar.value = Math.max(0, Math.min(1, values[i] || 0));
                            bar.previousElementSibling.textContent = `${i ? "Overall generation" : "Current image"}: ${Math.round(bar.value * 100)}% · Image ${state.progress.image}/${state.progress.count}`;
                        });
                    }
                }
            }
        } catch (_) {
            signature = null;
            if (current === request) document.body.classList.remove("pi-sensenova-active", "pi-looped-active");
        } finally {
            pending = false;
        }
    }
    function schedule() {
        if (timer !== null) return;
        timer = setTimeout(() => { timer = null; refresh(); }, 150);
    }
    if (typeof onUiLoaded === "function") onUiLoaded(schedule);
    if (typeof onUiUpdate === "function") onUiUpdate(schedule);
    // Input values can change without a DOM mutation. This compares locally and
    // only contacts the backend when the displayed selection changes.
    setInterval(schedule, 1000);
    document.addEventListener("change", event => {
        if (event.target.closest?.("#setting_sd_model_checkpoint")) { signature = null; schedule(); }
    });
})();
