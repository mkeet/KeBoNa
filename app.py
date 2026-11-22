"""
app.py - the desktop application of the spell checker.

    python app.py

A text is typed or pasted, checked against the models of the chosen language, and each word
flagged is underlined with its suggestions. A new language can be added by uploading a corpus,
which is cleaned and trained in a separate process.
"""

import asyncio
import html
import json
import logging
import multiprocessing
import os
import re
import shutil
import sys
import tempfile
import zipfile
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path

from nicegui import app, ui

BUNDLED = hasattr(sys, "_MEIPASS")
PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
sys.path.insert(0, str(PROJECT_ROOT))

import setup
from src.core.spell_checker import DEFAULT_THRESHOLD, SpellChecker
from src.models.threshold_classifier import ThresholdClassifier
from src.models.trigram_model import TrigramModel

multiprocessing.freeze_support()

# models shipped in a packaged app are read-only, so what the user trains goes here
USER_DIR = Path.home() / ".spellchecker"
USER_MODELS = USER_DIR / "data" / "models"
USER_CORPORA = USER_DIR / "data" / "corpora"

# the packaged app trains a language by starting itself again with --train
if len(sys.argv) >= 3 and sys.argv[1] == "--train":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    if BUNDLED:
        setup.PROJECT_ROOT = USER_DIR
        (USER_DIR / "data" / "models").mkdir(parents=True, exist_ok=True)
    steps = setup.setup_language(sys.argv[2])
    sys.exit(0 if all(steps.values()) else 1)

ui.add_head_html("""
<style>
:root {
    --bg: #faf6f0;
    --surface: #fff9f2;
    --border: #d4b896;
    --text: #2c1a0e;
    --muted: #9a7355;
    --accent: #b5522a;
    --err-text: #c0392b;
    --chip-bg: #f0d9bc;
}

body.body--dark {
    --bg: #0f0f0f;
    --surface: #1a1a1a;
    --border: #3a2a2a;
    --text: #f0e8e8;
    --muted: #a08090;
    --accent: #e07a3f;
    --err-text: #ff6b6b;
    --chip-bg: #2a2a2a;
}

html, body { margin:0; padding:0; height:100%; background:var(--bg); color:var(--text); }
.nicegui-content { padding:0 !important; height:100vh; }

.sidebar {
    width:240px; min-width:240px; height:100vh; overflow-y:auto;
    background:var(--bg); border-right:1px solid var(--border);
    padding:1.2rem; display:flex; flex-direction:column; gap:0.8rem;
}
.main-col {
    flex:1; height:100vh; display:flex; flex-direction:column; overflow:hidden; background:var(--bg);
}
.chip {
    display:inline-flex; align-items:center; padding:4px 12px; border-radius:20px;
    font-size:0.75rem; font-weight:600; margin-right:8px; border:1px solid var(--border);
    background: var(--chip-bg); color: var(--text);
}
.sec-label {
    font-size:0.68rem; font-weight:600; text-transform:uppercase;
    letter-spacing:.06em; color:var(--muted); margin-bottom:4px;
}

#sc-editor:empty:before {
    content: "Type or paste text here... (Ctrl+Enter to check)";
    color: var(--muted);
    pointer-events: none;
    display: block;
}

.err-word {
    text-decoration: underline wavy var(--err-text);
    text-underline-offset: 3px;
    text-decoration-thickness: 1px;
    cursor: pointer;
    border-radius: 3px;
    background-color: transparent;
    transition: background-color 0.2s;
}

.err-word:hover {
    background-color: color-mix(in srgb, var(--err-text) 15%, transparent);
}

.sugg-popup {
    position: fixed;
    z-index: 9999;
    display: none;
    flex-direction: column;
    gap: 4px;
    padding: 10px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid var(--border);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(8px);
    min-width: 180px;
}
body.body--dark .sugg-popup {
    background: rgba(30, 30, 30, 0.95);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
}

.sugg-item {
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 500;
    color: var(--text);
    transition: all 0.15s ease;
    font-size: 0.95rem;
}
.sugg-item:hover {
    background: var(--accent);
    color: #fff;
    transform: translateX(2px);
}
.sugg-label {
    font-size: 0.7rem;
    color: var(--muted);
    margin-bottom: 4px;
    margin-top: 2px;
    padding-left: 6px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>

<script>
var _hideTimer = null;

function cancelHide() {
    if (_hideTimer) { clearTimeout(_hideTimer); _hideTimer = null; }
}

function scheduleHide() {
    _hideTimer = setTimeout(function() {
        var popup = document.getElementById('sc-popup');
        if (popup) popup.style.display = 'none';
    }, 200);
}

function showPopup(span, x, y) {
    var popup = document.getElementById('sc-popup');
    if (!popup) return;

    var sugs = [];
    try { sugs = JSON.parse(span.getAttribute('data-sugs') || '[]'); } catch(err) {}

    while (popup.firstChild) popup.removeChild(popup.firstChild);

    var label = document.createElement('div');
    label.className = 'sugg-label';
    label.textContent = 'SUGGESTIONS';
    popup.appendChild(label);

    if (sugs.length === 0) {
        var none = document.createElement('div');
        none.style.cssText = 'padding:6px 12px;color:var(--muted);font-size:0.9rem';
        none.textContent = 'No suggestions';
        popup.appendChild(none);
    } else {
        sugs.forEach(function(sug) {
            var item = document.createElement('div');
            item.className = 'sugg-item';
            item.textContent = sug;
            item.addEventListener('mousedown', function(ev) {
                ev.preventDefault();
                ev.stopPropagation();
                cancelHide();
                var textNode = document.createTextNode(sug);
                span.parentNode.replaceChild(textNode, span);
                popup.style.display = 'none';
                setTimeout(function() {
                    var btn = document.getElementById('check-btn');
                    if (btn) btn.click();
                }, 60);
            });
            popup.appendChild(item);
        });
    }

    var rect = span.getBoundingClientRect();
    popup.style.left = rect.left + 'px';
    popup.style.top  = (rect.bottom + 6) + 'px';
    popup.style.display = 'flex';
}

document.addEventListener('mouseover', function(e) {
    var span = e.target.closest('.err-word');
    var onPopup = e.target.closest('#sc-popup');

    if (span) {
        cancelHide();
        showPopup(span);
    } else if (onPopup) {
        cancelHide();
    } else {
        scheduleHide();
    }
});

document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        var btn = document.getElementById('check-btn');
        if (btn) btn.click();
    }
});
</script>
""", shared=True)


_checker = None


def get_checker():
    """Loads the shipped models and the user's own models on first use."""
    global _checker
    if _checker is None:
        _checker = SpellChecker()
        for md in (PROJECT_ROOT / "data" / "models", USER_MODELS):
            if md.exists():
                _checker.load_models(md)
    return _checker


def load_language(lang):
    """Adds the three models of one newly trained language to the loaded checker."""
    models_dir = USER_MODELS if BUNDLED else PROJECT_ROOT / "data" / "models"
    checker = get_checker()
    checker.suggestion_models[lang] = TrigramModel.load(models_dir / f"{lang}_word.pkl")
    checker.detection_models[lang] = TrigramModel.load(models_dir / f"{lang}_trigram.pkl")
    classifier = models_dir / f"{lang}_classifier.pkl"
    if classifier.exists():
        checker.threshold_classifiers[lang] = ThresholdClassifier.load(classifier)


def get_languages():
    """Returns the languages that have models, or an empty list when loading fails."""
    try:
        return get_checker().get_available_languages()
    except Exception:
        return []

@ui.page("/")
async def main():
    ui.colors(primary='#b5522a', secondary='#9a7355', accent='#e07a3f')

    state = {"lang": "", "pending": None, "errors": {}}
    langs = get_languages()
    NAMES = {"zulu": "isiZulu (SADiLaR)", "xhosa": "isiXhosa (SADiLaR)",
             "ndebele": "isiNdebele (SADiLaR)", "swati": "siSwati (SADiLaR)",
             "zu": "isiZulu (NCHLT)", "xh": "isiXhosa (NCHLT)", "nd": "isiNdebele (NCHLT)",
             "sw": "siSwati (NCHLT)", "sotho": "Sesotho", "tswana": "Setswana",
             "shona": "ciShona", "swahili": "Kiswahili", "lozi": "Silozi", "kikuyu": "Gikuyu"}

    with ui.row().style("width:100%;height:100vh;gap:0;flex-wrap:nowrap"):

        with ui.element("div").classes("sidebar"):
            ui.label("Spell Checker").style("font-weight:700;font-size:1rem;margin-bottom:.4rem")
            ui.separator().style("margin:.4rem 0")

            ui.label("Check text in").classes("sec-label")
            lang_opts = {l: NAMES.get(l, l.title()) for l in langs} if langs else {"": "No models"}
            lang_sel = ui.select(options=lang_opts, value=langs[0] if langs else "",
                                on_change=lambda e: state.update({"lang": e.value})).style("width:100%")
            if langs:
                state["lang"] = langs[0]

            ui.separator().style("margin:.6rem 0")

            ui.label("Add a language").classes("sec-label")
            new_lang = ui.input(label="Language name", placeholder="e.g. shona").style("width:100%")
            file_label = ui.label("No file selected").style("font-size:.75rem;color:var(--muted);margin:.4rem 0")

            async def handle_upload(e):
                content = await e.file.read()
                if state.get("pending"):
                    shutil.rmtree(state["pending"], ignore_errors=True)
                    state["pending"] = None
                pending = Path(tempfile.mkdtemp(prefix="spellchecker_"))

                if e.file.name.endswith('.zip'):
                    try:
                        with zipfile.ZipFile(BytesIO(content)) as zf:
                            txt_files = [f for f in zf.namelist() if f.endswith('.txt')
                                        and not any(p.startswith('__MACOSX') or p.startswith('.') for p in f.split('/'))]
                            if not txt_files:
                                ui.notify("The zip has no .txt files in it.", type="warning")
                                shutil.rmtree(pending, ignore_errors=True)
                                return
                            extracted = 0
                            for txt_file in txt_files:
                                filename = os.path.basename(txt_file)
                                if filename:
                                    (pending / filename).write_bytes(zf.read(txt_file))
                                    extracted += 1
                            if extracted == 0:
                                ui.notify("The zip has no usable .txt files in it.", type="warning")
                                shutil.rmtree(pending, ignore_errors=True)
                                return
                            label = f"{extracted} files ready" if extracted > 1 else f"{os.path.basename(txt_files[0])} ready"
                    except zipfile.BadZipFile:
                        ui.notify("That zip file could not be opened.", type="negative")
                        shutil.rmtree(pending, ignore_errors=True)
                        return
                else:
                    (pending / e.file.name).write_bytes(content)
                    label = f"{e.file.name} ready"

                state["pending"] = pending
                file_label.set_text(label)
                train_btn.enable()
                ui.notify(label, type="positive")

            ui.upload(label="Corpus: .txt files or a .zip", on_upload=handle_upload,
                    auto_upload=True, max_files=1).props("flat bordered accept='.txt,.zip'").style("width:100%")

            pbar = ui.linear_progress(value=0).style("width:100%;margin-top:.5rem")
            pstep = ui.label("").style("font-size:.73rem;color:var(--muted)")

            async def do_train():
                lang = new_lang.value.strip().lower()
                if not lang:
                    ui.notify("Type a name for the language first.", type="warning")
                    return
                if not state.get("pending"):
                    ui.notify("Upload a corpus first.", type="warning")
                    return

                dest = (USER_CORPORA if BUNDLED else PROJECT_ROOT/"data"/"corpora")/lang/"raw"
                dest.mkdir(parents=True, exist_ok=True)
                for f in state["pending"].iterdir():
                    shutil.move(str(f), dest / f.name)
                shutil.rmtree(state["pending"], ignore_errors=True)
                state["pending"] = None

                train_btn.disable()
                pbar.set_value(0.1)
                pstep.set_text("Training...")

                cmd = ([sys.executable, "--train", lang] if BUNDLED
                       else [sys.executable, str(PROJECT_ROOT / "setup.py"), "--lang", lang])
                STEPS = {1: "Cleaning corpus", 2: "Splitting 80/10/10", 3: "Training word model",
                         4: "Building trigram corpus", 5: "Training trigram model",
                         6: "Training classifier"}
                proc = await asyncio.create_subprocess_exec(
                    *cmd, cwd=str(PROJECT_ROOT),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                state["proc"] = proc
                tail = []
                while True:
                    raw = await proc.stdout.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", "replace").rstrip()
                    tail.append(line)
                    del tail[:-40]
                    m = re.search(r"Step (\d) - ", line)
                    if m:
                        i = int(m.group(1))
                        pbar.set_value(i / 6)
                        pstep.set_text(f"{i}/6  {STEPS.get(i, '')}")
                await proc.wait()
                state["proc"] = None
                if proc.returncode != 0:
                    pbar.set_value(0)
                    pstep.set_text("Training failed")
                    print("\n".join(tail), file=sys.stderr)
                    ui.notify("Training failed. The app is still running - you can retry.",
                              type="negative", timeout=8000)
                    train_btn.enable()
                    return
                pbar.set_value(1.0)
                pstep.set_text("Done")
                await asyncio.sleep(0.3)

                await asyncio.get_event_loop().run_in_executor(None, load_language, lang)
                new_langs = get_languages()
                lang_sel.options = {l: NAMES.get(l,l.title()) for l in new_langs}
                lang_sel.value = lang
                state["lang"] = lang
                lang_sel.update()

                train_btn.enable()
                pbar.set_value(0)
                pstep.set_text("")
                ui.notify(f"{lang} is trained and ready to use.", type="positive")

            train_btn = ui.button("Train model", on_click=do_train).style(
                "width:100%;border-radius:8px;margin-top:.5rem;font-weight:bold")
            train_btn.disable()

            ui.separator().style("margin:.6rem 0")
            dm = ui.dark_mode()

            def toggle_theme():
                dm.value = not dm.value
                toggle_btn.props(f'icon={"dark_mode" if dm.value else "light_mode"}')

            toggle_btn = ui.button(icon="light_mode", on_click=toggle_theme).props('round flat').style(
                "width:100%;border-radius:8px").tooltip("Toggle theme")

        with ui.element("div").classes("main-col"):
            with ui.element("div").style("padding:1.5rem;display:flex;flex-direction:column;gap:1rem;height:100%"):

                with ui.row().style("width:100%; justify-content:space-between; align-items:flex-end;"):
                    stats_row = ui.row().style("flex-wrap:wrap; gap:8px; display:none;")

                    with ui.row().style("gap:8px; margin-left:auto;"):
                        check_btn = ui.button("Check Spelling", icon="spellcheck").props(
                            'id=check-btn'
                        ).style("border-radius:8px; font-weight:bold; padding:8px 18px;")

                ui.html('<div id="sc-popup" class="sugg-popup"></div>')

                ui.element('div').props('id="sc-editor" contenteditable="true" spellcheck="false"').classes(
                    'w-full bg-surface text-text rounded-xl border outline-none'
                ).style(
                    'flex: 1; padding: 1.5rem; height: 100%; min-height: 500px; font-size: 1.15rem; '
                    'line-height: 1.7; white-space: pre-wrap; overflow-y: auto; word-wrap: break-word;'
                )

                async def do_check():
                    lang = state["lang"]
                    if not lang:
                        ui.notify("Select language first", type="warning")
                        return

                    text = await ui.run_javascript('document.getElementById("sc-editor").innerText')
                    text = text.strip()
                    if not text:
                        ui.notify("Enter text to check", type="warning")
                        return

                    check_btn.disable()
                    check_btn.set_text("Checking...")

                    try:
                        checker = get_checker()

                        if lang not in checker.detection_models:
                            ui.notify(f"No model for '{lang}'", type="warning")
                            return

                        has_lr = lang in checker.threshold_classifiers
                        lr_str = f"NO   (fixed threshold {DEFAULT_THRESHOLD})"
                        if has_lr:
                            try:
                                clf = checker.threshold_classifiers[lang]
                                boundary = -clf._clf.intercept_[0] / clf._clf.coef_[0][0]
                                lr_str = f"YES  (tau={boundary:.3f})"
                            except Exception:
                                lr_str = "YES  (boundary unavailable)"

                        print("\n" + "-"*52)
                        print(f"  SPELL CHECK")
                        print(f"  Language : {NAMES.get(lang, lang.upper())}")
                        print(f"  LR Threshold : {lr_str}")
                        print(f"  Words    : {len(text.split())}")
                        print("-"*52)

                        result = await asyncio.get_event_loop().run_in_executor(None,
                            lambda: checker.check(text, language=lang, threshold=None, include_suggestions=True))

                        stats_row.style("display:flex")
                        stats_row.clear()
                        with stats_row:
                            n = result["total_words"]
                            ec = len(result["errors"])
                            cc = "border-color:var(--err-text);color:var(--err-text)" if ec else "border-color:#a5d6a7;color:#3b6d11"
                            ui.html(f'<span class="chip">{n} words</span>')
                            ui.html(f'<span class="chip" style="{cc}">{ec} error{"s" if ec!=1 else ""}</span>')


                        n  = result["total_words"]
                        ec = len(result["errors"])
                        print(f"  Result   : {n} words checked, {ec} error{'s' if ec!=1 else ''}")
                        if result["errors"]:
                            print(f"  Errors   :")
                            for err in result["errors"][:10]:
                                sugs_preview = [s[0] if isinstance(s,(list,tuple)) else s for s in err.get("suggestions",[])[:3]]
                                print(f"      {err['word']!r:20s} -> {sugs_preview}")
                            if len(result["errors"]) > 10:
                                print(f"    ... and {len(result['errors'])-10} more")
                        else:
                            print(f"  Result   :  All words OK")
                        print("-"*52 + "\n")

                        error_map = {}
                        for e in result.get("errors", []):
                            clean_w = e["word"].lower()
                            sugs = [s[0] if isinstance(s, (list, tuple)) else s for s in e.get("suggestions", [])]
                            error_map[clean_w] = sugs[:5]

                        parts = re.split(r"([A-Za-z0-9_'-]+)", text)
                        html_parts = []
                        current_idx = 0

                        for part in parts:
                            if not part:
                                continue
                            clean = part.lower()
                            if clean in error_map:
                                sugs_attr = html.escape(json.dumps(error_map[clean]))
                                html_parts.append(
                                    f'<span class="err-word" '
                                    f'data-word="{html.escape(clean)}" '
                                    f'data-idx="{current_idx}" '
                                    f'data-sugs=\'{sugs_attr}\'>'
                                    f'{html.escape(part)}</span>'
                                )
                            else:
                                html_parts.append(html.escape(part))
                            current_idx += len(part)

                        final_html = "".join(html_parts).replace('\n', '<br>')
                        final_html_js = final_html.replace('`', '\\`').replace('${', '\\${')
                        await ui.run_javascript(
                            f'document.getElementById("sc-editor").innerHTML = `{final_html_js}`;'
                        )

                    except Exception as ex:
                        ui.notify(f"Error: {ex}", type="negative")
                    finally:
                        check_btn.enable()
                        check_btn.set_text("Check Spelling")

                check_btn.on_click(do_check)


def _gtk_stack_available() -> bool:
    """Returns True when gi, webview and the Gtk and WebKit2 typelibs are present, without importing them."""
    try:
        if find_spec("gi") is None or find_spec("webview") is None:
            return False
    except Exception:
        return False
    dirs = [d for d in os.environ.get("GI_TYPELIB_PATH", "").split(os.pathsep) if d]
    dirs += ["/usr/lib/x86_64-linux-gnu/girepository-1.0", "/usr/lib/girepository-1.0",
             "/usr/lib64/girepository-1.0"]
    def has(stem: str) -> bool:
        return any((Path(d) / stem).exists() for d in dirs)
    return has("Gtk-3.0.typelib") and (has("WebKit2-4.1.typelib") or has("WebKit2-4.0.typelib"))


def _use_native() -> bool:
    """
    Decides between a native window and the browser. Windows and macOS ship a web view that
    pywebview finds by itself. On Linux the GTK bindings live in the system site-packages and
    not in the virtual environment, so that path is added and the gtk backend is named.
    SPELLCHECKER_NATIVE=0 forces the browser.
    """
    if os.environ.get("SPELLCHECKER_NATIVE") == "0":
        return False
    if sys.platform.startswith("linux"):
        for cand in ("/usr/lib/python3/dist-packages", "/usr/lib/python3.12/site-packages"):
            if cand not in sys.path and Path(cand).exists():
                sys.path.append(cand)
        # Gtk must not be imported here because it breaks the window process that NiceGUI forks
        if not _gtk_stack_available():
            return False
        app.native.start_args["gui"] = "gtk"
        return True
    return find_spec("webview") is not None

_NATIVE = _use_native()
ui.run(title="Spell Checker", port=int(os.environ.get("SPELLCHECKER_PORT", 8080)),
       reload=False, favicon="O", dark=False,
       native=_NATIVE, show=not _NATIVE,
       window_size=(1100, 800) if _NATIVE else None)
