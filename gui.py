import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
import shutil
import threading
import tempfile

DATASET = "dataset"
OUTPUT  = "output"
KEY_DB  = "key_db.txt"

BG      = "#f0f7f0"
PANEL   = "#ffffff"
BORDER  = "#b2d8b2"
ACCENT  = "#2e7d32"
ACCENT2 = "#1b5e20"
ACCENT3 = "#c62828"
ACCENT4 = "#4527a0"
ACCENT5 = "#e65100"
FG      = "#1b2e1b"
FG_DIM  = "#4a6a4a"
HOVER   = "#e0f0e0"
LOG_FG  = "#1b5e20"


def get_people():
    if not os.path.exists(DATASET):
        return []
    return sorted(d for d in os.listdir(DATASET)
                  if os.path.isdir(os.path.join(DATASET, d)))


def get_available_eyes(person):
    eyes = []
    for e in ["L", "R"]:
        if os.path.isdir(os.path.join(DATASET, person, e)):
            eyes.append(e)
    return eyes


def clear_output():
    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)
    os.makedirs(OUTPUT, exist_ok=True)


def ensure_output_dirs():
    for sub in ("norm", "mask", "seg", "code"):
        os.makedirs(os.path.join(OUTPUT, sub), exist_ok=True)


def write_key_db(person, eye, key):
    entry_id = f"{person}_{eye}"
    lines = []
    if os.path.exists(KEY_DB):
        with open(KEY_DB) as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                name, _ = line.split(":", 1)
                if name != entry_id:
                    lines.append(line)
    lines.append(f"{entry_id}:{key}")
    with open(KEY_DB, "w") as f:
        f.write("\n".join(lines) + "\n")


def styled_btn(parent, text, cmd, color=ACCENT, padx=14, pady=7, font_size=9):
    btn = tk.Label(parent, text=text, bg=color, fg="#ffffff",
                   font=("Courier New", font_size, "bold"),
                   cursor="hand2", padx=padx, pady=pady, relief="flat")
    btn.bind("<Button-1>", lambda e: cmd())
    btn.bind("<Enter>",    lambda e: btn.config(bg=_darken(color)))
    btn.bind("<Leave>",    lambda e: btn.config(bg=color))
    return btn


def _darken(hex_color):
    r, g, b = (int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(
        max(0, r-28), max(0, g-28), max(0, b-28))


def show_image_popup(path):
    if not os.path.exists(path):
        return
    if os.path.splitext(path)[1].lower() not in (".bmp", ".png", ".jpg", ".jpeg"):
        return
    try:
        img = Image.open(path)
    except Exception:
        return
    top = tk.Toplevel()
    top.title(os.path.basename(path))
    top.configure(bg=BG)
    top.resizable(True, True)
    w, h  = img.size
    sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
    scale = min(1.0, sw * 0.9 / w, sh * 0.9 / h)
    disp  = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS) if scale < 1.0 else img
    photo = ImageTk.PhotoImage(disp)
    tk.Label(top, image=photo, bg=BG).pack(padx=8, pady=8)
    top._photo = photo
    tk.Label(top, text=f"{os.path.basename(path)}  |  {w}x{h} px",
             bg=BG, fg=FG_DIM, font=("Courier New", 9)).pack(pady=8)


# ── FileBrowser ────────────────────────────────────────────────────────────────

class FileBrowser(tk.Frame):
    def __init__(self, parent, root_dir=OUTPUT, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self._root_dir   = root_dir
        self._at_root    = True
        self._cur_subdir = None

        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x", padx=8, pady=8)
        self._back_lbl = tk.Label(hdr, text="<- Back", bg=PANEL, fg=ACCENT,
                                   font=("Courier New", 9, "bold"), cursor="hand2")
        self._back_lbl.pack(side="left")
        self._back_lbl.bind("<Button-1>", lambda e: self._go_root())
        self._path_lbl = tk.Label(hdr, text=f"{root_dir}/", bg=PANEL, fg=FG_DIM,
                                   font=("Courier New", 9))
        self._path_lbl.pack(side="left", padx=8)

        box_wrap = tk.Frame(self, bg=BORDER, padx=1, pady=1)
        box_wrap.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._lb = tk.Listbox(box_wrap, bg=PANEL, fg=FG,
                               selectbackground=HOVER, selectforeground=ACCENT,
                               font=("Courier New", 10), borderwidth=0,
                               highlightthickness=0, activestyle="none", relief="flat")
        sb = tk.Scrollbar(box_wrap, orient="vertical", command=self._lb.yview,
                           bg=PANEL, troughcolor=PANEL, activebackground=ACCENT)
        self._lb.config(yscrollcommand=sb.set)
        self._lb.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._lb.bind("<<ListboxSelect>>", self._on_select)
        self._go_root()

    def refresh(self):
        if self._at_root: self._go_root()
        else: self._load_subdir(self._cur_subdir)

    def _go_root(self):
        self._at_root = True; self._cur_subdir = None
        self._path_lbl.config(text=f"{self._root_dir}/")
        self._lb.delete(0, tk.END)
        if not os.path.exists(self._root_dir):
            self._lb.insert(tk.END, "  (empty)"); return
        subdirs = [d for d in ("norm","mask","seg","code")
                   if os.path.isdir(os.path.join(self._root_dir, d))]
        if not subdirs:
            self._lb.insert(tk.END, "  (empty - run preprocess first)"); return
        for d in subdirs:
            count = len([f for f in os.listdir(os.path.join(self._root_dir, d))
                         if os.path.isfile(os.path.join(self._root_dir, d, f))])
            self._lb.insert(tk.END, f"  [DIR]  {d}  ({count} files)")

    def _load_subdir(self, subdir):
        self._at_root = False; self._cur_subdir = subdir
        path = os.path.join(self._root_dir, subdir)
        self._path_lbl.config(text=f"{self._root_dir}/{subdir}/")
        self._lb.delete(0, tk.END)
        if not os.path.exists(path):
            self._lb.insert(tk.END, "  (empty)"); return
        files = sorted(f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
        if not files: self._lb.insert(tk.END, "  (empty)"); return
        for f in files: self._lb.insert(tk.END, f"  {f}")

    def _on_select(self, _event):
        if not self._lb.curselection(): return
        raw = self._lb.get(self._lb.curselection()[0]).strip()
        if self._at_root:
            for d in ("norm","mask","seg","code"):
                if d in raw: self._load_subdir(d); break
        else:
            full = os.path.join(self._root_dir, self._cur_subdir, raw)
            if os.path.isfile(full): show_image_popup(full)
        self._lb.selection_clear(0, tk.END)


# ── LogBox ─────────────────────────────────────────────────────────────────────

class LogBox(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        tk.Label(self, text="  SYSTEM LOG", bg=PANEL, fg=FG_DIM,
                  font=("Courier New", 9, "bold"), anchor="w").pack(fill="x")
        self._txt = tk.Text(self, bg="#f8fff8", fg=LOG_FG, font=("Courier New", 9),
                             insertbackground=ACCENT, relief="flat",
                             borderwidth=0, state="disabled", wrap="word")
        sb = tk.Scrollbar(self, command=self._txt.yview, bg=PANEL, troughcolor=PANEL)
        self._txt.config(yscrollcommand=sb.set)
        self._txt.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        sb.pack(side="right", fill="y")

    def write(self, msg):
        self._txt.config(state="normal")
        self._txt.insert(tk.END, msg)
        self._txt.see(tk.END)
        self._txt.config(state="disabled")

    def clear(self):
        self._txt.config(state="normal")
        self._txt.delete("1.0", tk.END)
        self._txt.config(state="disabled")


# ══════════════════════════════════════════════════════════════════════════════
# Splash
# ══════════════════════════════════════════════════════════════════════════════

class SplashScreen(tk.Frame):
    def __init__(self, parent, on_choose):
        super().__init__(parent, bg=BG)
        self._cb = on_choose
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        for r in range(5): self.rowconfigure(r, weight=1)

        tk.Label(self, text=". . . . . . . . . . . . . . . . . . . . .",
                  bg=BG, fg=BORDER, font=("Courier New", 10)).grid(row=0, column=0, pady=40)
        tk.Label(self, text="IRIS BIOMETRIC KEY SYSTEM",
                  bg=BG, fg=ACCENT, font=("Courier New", 22, "bold")).grid(row=1, column=0)
        tk.Label(self,
                  text="Iris segmentation  .  CNN stabilisation  .  Fuzzy extractor  .  AES-256",
                  bg=BG, fg=FG_DIM, font=("Courier New", 10)).grid(row=2, column=0, pady=8)

        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.grid(row=3, column=0, pady=16)

        for title, sub, color, mode in [
            ("GENERATE KEY",  "Register iris and generate cryptographic key", ACCENT,  "generate"),
            ("VERIFY KEY",    "Verify by key input or iris image upload",      ACCENT4, "verify"),
            ("ANALYSIS",      "Performance charts for your paper",             ACCENT5, "analysis"),
        ]:
            self._mode_card(btn_frame, title, sub, color,
                             lambda m=mode: self._cb(m)).pack(side="left", padx=14)

        tk.Label(self, text="v1.0  .  CASIA-IrisV4",
                  bg=BG, fg=FG_DIM, font=("Courier New", 8)).grid(row=4, column=0, pady=8)

    @staticmethod
    def _mode_card(parent, title, subtitle, color, cmd):
        card = tk.Frame(parent, bg=PANEL, cursor="hand2",
                         highlightbackground=color, highlightthickness=2)
        tk.Label(card, text=title, bg=PANEL, fg=color,
                  font=("Courier New", 12, "bold"), padx=28, pady=12).pack()
        sub = tk.Label(card, text=subtitle, bg=PANEL, fg=FG_DIM,
                        font=("Courier New", 8), padx=20, pady=4)
        sub.pack(pady=(0, 10))

        def _enter(e):
            card.config(bg=HOVER)
            for w in card.winfo_children(): w.config(bg=HOVER)
        def _leave(e):
            card.config(bg=PANEL)
            for w in card.winfo_children(): w.config(bg=PANEL)

        card.bind("<Button-1>", lambda e: cmd())
        card.bind("<Enter>", _enter); card.bind("<Leave>", _leave)
        for w in card.winfo_children():
            w.bind("<Button-1>", lambda e: cmd())
            w.bind("<Enter>", _enter); w.bind("<Leave>", _leave)
        return card


# ══════════════════════════════════════════════════════════════════════════════
# Generate screen
# ══════════════════════════════════════════════════════════════════════════════

class GenerateScreen(tk.Frame):
    def __init__(self, parent, on_back):
        super().__init__(parent, bg=BG)
        self._on_back = on_back
        self._log = None; self._browser = None
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=16, pady=10)
        back = tk.Label(top, text="<- Back", bg=BG, fg=ACCENT,
                         font=("Courier New", 10, "bold"), cursor="hand2")
        back.pack(side="left")
        back.bind("<Button-1>", lambda e: self._on_back())
        tk.Label(top, text="KEY GENERATION MODE", bg=BG, fg=FG,
                  font=("Courier New", 13, "bold")).pack(side="left", padx=20)

        ctrl = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        ctrl.pack(fill="x", padx=16, pady=(0, 8))

        tk.Label(ctrl, text="Person", bg=PANEL, fg=FG_DIM,
                  font=("Courier New", 9)).grid(row=0, column=0, padx=(12,4), pady=10)
        self._person = tk.StringVar()
        people = get_people()
        self._person_cb = ttk.Combobox(ctrl, textvariable=self._person,
                                        values=people, width=8,
                                        font=("Courier New", 10), state="readonly")
        if people: self._person_cb.current(0)
        self._person_cb.grid(row=0, column=1, padx=4)
        self._person_cb.bind("<<ComboboxSelected>>", self._on_person_change)

        tk.Label(ctrl, text="Eye", bg=PANEL, fg=FG_DIM,
                  font=("Courier New", 9)).grid(row=0, column=2, padx=(16,4))
        self._eye = tk.StringVar(value="L")
        self._eye_cb = ttk.Combobox(ctrl, textvariable=self._eye,
                                     values=["L","R"], width=4,
                                     font=("Courier New", 10), state="readonly")
        self._eye_cb.grid(row=0, column=3, padx=4)
        if people: self._refresh_eyes(people[0])

        tk.Frame(ctrl, bg=PANEL, width=16).grid(row=0, column=4)
        for col, (label, cmd, color) in enumerate([
            ("Preprocess",      self._do_preprocess, ACCENT),
            ("Feature Extract", self._do_feature,    "#1565c0"),
            ("Generate Key",    self._do_generate,   ACCENT5),
        ], start=5):
            styled_btn(ctrl, label, cmd, color=color).grid(row=0, column=col, padx=6, pady=8)

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        main.columnconfigure(0, weight=1); main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        bw = tk.Frame(main, bg=BORDER, padx=1, pady=1)
        bw.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        self._browser = FileBrowser(bw)
        self._browser.pack(fill="both", expand=True)

        lw = tk.Frame(main, bg=BORDER, padx=1, pady=1)
        lw.grid(row=0, column=1, sticky="nsew")
        self._log = LogBox(lw)
        self._log.pack(fill="both", expand=True)

        kf = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        kf.pack(fill="x", padx=16, pady=(0, 12))
        tk.Label(kf, text="Generated Key :", bg=PANEL, fg=FG_DIM,
                  font=("Courier New", 9)).pack(side="left", padx=10)
        self._key_var = tk.StringVar(value="--")
        tk.Label(kf, textvariable=self._key_var, bg=PANEL, fg=ACCENT2,
                  font=("Courier New", 10, "bold")).pack(side="left")

    def _on_person_change(self, _=None):
        p = self._person.get()
        if p: self._refresh_eyes(p)

    def _refresh_eyes(self, person):
        eyes = get_available_eyes(person)
        self._eye_cb.config(values=eyes)
        if eyes: self._eye.set(eyes[0])
        else: self._eye.set("")

    def _run(self, fn): threading.Thread(target=fn, daemon=True).start()

    def _dataset_path(self):
        p, e = self._person.get(), self._eye.get()
        if not p: self._log.write("! Please select a person.\n"); return None
        if not e: self._log.write("! No eye data available.\n"); return None
        path = os.path.join(DATASET, p, e)
        if not os.path.exists(path): self._log.write(f"! Path not found: {path}\n"); return None
        return path

    def _do_preprocess(self):
        def task():
            path = self._dataset_path()
            if not path: return
            self._log.write(f"\n> Preprocessing  {path}\n")
            clear_output(); ensure_output_dirs()
            import main as m
            m.dataset_path = path
            m.run_segmentation(); m.stabilize_images()
            self._log.write("OK Preprocessing complete.\n")
            self.after(0, self._browser.refresh)
        self._run(task)

    def _do_feature(self):
        def task():
            self._log.write("\n> Feature extraction (Log-Gabor)...\n")
            import main as m
            m.run_feature()
            self._log.write("OK Feature extraction complete.\n")
            self.after(0, self._browser.refresh)
        self._run(task)

    def _do_generate(self):
        def task():
            self._log.write("\n> Rotation compensation + Hamming distance...\n")
            import main as m
            codes = m.rotation_test()
            if not codes: self._log.write("! No IrisCode found.\n"); return
            self._log.write("> Fuzzy extractor...\n")
            key = m.fuzzy_key_generation(codes)
            if not key: self._log.write("! Key generation failed.\n"); return
            p, e = self._person.get(), self._eye.get()
            write_key_db(p, e, key)
            self._log.write(f"\nKEY: {key}\n")
            self._log.write(f"Saved -> {KEY_DB}\n")
            self.after(0, lambda: self._key_var.set(key))
        self._run(task)


# ══════════════════════════════════════════════════════════════════════════════
# Verify screen（支持密钥输入 + 图像上传两种验证方式）
# ══════════════════════════════════════════════════════════════════════════════

class VerifyScreen(tk.Frame):
    def __init__(self, parent, on_back):
        super().__init__(parent, bg=BG)
        self._on_back    = on_back
        self._log        = None
        self._img_path   = None   # 用户上传的图像路径
        self._img_thumb  = None   # 防 GC
        self._build()

    def _build(self):
        # 顶栏
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=16, pady=10)
        back = tk.Label(top, text="<- Back", bg=BG, fg=ACCENT4,
                         font=("Courier New", 10, "bold"), cursor="hand2")
        back.pack(side="left")
        back.bind("<Button-1>", lambda e: self._on_back())
        tk.Label(top, text="KEY VERIFICATION MODE", bg=BG, fg=FG,
                  font=("Courier New", 13, "bold")).pack(side="left", padx=20)

        # 状态提示
        notice = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        notice.pack(fill="x", padx=16, pady=(0,8))
        self._status_lbl = tk.Label(
            notice,
            text="  Two ways to verify: enter key directly, or upload an iris image.",
            bg=PANEL, fg=FG_DIM, font=("Courier New", 9), anchor="w")
        self._status_lbl.pack(fill="x", padx=8, pady=8)

        # ── 方式一：输入密钥 ─────────────────────────────────────────────────
        sec1 = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        sec1.pack(fill="x", padx=16, pady=(0,6))

        tk.Label(sec1, text="  Way 1: Enter Key", bg=PANEL, fg=ACCENT4,
                  font=("Courier New", 10, "bold"), anchor="w").pack(fill="x", padx=8, pady=(8,4))

        inp1 = tk.Frame(sec1, bg=PANEL)
        inp1.pack(fill="x", padx=8, pady=(0,8))
        self._key_entry = tk.Entry(
            inp1, bg="#f8fff8", fg=ACCENT2, insertbackground=ACCENT2,
            font=("Courier New", 11), relief="flat", width=52,
            highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT4)
        self._key_entry.pack(side="left", padx=(0,6))
        self._key_entry.bind("<Return>", lambda e: self._do_verify_key())
        styled_btn(inp1, "Verify Key",  self._do_verify_key,  color=ACCENT4).pack(side="left", padx=4)
        styled_btn(inp1, "Rebuild DB",  self._do_rebuild,     color=ACCENT3).pack(side="left", padx=4)

        # ── 方式二：上传图像 ─────────────────────────────────────────────────
        sec2 = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        sec2.pack(fill="x", padx=16, pady=(0,6))

        tk.Label(sec2, text="  Way 2: Upload Iris Image", bg=PANEL, fg=ACCENT4,
                  font=("Courier New", 10, "bold"), anchor="w").pack(fill="x", padx=8, pady=(8,4))

        inp2 = tk.Frame(sec2, bg=PANEL)
        inp2.pack(fill="x", padx=8, pady=(0,8))

        styled_btn(inp2, "Select Image", self._do_select_image,
                   color=ACCENT).pack(side="left", padx=(0,8))

        self._img_label = tk.Label(inp2, text="No image selected", bg=PANEL, fg=FG_DIM,
                                    font=("Courier New", 9))
        self._img_label.pack(side="left", padx=4)

        styled_btn(inp2, "Verify Image", self._do_verify_image,
                   color=ACCENT4).pack(side="right", padx=4)

        # 图像预览（小缩略图）
        self._preview_frame = tk.Frame(sec2, bg=PANEL)
        self._preview_frame.pack(fill="x", padx=8, pady=(0,8))
        self._preview_lbl = tk.Label(self._preview_frame, bg=PANEL)
        self._preview_lbl.pack(side="left")

        # ── 结果横幅 ──────────────────────────────────────────────────────────
        result_frame = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        result_frame.pack(fill="x", padx=16, pady=(0,6))
        self._result_lbl = tk.Label(result_frame, text="", bg=PANEL, fg=FG,
                                     font=("Courier New", 15, "bold"), pady=12)
        self._result_lbl.pack()

        # ── 日志 ──────────────────────────────────────────────────────────────
        lw = tk.Frame(self, bg=BORDER, padx=1, pady=1)
        lw.pack(fill="both", expand=True, padx=16, pady=(0,12))
        self._log = LogBox(lw)
        self._log.pack(fill="both", expand=True)

    # ── 方式一：密钥验证 ──────────────────────────────────────────────────────

    def _do_verify_key(self):
        key = self._key_entry.get().strip()
        if not key:
            self._show_result("! Please enter a key", ACCENT3); return

        def task():
            if not os.path.exists(KEY_DB):
                self.after(0, lambda: self._status_lbl.config(
                    text="  Building key database...", fg=ACCENT5))
                self._build_database()
                self.after(0, lambda: self._status_lbl.config(
                    text="  Key database ready.", fg=ACCENT2))

            self._log.write("\n> [Key] Verifying key...\n")
            with open(KEY_DB) as f:
                for line in f:
                    line = line.strip()
                    if ":" not in line: continue
                    name, k = line.split(":", 1)
                    if k == key:
                        self._log.write(f"MATCH: {name}\n")
                        parts  = name.rsplit("_", 1)
                        person = parts[0]; eye = parts[1] if len(parts) > 1 else "?"
                        msg    = f"MATCH (Key): Person {person}  |  Eye {eye}"
                        self.after(0, lambda m=msg: self._show_result(m, ACCENT2))
                        return
            self._log.write("No match found.\n")
            self.after(0, lambda: self._show_result("NO MATCH FOUND", ACCENT3))

        threading.Thread(target=task, daemon=True).start()

    # ── 方式二：图像验证 ──────────────────────────────────────────────────────

    def _do_select_image(self):
        path = filedialog.askopenfilename(
            title="Select iris image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        self._img_path = path
        self._img_label.config(text=os.path.basename(path), fg=ACCENT2)

        # 显示小预览
        try:
            img   = Image.open(path)
            scale = min(1.0, 120 / img.width, 80 / img.height)
            thumb = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(thumb)
            self._preview_lbl.config(image=photo)
            self._img_thumb = photo
        except Exception:
            pass

    def _do_verify_image(self):
        if not self._img_path or not os.path.exists(self._img_path):
            self._show_result("! Please select an iris image first", ACCENT3); return

        def task():
            self._log.write(f"\n> [Image] Processing: {os.path.basename(self._img_path)}\n")

            # 在临时目录中处理上传的图像
            tmp_dir  = os.path.join("output", "verify_tmp")
            os.makedirs(tmp_dir, exist_ok=True)

            out_norm = os.path.join(tmp_dir, "query_norm.bmp")
            out_mask = os.path.join(tmp_dir, "query_mask.bmp")
            out_seg  = os.path.join(tmp_dir, "query_seg.bmp")
            out_code = os.path.join(tmp_dir, "query_code.png")

            for p in [out_norm, out_mask, out_seg, out_code]:
                if os.path.exists(p): os.remove(p)

            import main as m
            self._log.write("> Running segmentation + feature extraction...\n")
            query_code = m.process_single_image(
                self._img_path, out_norm, out_mask, out_seg, out_code)

            if query_code is None:
                self._log.write("! Segmentation/feature extraction failed.\n")
                self.after(0, lambda: self._show_result("PROCESSING FAILED", ACCENT3))
                return

            self._log.write(f"> IrisCode extracted: {len(query_code)} bits\n")

            # ── 遍历 dataset/ 下所有人，不依赖 key_db 是否有该人记录 ────────────
            # 策略：
            #   对 dataset/ 中每个人/眼的前3张参考图提取 IrisCode 并计算 HD，
            #   HD < HD_THRESH 则立即停止，返回匹配结果；
            #   遍历完所有人仍未找到则返回 NO MATCH。
            self._log.write("> Matching against all persons in dataset...\n")

            HD_THRESH    = 0.38
            best_name    = None
            best_hd      = 1.0
            found        = False
            ref_tmp_dir  = os.path.join("output", "verify_ref_tmp")

            from scripts.rotation_match import best_rotation_match as _brm

            for person in sorted(get_people()):
                if found:
                    break
                for eye in get_available_eyes(person):
                    if found:
                        break

                    person_path = os.path.join(DATASET, person, eye)
                    name        = f"{person}_{eye}"

                    img_files = sorted(
                        f for f in os.listdir(person_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))

                    os.makedirs(ref_tmp_dir, exist_ok=True)
                    person_min_hd = 1.0

                    # 只取前3张参考图（加速）
                    for ref_file in img_files[:3]:
                        ref_path = os.path.join(person_path, ref_file)
                        ref_stem = os.path.splitext(ref_file)[0]
                        r_norm = os.path.join(ref_tmp_dir, f"{ref_stem}_norm.bmp")
                        r_mask = os.path.join(ref_tmp_dir, f"{ref_stem}_mask.bmp")
                        r_seg  = os.path.join(ref_tmp_dir, f"{ref_stem}_seg.bmp")
                        r_code = os.path.join(ref_tmp_dir, f"{ref_stem}_code.png")

                        ref_code = m.process_single_image(
                            ref_path, r_norm, r_mask, r_seg, r_code)
                        if ref_code is None or len(ref_code) != len(query_code):
                            continue

                        hd = _brm(query_code, ref_code)
                        person_min_hd = min(person_min_hd, hd)
                        self._log.write(f"  {name}/{ref_file}: HD={hd:.4f}\n")

                        # 早退：只要某张参考图 HD < 阈值，立即确认匹配，停止搜索
                        if hd <= HD_THRESH:
                            best_hd   = hd
                            best_name = name
                            found     = True
                            self._log.write(
                                f"  -> Early stop: matched {name} "
                                )
                            break

                    if not found and person_min_hd < best_hd:
                        best_hd   = person_min_hd
                        best_name = name

                    shutil.rmtree(ref_tmp_dir, ignore_errors=True)

            self._log.write(
                f"\nBest match: {best_name}  HD={best_hd:.4f}  "
                )

            if best_name and best_hd <= HD_THRESH:
                p_parts = best_name.rsplit("_", 1)
                pmatch  = p_parts[0]
                ematch  = p_parts[1] if len(p_parts) > 1 else "?"
                msg = (f"MATCH (Image): Person {pmatch}  |  "
                       f"Eye {ematch}  ")
                self.after(0, lambda m=msg: self._show_result(m, ACCENT2))
            else:
                msg = f"NO MATCH  (best HD={best_hd:.4f} > threshold={HD_THRESH})"
                self.after(0, lambda m=msg: self._show_result(m, ACCENT3))

        threading.Thread(target=task, daemon=True).start()

    # ── 重建数据库 ────────────────────────────────────────────────────────────

    def _do_rebuild(self):
        def task():
            self.after(0, lambda: self._status_lbl.config(
                text="  Rebuilding...", fg=ACCENT5))
            if os.path.exists(KEY_DB): os.remove(KEY_DB)
            self._build_database()
            self._log.write("Database rebuilt.\n")
            self.after(0, lambda: self._status_lbl.config(
                text="  Key database ready.", fg=ACCENT2))
        threading.Thread(target=task, daemon=True).start()

    def _build_database(self):
        import main as m
        for p in get_people():
            for e in get_available_eyes(p):
                path = os.path.join(DATASET, p, e)
                self._log.write(f"  Building {p}-{e}...\n")
                clear_output(); ensure_output_dirs()
                m.dataset_path = path
                m.run_segmentation(); m.stabilize_images(); m.run_feature()
                codes = m.rotation_test()
                if not codes: continue
                key = m.fuzzy_key_generation(codes)
                if key: write_key_db(p, e, key)

    def _show_result(self, msg, color):
        self._result_lbl.config(text=msg, fg=color)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis screen
# ══════════════════════════════════════════════════════════════════════════════

class AnalysisScreen(tk.Frame):
    CHART_META = [
        ("hd_distribution",           "HD Distribution",     "Genuine vs Impostor HD"),
        ("roc_curve",                  "ROC Curve",           "AUC / EER"),
        ("far_frr_curve",              "FAR / FRR Curve",     "Threshold sensitivity"),
        ("stable_bits_vs_consistency", "Bits vs Consistency", "Threshold sweep"),
    ]

    def __init__(self, parent, on_back):
        super().__init__(parent, bg=BG)
        self._on_back = on_back; self._log = None; self._thumb_refs = []
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=16, pady=10)
        back = tk.Label(top, text="<- Back", bg=BG, fg=ACCENT5,
                         font=("Courier New", 10, "bold"), cursor="hand2")
        back.pack(side="left"); back.bind("<Button-1>", lambda e: self._on_back())
        tk.Label(top, text="PERFORMANCE ANALYSIS", bg=BG, fg=FG,
                  font=("Courier New", 13, "bold")).pack(side="left", padx=20)

        hint = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        hint.pack(fill="x", padx=16, pady=(0,8))
        tk.Label(hint, text="  Run preprocess + feature extract first. Then click Generate Charts.",
                  bg=PANEL, fg=FG_DIM, font=("Courier New", 9), anchor="w").pack(fill="x", padx=8, pady=8)

        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(fill="x", padx=16, pady=(0,8))
        styled_btn(btn_row, "Generate All Charts", self._do_generate,
                   color=ACCENT5, font_size=10).pack(side="left")
        tk.Label(btn_row, text="  Charts saved to output/plots/",
                  bg=BG, fg=FG_DIM, font=("Courier New", 9)).pack(side="left", padx=12)

        self._grid_frame = tk.Frame(self, bg=BG)
        self._grid_frame.pack(fill="both", expand=True, padx=16, pady=(0,8))
        self._grid_frame.columnconfigure(0, weight=1); self._grid_frame.columnconfigure(1, weight=1)

        self._cards = {}
        for idx, (key, title, subtitle) in enumerate(self.CHART_META):
            row, col = divmod(idx, 2)
            card = self._make_placeholder_card(self._grid_frame, title, subtitle)
            card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            self._cards[key] = card
            self._grid_frame.rowconfigure(row, weight=1)

        lw = tk.Frame(self, bg=BORDER, padx=1, pady=1)
        lw.pack(fill="x", padx=16, pady=(0,12))
        self._log = LogBox(lw); self._log.pack(fill="both", expand=True)
        lw.config(height=120)

    def _make_placeholder_card(self, parent, title, subtitle):
        card = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        tk.Label(card, text=title, bg=PANEL, fg=ACCENT5, font=("Courier New", 10, "bold")).pack(pady=(8,2))
        tk.Label(card, text=subtitle, bg=PANEL, fg=FG_DIM, font=("Courier New", 8)).pack()
        tk.Label(card, text="(not generated yet)", bg=PANEL, fg=BORDER, font=("Courier New", 8)).pack(pady=(4,8))
        return card

    def _update_card(self, key, img_path):
        if key not in self._cards: return
        card = self._cards[key]
        for w in card.winfo_children(): w.destroy()
        # 始终用绝对路径，避免工作目录不同导致图片加载失败（ROC图不显示的根因）
        abs_path = os.path.abspath(img_path) if img_path else ""
        try:
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Not found: {abs_path}")
            img   = Image.open(abs_path)
            scale = 440 / img.width
            thumb = img.resize((440, int(img.height*scale)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(thumb)
            self._thumb_refs.append(photo)
            lbl = tk.Label(card, image=photo, bg=PANEL, cursor="hand2")
            lbl.pack(padx=4, pady=4)
            lbl.bind("<Button-1>", lambda e, p=abs_path: show_image_popup(p))
            meta = next((m for m in self.CHART_META if m[0] == key), None)
            if meta:
                tk.Label(card, text=meta[1], bg=PANEL, fg=ACCENT5,
                          font=("Courier New", 9, "bold")).pack()
                tk.Label(card, text=f"Click to view  |  {os.path.basename(abs_path)}",
                          bg=PANEL, fg=FG_DIM,
                          font=("Courier New", 7)).pack(pady=(0,4))
        except Exception as e:
            tk.Label(card, text=f"Load error: {os.path.basename(str(img_path))}",
                      bg=PANEL, fg=ACCENT3,
                      font=("Courier New", 8)).pack(pady=(8,2))
            tk.Label(card, text=str(e)[:100], bg=PANEL, fg=FG_DIM,
                      font=("Courier New", 7), wraplength=380).pack(pady=(0,8))

    def _do_generate(self):
        def task():
            self._log.write("\n> Generating performance charts...\n")
            try:
                import analysis, importlib
                importlib.reload(analysis)
                results = analysis.generate_all_plots(progress_callback=self._log.write)
            except Exception as e:
                self._log.write(f"ERROR: {e}\n"); return
            self._log.write(f"\nDone. {len(results)} charts generated.\n")
            for key, path in results.items():
                self.after(0, lambda k=key, p=path: self._update_card(k, p))
        threading.Thread(target=task, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Iris Biometric Key System")
        self.geometry("1200x820")
        self.minsize(960, 680)
        self.configure(bg=BG)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground="#f8fff8", background=PANEL,
                          foreground=FG, selectbackground=HOVER, selectforeground=FG,
                          bordercolor=BORDER, arrowcolor=ACCENT)
        style.map("TCombobox", fieldbackground=[("readonly","#f8fff8")],
                   foreground=[("readonly",FG)], background=[("readonly",PANEL)])

        self._current = None
        self._show_splash()

    def _show_splash(self): self._switch(SplashScreen(self, self._on_mode))

    def _on_mode(self, mode):
        screens = {"generate": GenerateScreen, "verify": VerifyScreen, "analysis": AnalysisScreen}
        cls = screens.get(mode)
        if cls: self._switch(cls(self, self._show_splash))

    def _switch(self, frame):
        if self._current is not None: self._current.destroy()
        self._current = frame
        frame.pack(fill="both", expand=True)


if __name__ == "__main__":
    App().mainloop()