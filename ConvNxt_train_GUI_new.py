"""
OCR Training GUI Application
A graphical user interface for training OCR Detection and Recognition models
"""

import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
import queue

# Try to import ttkbootstrap for modern themes (DO NOT overwrite tkinter.ttk)
try:
    import ttkbootstrap as ttkb
    MODERN_THEME = True
except ImportError:
    ttkb = ttk  # type: ignore
    MODERN_THEME = False
    print("ttkbootstrap not found. Install with: pip install ttkbootstrap")


class OCRTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Training Studio")
        self.root.geometry("1400x900")

        # Choose widget factory (bootstrap or tkinter)
        self.UI = ttkb if MODERN_THEME else ttk  # type: ignore

        # Apply theme / style
        if MODERN_THEME:
            # If root is a ttkbootstrap.Window, theme is already applied
            # Creating Style is still safe and allows later style tweaks.
            try:
                self.style = ttkb.Style("darkly")  # type: ignore  # Dark theme: darkly, superhero, cyborg, solar, vapor
            except Exception:
                self.style = ttkb.Style()  # type: ignore
        else:
            self.setup_custom_style()

        # Queue for thread-safe logging
        self.log_queue = queue.Queue()
        self.training_thread = None
        self.is_training = False

        # Create main container
        main_container = self.UI.Frame(root, padding=15)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Create notebook for tabs
        self.notebook = self.UI.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.create_config_tab()
        self.create_training_tab()
        self.create_log_tab()

        # Status bar
        status_frame = self.UI.Frame(root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.status_var = tk.StringVar(value="Ready")
        if MODERN_THEME:
            status_bar = self.UI.Label(
                status_frame, textvariable=self.status_var, bootstyle="inverse-secondary"  # type: ignore
            )
        else:
            status_bar = self.UI.Label(
                status_frame,
                textvariable=self.status_var,
                relief=tk.SUNKEN,
                anchor=tk.W,
                background="#2c3e50",
                foreground="white",
                padding=5,
            )
        status_bar.pack(fill=tk.X)

        # Start queue checking
        self.check_log_queue()

    def setup_custom_style(self):
        """Setup custom modern styling when ttkbootstrap is not available."""
        style = ttk.Style()

        # Dark theme colors
        bg_color = "#1e1e1e"  # Dark background
        fg_color = "#e0e0e0"  # Light text
        accent_color = "#3498db"  # Blue accent

        self.root.configure(bg=bg_color)

        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color, font=("Segoe UI", 10))
        style.configure("TLabelframe", background=bg_color, foreground=fg_color)
        style.configure("TLabelframe.Label", background=bg_color, foreground=fg_color, font=("Segoe UI", 10, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=8, background="#2c3e50", foreground=fg_color)
        style.configure("TRadiobutton", background=bg_color, foreground=fg_color, font=("Segoe UI", 10))
        style.configure("TCheckbutton", background=bg_color, foreground=fg_color, font=("Segoe UI", 10))
        style.configure("TEntry", font=("Segoe UI", 10), padding=5, fieldbackground="#2c3e50", foreground=fg_color)
        style.configure("TNotebook", background=bg_color, borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[20, 10], background="#2c3e50", foreground=fg_color)
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=10, background=accent_color, foreground="white")
        style.configure("TProgressbar", thickness=25, borderwidth=0, background=accent_color)

    # -------------------------
    # UI helpers
    # -------------------------
    def _btn(self, parent, text, command, bootstyle=None, **kwargs):
        """Create a button that works with or without ttkbootstrap."""
        if MODERN_THEME and bootstyle:
            return self.UI.Button(parent, text=text, command=command, bootstyle=bootstyle, **kwargs)  # type: ignore
        return self.UI.Button(parent, text=text, command=command, **kwargs)

    def create_path_row(self, parent, label_text, var, is_dir=True):
        frame = self.UI.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        self.UI.Label(frame, text=label_text, width=20).pack(side=tk.LEFT, padx=5)
        self.UI.Entry(frame, textvariable=var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        if is_dir:
            self._btn(frame, "Browse", lambda: self.browse_directory(var)).pack(side=tk.LEFT, padx=5)
        else:
            self._btn(frame, "Browse", lambda: self.browse_file(var)).pack(side=tk.LEFT, padx=5)

    def browse_directory(self, var):
        directory = filedialog.askdirectory(initialdir=var.get() if var.get() else ".")
        if directory:
            var.set(directory)

    def browse_file(self, var):
        initial = os.path.dirname(var.get()) if var.get() else "."
        file_path = filedialog.askopenfilename(initialdir=initial)
        if file_path:
            var.set(file_path)

    # -------------------------
    # Tabs
    # -------------------------
    def create_config_tab(self):
        config_frame = self.UI.Frame(self.notebook, padding=10)
        self.notebook.add(config_frame, text="⚙️ Configuration")

        # Scrollable canvas (Canvas is tk, not ttk)
        canvas = tk.Canvas(config_frame, highlightthickness=0)
        scrollbar = self.UI.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        def _on_configure(_):
            canvas.configure(scrollregion=canvas.bbox("all"))

        scrollable_frame.bind("<Configure>", _on_configure)
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --- Training Mode ---
        mode_section = self.UI.LabelFrame(scrollable_frame, text="🎯 Training Mode")
        mode_section.pack(fill=tk.X, padx=10, pady=10)

        mode_content = self.UI.Frame(mode_section)
        mode_content.pack(fill=tk.X, padx=15, pady=15)

        self.training_mode_var = tk.StringVar(value="det_rec")

        mode_desc = self.UI.Frame(mode_content)
        mode_desc.pack(fill=tk.X, pady=(0, 10))
        self.UI.Label(mode_desc, text="Select the training mode for your OCR model:", font=("Segoe UI", 9, "italic")).pack(anchor=tk.W)

        self.UI.Radiobutton(mode_content, text="🔍 Detection + Recognition (Joint Training)",
                            variable=self.training_mode_var, value="det_rec").pack(anchor=tk.W, pady=3)
        self.UI.Radiobutton(mode_content, text="📦 Detection Only",
                            variable=self.training_mode_var, value="det_only").pack(anchor=tk.W, pady=3)
        self.UI.Radiobutton(mode_content, text="📝 Recognition Only (Pre-cropped Images)",
                            variable=self.training_mode_var, value="rec_only").pack(anchor=tk.W, pady=3)

        # --- Dataset Paths ---
        paths_section = self.UI.LabelFrame(scrollable_frame, text="📁 Dataset Paths")
        paths_section.pack(fill=tk.X, padx=10, pady=10)

        paths_content = self.UI.Frame(paths_section)
        paths_content.pack(fill=tk.X, padx=15, pady=15)

        self.train_img_var = tk.StringVar(value="")
        self.create_path_row(paths_content, "Training Images:", self.train_img_var, is_dir=True)

        self.train_label_var = tk.StringVar(value="")
        self.create_path_row(paths_content, "Training Labels:", self.train_label_var, is_dir=False)

        self.val_img_var = tk.StringVar(value="")
        self.create_path_row(paths_content, "Validation Images:", self.val_img_var, is_dir=True)

        self.val_label_var = tk.StringVar(value="")
        self.create_path_row(paths_content, "Validation Labels:", self.val_label_var, is_dir=False)

        self.output_dir_var = tk.StringVar(value="")
        self.create_path_row(paths_content, "Output Directory:", self.output_dir_var, is_dir=True)

        # --- Model Checkpoints ---
        ckpt_section = self.UI.LabelFrame(scrollable_frame, text="💾 Model Checkpoints")
        ckpt_section.pack(fill=tk.X, padx=10, pady=10)

        ckpt_content = self.UI.Frame(ckpt_section)
        ckpt_content.pack(fill=tk.X, padx=15, pady=15)

        self.pretrained_type_var = tk.StringVar(value="fcmae")

        self.UI.Radiobutton(ckpt_content, text="🚫 No Pretrained Weights",
                            variable=self.pretrained_type_var, value="none").pack(anchor=tk.W, pady=3)
        self.UI.Radiobutton(ckpt_content, text="🎨 Use FCMAE Pretrained Backbone",
                            variable=self.pretrained_type_var, value="fcmae").pack(anchor=tk.W, pady=3)
        self.UI.Radiobutton(ckpt_content, text="📚 Use OCR Pretrained Checkpoint (Recognition Only)",
                            variable=self.pretrained_type_var, value="ocr").pack(anchor=tk.W, pady=3)

        self.fcmae_ckpt_var = tk.StringVar(value="")
        self.create_path_row(ckpt_content, "FCMAE Checkpoint:", self.fcmae_ckpt_var, is_dir=False)

        self.ocr_ckpt_var = tk.StringVar(value="")
        self.create_path_row(ckpt_content, "OCR Checkpoint:", self.ocr_ckpt_var, is_dir=False)

        # --- Image Dimensions ---
        dims_section = self.UI.LabelFrame(scrollable_frame, text="📐 Image Dimensions")
        dims_section.pack(fill=tk.X, padx=10, pady=10)

        dims_grid = self.UI.Frame(dims_section)
        dims_grid.pack(fill=tk.X, padx=15, pady=15)

        self.UI.Label(dims_grid, text="Detection Image Height:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.det_imgH_var = tk.StringVar(value="640")
        self.UI.Entry(dims_grid, textvariable=self.det_imgH_var, width=15).grid(row=0, column=1, padx=5, pady=2)

        self.UI.Label(dims_grid, text="Detection Image Width:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.det_imgW_var = tk.StringVar(value="640")
        self.UI.Entry(dims_grid, textvariable=self.det_imgW_var, width=15).grid(row=0, column=3, padx=5, pady=2)

        self.UI.Label(dims_grid, text="Recognition Image Height:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.rec_imgH_var = tk.StringVar(value="96")
        self.UI.Entry(dims_grid, textvariable=self.rec_imgH_var, width=15).grid(row=1, column=1, padx=5, pady=2)

        self.UI.Label(dims_grid, text="Recognition Image Width:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.rec_imgW_var = tk.StringVar(value="198")
        self.UI.Entry(dims_grid, textvariable=self.rec_imgW_var, width=15).grid(row=1, column=3, padx=5, pady=2)

        # --- Training Parameters ---
        params_section = self.UI.LabelFrame(scrollable_frame, text="⚡ Training Parameters")
        params_section.pack(fill=tk.X, padx=10, pady=10)

        params_grid = self.UI.Frame(params_section)
        params_grid.pack(fill=tk.X, padx=15, pady=15)

        self.UI.Label(params_grid, text="Epochs:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.epochs_var = tk.StringVar(value="100")
        self.UI.Entry(params_grid, textvariable=self.epochs_var, width=15).grid(row=0, column=1, padx=5, pady=2)

        self.UI.Label(params_grid, text="Batch Size:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.batch_size_var = tk.StringVar(value="8")
        self.UI.Entry(params_grid, textvariable=self.batch_size_var, width=15).grid(row=0, column=3, padx=5, pady=2)

        self.UI.Label(params_grid, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.lr_var = tk.StringVar(value="0.001")
        self.UI.Entry(params_grid, textvariable=self.lr_var, width=15).grid(row=1, column=1, padx=5, pady=2)

        self.UI.Label(params_grid, text="Early Stop Patience:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.patience_var = tk.StringVar(value="10")
        self.UI.Entry(params_grid, textvariable=self.patience_var, width=15).grid(row=1, column=3, padx=5, pady=2)

        # --- Advanced Options ---
        advanced_section = self.UI.LabelFrame(scrollable_frame, text="🔧 Advanced Options")
        advanced_section.pack(fill=tk.X, padx=10, pady=10)

        advanced_content = self.UI.Frame(advanced_section)
        advanced_content.pack(fill=tk.X, padx=15, pady=15)

        self.maintain_aspect_var = tk.BooleanVar(value=True)
        self.UI.Checkbutton(advanced_content, text="Maintain Aspect Ratio (Recognition)",
                            variable=self.maintain_aspect_var).pack(anchor=tk.W)

        self.visualize_var = tk.BooleanVar(value=False)
        self.UI.Checkbutton(advanced_content, text="Visualize Training Samples",
                            variable=self.visualize_var).pack(anchor=tk.W)

        # --- IDs to Train ---
        ids_section = self.UI.LabelFrame(scrollable_frame, text="🎲 IDs to Train")
        ids_section.pack(fill=tk.X, padx=10, pady=10)

        ids_content = self.UI.Frame(ids_section)
        ids_content.pack(fill=tk.X, padx=15, pady=15)

        self.UI.Label(ids_content, text="Specific IDs (comma-separated, leave empty for all):").pack(anchor=tk.W)
        self.ids_var = tk.StringVar(value="")
        self.UI.Entry(ids_content, textvariable=self.ids_var, width=50).pack(fill=tk.X, pady=5)
        self.UI.Label(ids_content,
                      text="Examples: '0' for ID 0 only, '0,1,2' for multiple IDs, '' for all IDs",
                      font=("TkDefaultFont", 8, "italic")).pack(anchor=tk.W)

        # --- Action Buttons ---
        action_frame = self.UI.Frame(scrollable_frame)
        action_frame.pack(fill=tk.X, padx=10, pady=15)

        self._btn(action_frame, "💾 Save Configuration", self.save_config,
                  bootstyle="success").pack(side=tk.LEFT, padx=5)
        self._btn(action_frame, "📂 Load Configuration", self.load_config,
                  bootstyle="info").pack(side=tk.LEFT, padx=5)
        self._btn(action_frame, "✓ Validate Paths", self.validate_paths,
                  bootstyle="warning").pack(side=tk.LEFT, padx=5)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_training_tab(self):
        training_frame = self.UI.Frame(self.notebook, padding=10)
        self.notebook.add(training_frame, text="🚀 Training")

        control_section = self.UI.LabelFrame(training_frame, text="🎮 Training Control")
        control_section.pack(fill=tk.X, padx=10, pady=10)

        button_frame = self.UI.Frame(control_section)
        button_frame.pack(padx=15, pady=15)

        self.start_btn = self._btn(button_frame, "▶️ Start Training", self.start_training,
                                   bootstyle="success", width=25)
        self.start_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.stop_btn = self._btn(button_frame, "⏹️ Stop Training", self.stop_training,
                                  bootstyle="danger", width=25)
        self.stop_btn.pack(side=tk.LEFT, padx=10, pady=10)
        self.stop_btn.config(state=tk.DISABLED)

        progress_section = self.UI.LabelFrame(training_frame, text="📊 Training Progress")
        progress_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        epoch_frame = self.UI.Frame(progress_section)
        epoch_frame.pack(fill=tk.X, pady=5)

        self.epoch_label = self.UI.Label(epoch_frame, text="Epoch: 0/0")
        self.epoch_label.pack(side=tk.LEFT, padx=5)

        self.epoch_progress = self.UI.Progressbar(epoch_frame, mode="determinate", length=400)
        self.epoch_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        metrics_frame = self.UI.LabelFrame(progress_section, text="📈 Current Metrics")
        metrics_frame.pack(fill=tk.X, pady=10)

        metrics_grid = self.UI.Frame(metrics_frame)
        metrics_grid.pack(padx=15, pady=15)

        self.UI.Label(metrics_grid, text="Detection Loss:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=2)
        self.det_loss_var = tk.StringVar(value="N/A")
        self.UI.Label(metrics_grid, textvariable=self.det_loss_var, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)

        self.UI.Label(metrics_grid, text="Recognition Loss:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=2)
        self.rec_loss_var = tk.StringVar(value="N/A")
        self.UI.Label(metrics_grid, textvariable=self.rec_loss_var, font=("TkDefaultFont", 10, "bold")).grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)

        self.UI.Label(metrics_grid, text="Validation Loss:").grid(row=0, column=2, sticky=tk.W, padx=10, pady=2)
        self.val_loss_var = tk.StringVar(value="N/A")
        self.UI.Label(metrics_grid, textvariable=self.val_loss_var, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=3, sticky=tk.W, padx=10, pady=2)

        self.UI.Label(metrics_grid, text="Accuracy:").grid(row=1, column=2, sticky=tk.W, padx=10, pady=2)
        self.accuracy_var = tk.StringVar(value="N/A")
        self.UI.Label(metrics_grid, textvariable=self.accuracy_var, font=("TkDefaultFont", 10, "bold")).grid(row=1, column=3, sticky=tk.W, padx=10, pady=2)

        console_section = self.UI.LabelFrame(progress_section, text="💻 Training Output")
        console_section.pack(fill=tk.BOTH, expand=True, pady=10)

        self.training_console = scrolledtext.ScrolledText(
            console_section,
            height=15,
            bg="#1e1e1e",
            fg="#00ff00",
            font=("Consolas", 10),
            insertbackground="#00ff00",
            selectbackground="#264f78",
            selectforeground="white",
        )
        self.training_console.pack(fill=tk.BOTH, expand=True)
        self.training_console.config(state=tk.DISABLED)

    def create_log_tab(self):
        log_frame = self.UI.Frame(self.notebook, padding=10)
        self.notebook.add(log_frame, text="📋 Logs")

        control_frame = self.UI.Frame(log_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        self._btn(control_frame, "🗑️ Clear Logs", self.clear_logs,
                  bootstyle="warning-outline").pack(side=tk.LEFT, padx=5)
        self._btn(control_frame, "💾 Save Logs", self.save_logs,
                  bootstyle="info-outline").pack(side=tk.LEFT, padx=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=30,
            wrap=tk.WORD,
            bg="#f8f9fa",
            fg="#2c3e50",
            font=("Consolas", 9),
            relief=tk.FLAT,
            borderwidth=2,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # -------------------------
    # Validation / config IO
    # -------------------------
    def validate_paths(self):
        errors = []

        train_img = self.train_img_var.get().strip()
        train_lbl = self.train_label_var.get().strip()
        val_img = self.val_img_var.get().strip()
        val_lbl = self.val_label_var.get().strip()
        out_dir = self.output_dir_var.get().strip()

        if not train_img or not os.path.exists(train_img):
            errors.append(f"Training images directory not found: {train_img}")
        if not train_lbl or not os.path.exists(train_lbl):
            errors.append(f"Training labels file not found: {train_lbl}")

        if val_img and not os.path.exists(val_img):
            errors.append(f"Validation images directory not found: {val_img}")
        if val_lbl and not os.path.exists(val_lbl):
            errors.append(f"Validation labels file not found: {val_lbl}")
        if not out_dir:
            errors.append("Output directory is empty.")
        else:
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory '{out_dir}': {e}")


        ptype = self.pretrained_type_var.get()
        if ptype == "fcmae":
            ck = self.fcmae_ckpt_var.get().strip()
            if not ck or not os.path.exists(ck):
                errors.append(f"FCMAE checkpoint not found: {ck}")
        if ptype == "ocr":
            ck = self.ocr_ckpt_var.get().strip()
            if not ck or not os.path.exists(ck):
                errors.append(f"OCR checkpoint not found: {ck}")

        if errors:
            messagebox.showerror("Validation Failed", "\n\n".join(errors))
            return False

        messagebox.showinfo("Validation Successful", "All paths look valid!")
        return True

    def save_config(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return

        config = {
            "training_mode": self.training_mode_var.get(),
            "train_img": self.train_img_var.get(),
            "train_label": self.train_label_var.get(),
            "val_img": self.val_img_var.get(),
            "val_label": self.val_label_var.get(),
            "output_dir": self.output_dir_var.get(),
            "pretrained_type": self.pretrained_type_var.get(),
            "fcmae_ckpt": self.fcmae_ckpt_var.get(),
            "ocr_ckpt": self.ocr_ckpt_var.get(),
            "det_imgH": self.det_imgH_var.get(),
            "det_imgW": self.det_imgW_var.get(),
            "rec_imgH": self.rec_imgH_var.get(),
            "rec_imgW": self.rec_imgW_var.get(),
            "epochs": self.epochs_var.get(),
            "batch_size": self.batch_size_var.get(),
            "learning_rate": self.lr_var.get(),
            "patience": self.patience_var.get(),
            "maintain_aspect": self.maintain_aspect_var.get(),
            "visualize": self.visualize_var.get(),
            "ids_to_train": self.ids_var.get(),
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            messagebox.showinfo("Success", f"Configuration saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def load_config(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            self.training_mode_var.set(config.get("training_mode", "det_rec"))
            self.train_img_var.set(config.get("train_img", ""))
            self.train_label_var.set(config.get("train_label", ""))
            self.val_img_var.set(config.get("val_img", ""))
            self.val_label_var.set(config.get("val_label", ""))
            self.output_dir_var.set(config.get("output_dir", ""))
            self.pretrained_type_var.set(config.get("pretrained_type", "fcmae"))
            self.fcmae_ckpt_var.set(config.get("fcmae_ckpt", ""))
            self.ocr_ckpt_var.set(config.get("ocr_ckpt", ""))
            self.det_imgH_var.set(config.get("det_imgH", "640"))
            self.det_imgW_var.set(config.get("det_imgW", "640"))
            self.rec_imgH_var.set(config.get("rec_imgH", "96"))
            self.rec_imgW_var.set(config.get("rec_imgW", "198"))
            self.epochs_var.set(config.get("epochs", "100"))
            self.batch_size_var.set(config.get("batch_size", "8"))
            self.lr_var.set(config.get("learning_rate", "0.001"))
            self.patience_var.set(config.get("patience", "10"))
            self.maintain_aspect_var.set(bool(config.get("maintain_aspect", True)))
            self.visualize_var.set(bool(config.get("visualize", False)))
            self.ids_var.set(config.get("ids_to_train", ""))

            messagebox.showinfo("Success", f"Configuration loaded from {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {e}")

    # -------------------------
    # Training control
    # -------------------------
    def start_training(self):
        if self.is_training:
            messagebox.showwarning("Training in Progress", "Training is already running!")
            return

        if not self.validate_paths():
            return

        self.is_training = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Training in progress...")

        self.training_console.config(state=tk.NORMAL)
        self.training_console.delete(1.0, tk.END)
        self.training_console.config(state=tk.DISABLED)

        self.epoch_progress["value"] = 0
        self.epoch_label.config(text="Epoch: 0/0")

        self.training_thread = threading.Thread(target=self.run_training, daemon=True)
        self.training_thread.start()

    def stop_training(self):
        if messagebox.askyesno("Stop Training", "Are you sure you want to stop the training?"):
            self.is_training = False
            self.status_var.set("Stopping training...")
            self.log_message("Training stop requested by user...")

    def _safe_script_dir(self):
        """Handle environments where __file__ doesn't exist."""
        try:
            return os.path.dirname(os.path.abspath(__file__))
        except NameError:
            return os.getcwd()

    def _parse_ids(self):
        """Return ids_to_train as list[int] or None (consistent type)."""
        ids_str = self.ids_var.get().strip()
        if not ids_str:
            return None
        return [int(x.strip()) for x in ids_str.split(",") if x.strip()]

    def run_training(self):
        try:
            self.log_message("=" * 60)
            self.log_message(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_message("=" * 60)

            ids_to_train = self._parse_ids()

            # Enforce checkpoint selection consistency
            ptype = self.pretrained_type_var.get().strip().lower()
            fcmae_ckpt = self.fcmae_ckpt_var.get().strip() if ptype == "fcmae" else None
            ocr_ckpt = self.ocr_ckpt_var.get().strip() if ptype == "ocr" else None

            config = {
                "training_mode": self.training_mode_var.get(),
                "train_img": self.train_img_var.get(),
                "train_label": self.train_label_var.get(),
                "val_img": self.val_img_var.get() if self.val_img_var.get().strip() else None,
                "val_label": self.val_label_var.get() if self.val_label_var.get().strip() else None,
                "output_dir": self.output_dir_var.get(),
                "pretrained_type": ptype,
                "fcmae_ckpt": fcmae_ckpt,
                "ocr_ckpt": ocr_ckpt,
                "det_imgH": int(self.det_imgH_var.get()),
                "det_imgW": int(self.det_imgW_var.get()),
                "rec_imgH": int(self.rec_imgH_var.get()),
                "rec_imgW": int(self.rec_imgW_var.get()),
                "epochs": int(self.epochs_var.get()),
                "batch_size": int(self.batch_size_var.get()),
                "learning_rate": float(self.lr_var.get()),
                "patience": int(self.patience_var.get()),
                "maintain_aspect": bool(self.maintain_aspect_var.get()),
                "visualize": bool(self.visualize_var.get()),
                "ids_to_train": ids_to_train,
            }

            self.log_message("\nConfiguration:")
            for k in ("training_mode", "train_img", "train_label", "output_dir", "epochs", "batch_size", "learning_rate", "ids_to_train"):
                self.log_message(f"  {k}: {config[k]}")
            self.log_message("")

            self.log_message("Importing training modules...")

            script_dir = self._safe_script_dir()
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)

            from training_wrapper import run_training_with_config  # noqa: F401

            self.log_message("Starting training process...\n")

            # IMPORTANT: This assumes stop_check_callback returns True when it should stop.
            success = run_training_with_config(
                config,
                log_callback=self.log_message,
                stop_check_callback=lambda: (not self.is_training),
                # OPTIONAL: if your wrapper supports it, you can pass progress callback:
                # progress_callback=self.update_progress,
            )

            if success:
                self.log_message("\n" + "=" * 60)
                self.log_message("Training completed successfully!")
                self.log_message("=" * 60)
            else:
                self.log_message("\n" + "=" * 60)
                self.log_message("Training failed or was stopped!")
                self.log_message("=" * 60)

            self.root.after(0, lambda: messagebox.showinfo("Training Complete", "Training has completed!"))

        except Exception as e:
            error_msg = f"Error during training: {e}"
            self.log_message(f"\n{error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Training Error", error_msg))

        finally:
            self.is_training = False
            self.root.after(0, self.reset_training_ui)

    def reset_training_ui(self):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Ready")

    # -------------------------
    # Logging & UI updates
    # -------------------------
    def update_progress(self, current_epoch, total_epochs):
        if total_epochs <= 0:
            return
        progress = (current_epoch / total_epochs) * 100
        self.root.after(0, lambda: self.epoch_progress.config(value=progress))
        self.root.after(0, lambda: self.epoch_label.config(text=f"Epoch: {current_epoch}/{total_epochs}"))

    def log_message(self, message):
        self.log_queue.put(str(message))

    def check_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()

                self.training_console.config(state=tk.NORMAL)
                self.training_console.insert(tk.END, message + "\n")
                self.training_console.see(tk.END)
                self.training_console.config(state=tk.DISABLED)

                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)

        except queue.Empty:
            pass

        self.root.after(100, self.check_log_queue)

    def clear_logs(self):
        self.log_text.delete(1.0, tk.END)

    def save_logs(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.log_text.get(1.0, tk.END))
            messagebox.showinfo("Success", f"Logs saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save logs: {e}")


def main():
    if MODERN_THEME:
        # Use dark theme - options: darkly, superhero, cyborg, solar, vapor
        root = ttkb.Window(themename="darkly")
    else:
        root = tk.Tk()

    app = OCRTrainingGUI(root)

    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
