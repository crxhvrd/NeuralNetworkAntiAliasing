"""
NNAA Shader Studio - A unified GUI for training, converting, and testing
neural network anti-aliasing models for ReShade.

Requires: tensorflow, numpy, Pillow
Optional: matplotlib (for loss chart)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import os
import sys
import json
import time
import traceback

# ============================================================================
# Theme
# ============================================================================
COLORS = {
    'bg':           '#1e1e2e',
    'bg_secondary': '#282840',
    'bg_input':     '#313150',
    'fg':           '#cdd6f4',
    'fg_dim':       '#6c7086',
    'fg_bright':    '#ffffff',
    'accent':       '#b4befe',
    'accent_hover': '#cba6f7',
    'accent_bg':    '#45475a',
    'success':      '#a6e3a1',
    'error':        '#f38ba8',
    'warning':      '#fab387',
    'border':       '#45475a',
    'tab_active':   '#b4befe',
    'tab_inactive': '#6c7086',
    'console_bg':   '#11111b',
    'console_fg':   '#a6adc8',
    'button_bg':    '#7c3aed',
    'button_fg':    '#ffffff',
    'button_hover': '#9333ea',
    'stop_bg':      '#dc2626',
    'stop_hover':   '#ef4444',
    'chart_line':   '#b4befe',
    'chart_best':   '#a6e3a1',
}

FONT_FAMILY = 'Segoe UI'
FONT_LABEL = (FONT_FAMILY, 10)
FONT_HEADING = (FONT_FAMILY, 12, 'bold')
FONT_TITLE = (FONT_FAMILY, 18, 'bold')
FONT_CONSOLE = ('Consolas', 9)
FONT_BUTTON = (FONT_FAMILY, 10, 'bold')
FONT_TAB = (FONT_FAMILY, 11, 'bold')
FONT_SMALL = (FONT_FAMILY, 9)

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.nnaa_studio_settings.json')

# ============================================================================
# Settings persistence
# ============================================================================

def load_settings():
    """Load saved settings from JSON file."""
    try:
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(data):
    """Save settings to JSON file."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


# ============================================================================
# Styled widgets
# ============================================================================

class StyledEntry(tk.Entry):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault('bg', COLORS['bg_input'])
        kwargs.setdefault('fg', COLORS['fg'])
        kwargs.setdefault('insertbackground', COLORS['fg'])
        kwargs.setdefault('relief', 'flat')
        kwargs.setdefault('font', FONT_LABEL)
        kwargs.setdefault('highlightthickness', 1)
        kwargs.setdefault('highlightbackground', COLORS['border'])
        kwargs.setdefault('highlightcolor', COLORS['accent'])
        super().__init__(parent, **kwargs)


class StyledButton(tk.Button):
    def __init__(self, parent, accent=True, danger=False, **kwargs):
        if danger:
            bg, hover = COLORS['stop_bg'], COLORS['stop_hover']
        elif accent:
            bg, hover = COLORS['button_bg'], COLORS['button_hover']
        else:
            bg, hover = COLORS['accent_bg'], COLORS['border']
        
        kwargs.setdefault('bg', bg)
        kwargs.setdefault('fg', COLORS['button_fg'])
        kwargs.setdefault('activebackground', hover)
        kwargs.setdefault('activeforeground', COLORS['button_fg'])
        kwargs.setdefault('relief', 'flat')
        kwargs.setdefault('font', FONT_BUTTON)
        kwargs.setdefault('cursor', 'hand2')
        kwargs.setdefault('padx', 16)
        kwargs.setdefault('pady', 6)
        kwargs.setdefault('bd', 0)
        super().__init__(parent, **kwargs)
        
        self._bg, self._hover = bg, hover
        self.bind('<Enter>', lambda e: self.config(bg=self._hover))
        self.bind('<Leave>', lambda e: self.config(bg=self._bg))


class StyledLabel(tk.Label):
    def __init__(self, parent, heading=False, dim=False, **kwargs):
        if heading:
            kwargs.setdefault('font', FONT_HEADING)
            kwargs.setdefault('fg', COLORS['fg_bright'])
        elif dim:
            kwargs.setdefault('font', FONT_LABEL)
            kwargs.setdefault('fg', COLORS['fg_dim'])
        else:
            kwargs.setdefault('font', FONT_LABEL)
            kwargs.setdefault('fg', COLORS['fg'])
        kwargs.setdefault('bg', COLORS['bg'])
        super().__init__(parent, **kwargs)


class ThreadSafeConsole(tk.Frame):
    """A console text widget that is safe to log to from any thread."""
    
    def __init__(self, parent, height=12, **kwargs):
        super().__init__(parent, bg=COLORS['bg'])
        self._queue = queue.Queue()
        
        self._text = tk.Text(self, bg=COLORS['console_bg'], fg=COLORS['console_fg'],
                             font=FONT_CONSOLE, relief='flat',
                             insertbackground=COLORS['console_fg'],
                             selectbackground=COLORS['accent_bg'],
                             highlightthickness=1,
                             highlightbackground=COLORS['border'],
                             highlightcolor=COLORS['border'],
                             state='disabled', wrap='word', height=height)
        scrollbar = tk.Scrollbar(self, command=self._text.yview,
                                 bg=COLORS['bg_secondary'], troughcolor=COLORS['console_bg'],
                                 highlightthickness=0, bd=0)
        self._text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self._text.pack(fill='both', expand=True)

        for tag, color in [('success', COLORS['success']), ('error', COLORS['error']),
                           ('warning', COLORS['warning']), ('accent', COLORS['accent'])]:
            self._text.tag_configure(tag, foreground=color)

        self._poll()

    def _poll(self):
        while not self._queue.empty():
            msg, tag = self._queue.get_nowait()
            self._text.config(state='normal')
            if tag:
                self._text.insert('end', msg, tag)
            else:
                self._text.insert('end', msg)
            self._text.see('end')
            self._text.config(state='disabled')
        self.after(80, self._poll)

    def log(self, msg, tag=None):
        """Thread-safe log method — can be called from any thread."""
        self._queue.put((msg, tag))

    def clear(self):
        """Clear the console (main thread only)."""
        self._text.config(state='normal')
        self._text.delete('1.0', 'end')
        self._text.config(state='disabled')


def make_path_row(parent, label_text, var, row, browse_type='folder'):
    """Create a label + entry + browse button row."""
    StyledLabel(parent, text=label_text).grid(row=row, column=0, sticky='w', pady=(6, 2))
    entry = StyledEntry(parent, textvariable=var)
    entry.grid(row=row, column=1, sticky='ew', padx=(8, 4), pady=(6, 2))

    def browse():
        if browse_type == 'folder':
            path = filedialog.askdirectory()
        elif browse_type == 'open':
            path = filedialog.askopenfilename(filetypes=[
                ("Keras Models", "*.keras *.h5"), ("All Files", "*.*")])
        elif browse_type == 'open_image':
            path = filedialog.askopenfilename(filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")])
        elif browse_type == 'save':
            path = filedialog.asksaveasfilename(
                defaultextension='.fx',
                filetypes=[("ReShade FX", "*.fx"), ("All Files", "*.*")])
        else:
            path = None
        if path:
            var.set(path)

    StyledButton(parent, text="Browse", accent=False, command=browse).grid(
        row=row, column=2, padx=(0, 4), pady=(6, 2))
    return entry


def make_param_row(parent, label_text, var, row, col_offset=0):
    """Create a label + entry row for a hyperparameter."""
    StyledLabel(parent, text=label_text).grid(
        row=row, column=col_offset, sticky='w', pady=(4, 2), padx=(0, 4))
    entry = StyledEntry(parent, textvariable=var, width=14)
    entry.grid(row=row, column=col_offset + 1, sticky='w', pady=(4, 2), padx=(0, 16))
    return entry


def format_time(seconds):
    """Format seconds as H:MM:SS or M:SS."""
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ============================================================================
# Loss chart widget (canvas-based, no matplotlib dependency)
# ============================================================================

class LossChart(tk.Canvas):
    """A simple loss sparkline chart drawn on a canvas."""
    
    def __init__(self, parent, height=80, **kwargs):
        super().__init__(parent, bg=COLORS['console_bg'], highlightthickness=1,
                         highlightbackground=COLORS['border'], height=height, **kwargs)
        self.losses = []
        self.best_loss = None
        self._pad = 8

    def add_loss(self, loss_val):
        self.losses.append(loss_val)
        if self.best_loss is None or loss_val < self.best_loss:
            self.best_loss = loss_val
        self._redraw()

    def clear_data(self):
        self.losses = []
        self.best_loss = None
        self.delete('all')

    def _redraw(self):
        self.delete('all')
        if len(self.losses) < 2:
            return

        w = self.winfo_width()
        h = self.winfo_height()
        if w < 20 or h < 20:
            return

        pad = self._pad
        plot_w = w - 2 * pad
        plot_h = h - 2 * pad

        min_v = min(self.losses)
        max_v = max(self.losses)
        val_range = max_v - min_v if max_v > min_v else 1e-8

        n = len(self.losses)
        points = []
        for i, v in enumerate(self.losses):
            x = pad + (i / (n - 1)) * plot_w
            y = pad + (1 - (v - min_v) / val_range) * plot_h
            points.append((x, y))

        # Draw best loss line
        if self.best_loss is not None:
            by = pad + (1 - (self.best_loss - min_v) / val_range) * plot_h
            self.create_line(pad, by, w - pad, by, fill=COLORS['chart_best'],
                           dash=(4, 4), width=1)
            self.create_text(w - pad - 2, by - 8, text=f"best: {self.best_loss:.6f}",
                           fill=COLORS['chart_best'], font=FONT_SMALL, anchor='e')

        # Draw loss curve
        for i in range(len(points) - 1):
            self.create_line(points[i][0], points[i][1],
                           points[i+1][0], points[i+1][1],
                           fill=COLORS['chart_line'], width=2, smooth=True)

        # Current value label
        self.create_text(pad + 2, pad, text=f"{max_v:.6f}",
                        fill=COLORS['fg_dim'], font=FONT_SMALL, anchor='nw')
        self.create_text(pad + 2, h - pad, text=f"{min_v:.6f}",
                        fill=COLORS['fg_dim'], font=FONT_SMALL, anchor='sw')


# ============================================================================
# Tab: Train
# ============================================================================

class TrainTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=COLORS['bg'])
        self.app = app
        self.training_thread = None
        self.stop_event = threading.Event()
        self.start_time = None

        settings = app.settings

        # ── Dataset paths ──
        section = tk.Frame(self, bg=COLORS['bg'])
        section.pack(fill='x', padx=20, pady=(12, 0))
        StyledLabel(section, text="📁  Dataset Configuration", heading=True).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 4))
        section.columnconfigure(1, weight=1)

        self.train_bad = tk.StringVar(value=settings.get('train_bad', 'data/train/bad/1280x720'))
        self.train_good = tk.StringVar(value=settings.get('train_good', 'data/train/fixed/1280x720'))
        self.test_bad = tk.StringVar(value=settings.get('test_bad', 'data/test/bad/2560x1440'))
        self.test_good = tk.StringVar(value=settings.get('test_good', 'data/test/fixed/2560x1440'))

        make_path_row(section, "Train — No AA:", self.train_bad, 1)
        make_path_row(section, "Train — With AA:", self.train_good, 2)
        make_path_row(section, "Test — No AA:", self.test_bad, 3)
        make_path_row(section, "Test — With AA:", self.test_good, 4)

        # ── Hyperparameters ──
        params = tk.Frame(self, bg=COLORS['bg'])
        params.pack(fill='x', padx=20, pady=(12, 0))
        StyledLabel(params, text="⚙  Hyperparameters", heading=True).grid(
            row=0, column=0, columnspan=8, sticky='w', pady=(0, 4))

        self.lr = tk.StringVar(value=settings.get('lr', '0.00001'))
        self.batch_size = tk.StringVar(value=settings.get('batch_size', '16'))
        self.test_batch = tk.StringVar(value=settings.get('test_batch', '4'))
        self.epochs_per_run = tk.StringVar(value=settings.get('epochs_per_run', '5'))
        self.patch_size = tk.StringVar(value=settings.get('patch_size', '128'))
        self.patience = tk.StringVar(value=settings.get('patience', '20'))
        self.augment = tk.BooleanVar(value=settings.get('augment', True))

        make_param_row(params, "Learning Rate:", self.lr, 1, 0)
        make_param_row(params, "Train Batch:", self.batch_size, 1, 2)
        make_param_row(params, "Test Batch:", self.test_batch, 1, 4)
        make_param_row(params, "Epochs/Run:", self.epochs_per_run, 2, 0)
        make_param_row(params, "Patch Size:", self.patch_size, 2, 2)
        make_param_row(params, "Patience:", self.patience, 2, 4)

        # Augment checkbox
        aug_cb = tk.Checkbutton(params, text="Augment (flips)", variable=self.augment,
                                bg=COLORS['bg'], fg=COLORS['fg'], selectcolor=COLORS['bg_input'],
                                activebackground=COLORS['bg'], activeforeground=COLORS['fg'],
                                font=FONT_LABEL, cursor='hand2')
        aug_cb.grid(row=2, column=6, columnspan=2, sticky='w', padx=(0, 4), pady=(4, 2))

        # ── Model output ──
        model_frame = tk.Frame(self, bg=COLORS['bg'])
        model_frame.pack(fill='x', padx=20, pady=(12, 0))
        model_frame.columnconfigure(1, weight=1)
        StyledLabel(model_frame, text="💾  Model Output", heading=True).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 4))

        self.model_name = tk.StringVar(value=settings.get('model_name', 'nnaa'))
        self.model_dir = tk.StringVar(value=settings.get('model_dir', '..'))
        make_param_row(model_frame, "Model Name:", self.model_name, 1, 0)
        StyledLabel(model_frame, text="Output Dir:").grid(row=2, column=0, sticky='w', pady=(4, 2))
        StyledEntry(model_frame, textvariable=self.model_dir).grid(
            row=2, column=1, sticky='ew', padx=(8, 4), pady=(4, 2))
        StyledButton(model_frame, text="Browse", accent=False,
                     command=lambda: self.model_dir.set(
                         filedialog.askdirectory() or self.model_dir.get()
                     )).grid(row=2, column=2, padx=(0, 4), pady=(4, 2))

        # ── Buttons + status ──
        btn_frame = tk.Frame(self, bg=COLORS['bg'])
        btn_frame.pack(fill='x', padx=20, pady=(12, 0))

        self.start_btn = StyledButton(btn_frame, text="▶  Start Training", command=self.start_training)
        self.start_btn.pack(side='left', padx=(0, 8))
        self.stop_btn = StyledButton(btn_frame, text="■  Stop", danger=True, command=self.stop_training)
        self.stop_btn.pack(side='left')
        self.stop_btn.config(state='disabled')
        
        self.status_label = StyledLabel(btn_frame, text="Idle", dim=True)
        self.status_label.pack(side='right')
        self.timer_label = StyledLabel(btn_frame, text="", dim=True)
        self.timer_label.pack(side='right', padx=(0, 12))

        # ── Loss chart ──
        chart_frame = tk.Frame(self, bg=COLORS['bg'])
        chart_frame.pack(fill='x', padx=20, pady=(8, 0))
        StyledLabel(chart_frame, text="📈  Loss", heading=True).pack(anchor='w', pady=(0, 2))
        self.loss_chart = LossChart(chart_frame, height=80)
        self.loss_chart.pack(fill='x')

        # ── Console ──
        console_frame = tk.Frame(self, bg=COLORS['bg'])
        console_frame.pack(fill='both', expand=True, padx=20, pady=(8, 12))
        StyledLabel(console_frame, text="📋  Training Log", heading=True).pack(anchor='w', pady=(0, 2))
        self.console = ThreadSafeConsole(console_frame, height=8)
        self.console.pack(fill='both', expand=True)

        self.console.log("Welcome to NNAA Shader Studio!\n", 'accent')
        self.console.log("Configure your dataset paths and click Start Training.\n")

    def _save_settings(self):
        """Persist current field values."""
        self.app.settings.update({
            'train_bad': self.train_bad.get(),
            'train_good': self.train_good.get(),
            'test_bad': self.test_bad.get(),
            'test_good': self.test_good.get(),
            'lr': self.lr.get(),
            'batch_size': self.batch_size.get(),
            'test_batch': self.test_batch.get(),
            'epochs_per_run': self.epochs_per_run.get(),
            'patch_size': self.patch_size.get(),
            'patience': self.patience.get(),
            'augment': self.augment.get(),
            'model_name': self.model_name.get(),
            'model_dir': self.model_dir.get(),
        })
        save_settings(self.app.settings)

    def _validate_paths(self):
        """Check that dataset directories exist and contain matching files."""
        paths = {
            'Train No AA': self.train_bad.get(),
            'Train With AA': self.train_good.get(),
            'Test No AA': self.test_bad.get(),
            'Test With AA': self.test_good.get(),
        }
        for name, path in paths.items():
            if not os.path.isdir(path):
                messagebox.showerror("Invalid Path",
                    f'"{name}" directory does not exist:\n{path}')
                return False
        
        # Check for matching files in train set
        train_bad_files = set(os.listdir(self.train_bad.get()))
        train_good_files = set(os.listdir(self.train_good.get()))
        common = train_bad_files & train_good_files
        if not common:
            messagebox.showerror("No Matching Files",
                "No matching filenames found between the Train directories.\n"
                "Both folders must contain images with the same filenames.")
            return False
        return True

    def start_training(self):
        if not self._validate_paths():
            return
        self._save_settings()
        
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.stop_event.clear()
        self.start_time = time.time()
        self.status_label.config(text="Training...", fg=COLORS['success'])
        self.console.clear()
        self.loss_chart.clear_data()
        self.console.log("Starting training...\n", 'accent')
        self._update_timer()

        self.training_thread = threading.Thread(target=self._train_worker, daemon=True)
        self.training_thread.start()

    def stop_training(self):
        self.stop_event.set()
        self.console.log("\n⏹  Stop requested. Finishing current epoch...\n", 'warning')

    def _update_timer(self):
        """Update the elapsed time display."""
        if self.start_time and not self.stop_event.is_set():
            elapsed = time.time() - self.start_time
            self.timer_label.config(text=f"⏱ {format_time(elapsed)}")
            self.after(1000, self._update_timer)

    def _train_worker(self):
        try:
            self.console.log("Importing TensorFlow... ", None)
            import tensorflow as tf
            import numpy as np
            self.console.log("OK\n", 'success')

            script_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, script_dir)
            from nnaa_train import NnaaDataset

            lr = float(self.lr.get())
            batch_size = int(self.batch_size.get())
            test_batch = int(self.test_batch.get())
            epochs = int(self.epochs_per_run.get())
            patch_size = int(self.patch_size.get())
            patience_val = int(self.patience.get())
            do_augment = self.augment.get()
            model_name = self.model_name.get()
            models_path = self.model_dir.get()

            model_directory = os.path.join(models_path, model_name)
            model_path = os.path.join(model_directory, model_name) + ".keras"

            self.console.log(f"Model path: {model_path}\n")
            self.console.log(f"LR: {lr}  |  Batch: {batch_size}/{test_batch}  |  Epochs/run: {epochs}\n")
            self.console.log(f"Patch: {patch_size if patch_size > 0 else 'full'}  |  Augment: {do_augment}  |  Patience: {patience_val}\n\n")

            os.makedirs(model_directory, exist_ok=True)
            loss_fn = tf.keras.losses.MeanSquaredError()

            if os.path.isfile(model_path):
                self.console.log("Loading existing model... ", None)
                model = tf.keras.models.load_model(model_path)
                self.console.log("OK\n", 'success')
            else:
                self.console.log("Creating new model... ", None)
                input_layer = tf.keras.Input(shape=(None, None, 1), name="img")
                x = tf.keras.layers.PReLU(shared_axes=[1, 2])(
                    tf.keras.layers.Conv2D(32, 8, strides=2, padding='same')(input_layer))
                x = tf.keras.layers.PReLU(shared_axes=[1, 2])(
                    tf.keras.layers.Conv2D(32, 3, strides=1, padding='same')(x))
                x = tf.keras.layers.PReLU(shared_axes=[1, 2])(
                    tf.keras.layers.Conv2D(32, 3, strides=1, padding='same')(x))
                x = tf.keras.layers.PReLU(shared_axes=[1, 2])(
                    tf.keras.layers.Conv2D(32, 3, strides=1, padding='same')(x))
                output = tf.keras.layers.Conv2DTranspose(1, 2, strides=2, padding='same',
                                                         name='conv2d_final')(x)
                model = tf.keras.Model(input_layer, output, name=model_name)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                              loss=loss_fn, metrics=['mean_squared_error'])
                self.console.log("OK\n", 'success')

            if model.optimizer.learning_rate != lr:
                model.optimizer.learning_rate = lr
                self.console.log(f"Updated learning rate to {lr}\n", 'warning')

            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            self.console.log('\n'.join(summary_lines) + '\n\n', None)

            self.console.log("Loading dataset... ", None)
            train_dataset = NnaaDataset(self.train_bad.get(), self.train_good.get(),
                                        batch_size, use_cache=True,
                                        patch_size=patch_size, augment=do_augment)
            test_dataset = NnaaDataset(self.test_bad.get(), self.test_good.get(),
                                       test_batch, use_cache=True,
                                       patch_size=0, augment=False)
            self.console.log(
                f"OK ({len(train_dataset)} train, {len(test_dataset)} test batches, "
                f"{len(train_dataset.img_names)} images)\n", 'success')

            # LR scheduler — halve LR when training loss plateaus
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=3, min_lr=1e-7, verbose=0)

            best_error = float('inf')
            best_path = os.path.join(model_directory, "bestError.npy")
            if os.path.isfile(best_path):
                best_error = np.load(best_path).item()
            self.console.log(f"Best error so far: {best_error}\n\n")

            no_improve = 0
            run = 0
            while not self.stop_event.is_set():
                run += 1
                self.console.log(f"━━━ Run {run} ━━━\n", 'accent')

                history = model.fit(train_dataset, epochs=epochs, verbose=0,
                                    callbacks=[lr_scheduler])
                for epoch_idx, loss_val in enumerate(history.history['loss']):
                    self.console.log(f"  Epoch {epoch_idx + 1}/{epochs} — loss: {loss_val:.8f}\n")
                    self.after(0, lambda v=loss_val: self.loss_chart.add_loss(v))

                current_lr = float(model.optimizer.learning_rate)
                self.console.log(f"  LR: {current_lr:.2e}\n", 'accent')

                if self.stop_event.is_set():
                    break

                eval_result = model.evaluate(test_dataset, verbose=0)
                self.console.log(f"  Eval — loss: {eval_result[0]:.8f}, mse: {eval_result[1]:.8f}\n")

                if eval_result[0] < best_error:
                    best_error = eval_result[0]
                    np.save(best_path, best_error)
                    model.save(model_path)
                    self.console.log(f"  ★ New best! Saved (error: {best_error:.8f})\n", 'success')
                    no_improve = 0
                else:
                    no_improve += 1
                    self.console.log(
                        f"  No improvement ({no_improve}/{patience_val}, best: {best_error:.8f})\n", 'warning')

                    if patience_val > 0 and no_improve >= patience_val:
                        self.console.log(
                            f"\n⏹  Early stopping: no improvement for {patience_val} runs.\n", 'warning')
                        break
                self.console.log('\n')

            self.console.log("Training stopped.\n", 'accent')

        except Exception as e:
            self.console.log(f"\n✗ Error: {e}\n", 'error')
            self.console.log(traceback.format_exc() + '\n', 'error')
        finally:
            self.after(0, self._training_finished)

    def _training_finished(self):
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.status_label.config(text="Idle", fg=COLORS['fg_dim'])
        self.timer_label.config(text=f"Done in {format_time(elapsed)}")
        self.start_time = None


# ============================================================================
# Tab: Convert
# ============================================================================

class ConvertTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=COLORS['bg'])
        self.app = app

        section = tk.Frame(self, bg=COLORS['bg'])
        section.pack(fill='x', padx=20, pady=(20, 0))
        section.columnconfigure(1, weight=1)
        StyledLabel(section, text="🔄  Keras → ReShade FX Converter", heading=True).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))

        settings = app.settings
        self.model_path = tk.StringVar(value=settings.get('convert_model', 'nnaa.keras'))
        self.output_path = tk.StringVar(value=settings.get('convert_output', 'out_nnaa.fx'))
        make_path_row(section, "Keras Model:", self.model_path, 1, browse_type='open')
        make_path_row(section, "Output .fx:", self.output_path, 2, browse_type='save')

        btn_frame = tk.Frame(self, bg=COLORS['bg'])
        btn_frame.pack(fill='x', padx=20, pady=(16, 0))
        self.convert_btn = StyledButton(btn_frame, text="⚡  Convert to Shader",
                                        command=self.do_convert)
        self.convert_btn.pack(side='left')
        self.status = StyledLabel(btn_frame, text="", dim=True)
        self.status.pack(side='left', padx=(12, 0))

        console_frame = tk.Frame(self, bg=COLORS['bg'])
        console_frame.pack(fill='both', expand=True, padx=20, pady=(12, 16))
        StyledLabel(console_frame, text="📋  Conversion Log", heading=True).pack(anchor='w', pady=(0, 4))
        self.console = ThreadSafeConsole(console_frame, height=16)
        self.console.pack(fill='both', expand=True)
        self.console.log("Select a .keras model file and output path, then click Convert.\n")

    def do_convert(self):
        self.app.settings.update({
            'convert_model': self.model_path.get(),
            'convert_output': self.output_path.get(),
        })
        save_settings(self.app.settings)
        
        self.convert_btn.config(state='disabled')
        self.status.config(text="Converting...", fg=COLORS['warning'])
        self.console.clear()
        threading.Thread(target=self._convert_worker, daemon=True).start()

    def _convert_worker(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, script_dir)
            import importlib
            import convert
            importlib.reload(convert)

            result = convert.convert_model(
                self.model_path.get(),
                self.output_path.get(),
                log_fn=self.console.log
            )
            self.after(0, lambda: self.status.config(text="Done!", fg=COLORS['success']))

        except (FileNotFoundError, ValueError) as e:
            self.console.log(f"\n✗ {e}\n", 'error')
            self.after(0, lambda: self.status.config(text="Failed", fg=COLORS['error']))
        except Exception as e:
            self.console.log(f"\n✗ Error: {e}\n", 'error')
            self.console.log(traceback.format_exc() + '\n', 'error')
            self.after(0, lambda: self.status.config(text="Failed", fg=COLORS['error']))
        finally:
            self.after(0, lambda: self.convert_btn.config(state='normal'))


# ============================================================================
# Tab: Test
# ============================================================================

class SyncZoomViewer:
    """
    Synchronized zoom/pan controller for two canvases displaying paired images.
    
    Controls:
      - Scroll wheel: zoom in/out (1× to 32×)
      - Click + drag: pan
      - Double-click: reset to fit view
    
    Both canvases always show the same region of the image at the same zoom level.
    """

    ZOOM_LEVELS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]

    def __init__(self, canvas_a, canvas_b, status_label=None):
        self.canvas_a = canvas_a
        self.canvas_b = canvas_b
        self.status_label = status_label

        self.img_a = None  # PIL Image (original)
        self.img_b = None  # PIL Image (result)
        self.photo_a = None  # keep reference to prevent GC
        self.photo_b = None

        # Viewport state (in image pixel coordinates)
        self._zoom_idx = None   # index into ZOOM_LEVELS, None = fit mode
        self._center_x = 0.0   # center of viewport in image coords
        self._center_y = 0.0
        self._drag_start = None

        # Bind events on both canvases
        for canvas in (canvas_a, canvas_b):
            canvas.bind('<MouseWheel>', self._on_scroll)
            canvas.bind('<Button-4>', self._on_scroll)   # Linux scroll up
            canvas.bind('<Button-5>', self._on_scroll)   # Linux scroll down
            canvas.bind('<ButtonPress-1>', self._on_drag_start)
            canvas.bind('<B1-Motion>', self._on_drag)
            canvas.bind('<ButtonRelease-1>', self._on_drag_end)
            canvas.bind('<Double-Button-1>', self._on_reset)
            canvas.bind('<Configure>', lambda e: self._redraw())

    def set_images(self, pil_a, pil_b):
        """Set new image pair and reset to fit view."""
        self.img_a = pil_a
        self.img_b = pil_b
        self._zoom_idx = None
        if pil_a:
            iw, ih = pil_a.size
            self._center_x = iw / 2
            self._center_y = ih / 2
        self._redraw()

    def _get_zoom(self):
        """Return current zoom factor. None = fit-to-canvas."""
        if self._zoom_idx is None:
            return None
        return self.ZOOM_LEVELS[self._zoom_idx]

    def _fit_zoom(self, canvas):
        """Calculate the zoom factor that fits the image to the canvas."""
        if not self.img_a:
            return 1.0
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 50)
        ch = max(canvas.winfo_height(), 50)
        iw, ih = self.img_a.size
        return min(cw / iw, ch / ih)

    def _on_scroll(self, event):
        if not self.img_a:
            return

        # Determine scroll direction
        if event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            direction = -1  # zoom out
        else:
            direction = 1   # zoom in

        # If currently in fit mode, find the nearest zoom level
        if self._zoom_idx is None:
            fit_z = self._fit_zoom(self.canvas_a)
            self._zoom_idx = 0
            for i, z in enumerate(self.ZOOM_LEVELS):
                if z <= fit_z:
                    self._zoom_idx = i
                else:
                    break

        # Find the image coordinate under the mouse cursor BEFORE zoom
        canvas = event.widget
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        old_zoom = self.ZOOM_LEVELS[self._zoom_idx]

        # Mouse position relative to canvas center
        mx = event.x - cw / 2
        my = event.y - ch / 2

        # Image coordinate under cursor
        img_x = self._center_x + mx / old_zoom
        img_y = self._center_y + my / old_zoom

        # Apply zoom step
        new_idx = self._zoom_idx + direction
        new_idx = max(0, min(len(self.ZOOM_LEVELS) - 1, new_idx))
        self._zoom_idx = new_idx
        new_zoom = self.ZOOM_LEVELS[new_idx]

        # Adjust center so the same image point stays under the cursor
        self._center_x = img_x - mx / new_zoom
        self._center_y = img_y - my / new_zoom

        self._clamp_center(new_zoom)
        self._redraw()

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._drag_start is None or not self.img_a:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)

        zoom = self._get_zoom()
        if zoom is None:
            zoom = self._fit_zoom(self.canvas_a)

        self._center_x -= dx / zoom
        self._center_y -= dy / zoom
        self._clamp_center(zoom)
        self._redraw()

    def _on_drag_end(self, event):
        self._drag_start = None

    def _on_reset(self, event):
        """Double-click resets to fit view."""
        self._zoom_idx = None
        if self.img_a:
            iw, ih = self.img_a.size
            self._center_x = iw / 2
            self._center_y = ih / 2
        self._redraw()

    def _clamp_center(self, zoom):
        """Keep the viewport from scrolling too far beyond the image edges."""
        if not self.img_a:
            return
        iw, ih = self.img_a.size
        # Allow panning up to half a canvas width beyond the edge
        self._center_x = max(0, min(iw, self._center_x))
        self._center_y = max(0, min(ih, self._center_y))

    def _render_canvas(self, canvas, pil_img, zoom, cx, cy):
        """Crop and scale the relevant region of pil_img and display on canvas."""
        from PIL import ImageTk, Image as PILImage
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 50)
        ch = max(canvas.winfo_height(), 50)

        if pil_img is None:
            canvas.delete('all')
            return None

        iw, ih = pil_img.size

        # How many image pixels are visible
        view_w = cw / zoom
        view_h = ch / zoom

        # Crop bounds in image space
        left = cx - view_w / 2
        top = cy - view_h / 2
        right = cx + view_w / 2
        bottom = cy + view_h / 2

        # Clamp to image bounds, calculate canvas offset for out-of-bounds
        crop_left = max(0, int(left))
        crop_top = max(0, int(top))
        crop_right = min(iw, int(right) + 1)
        crop_bottom = min(ih, int(bottom) + 1)

        if crop_right <= crop_left or crop_bottom <= crop_top:
            canvas.delete('all')
            return None

        cropped = pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))

        # Scale to display size
        display_w = max(1, int((crop_right - crop_left) * zoom))
        display_h = max(1, int((crop_bottom - crop_top) * zoom))

        # Use NEAREST for zoom > 2× (pixel-level inspection), LANCZOS otherwise
        resample = PILImage.Resampling.NEAREST if zoom >= 2.0 else PILImage.Resampling.LANCZOS
        scaled = cropped.resize((display_w, display_h), resample)

        photo = ImageTk.PhotoImage(scaled)

        # Position on canvas: account for image edge offset
        offset_x = int((crop_left - left) * zoom)
        offset_y = int((crop_top - top) * zoom)

        canvas.delete('all')
        canvas.create_image(offset_x, offset_y, image=photo, anchor='nw')

        return photo

    def _redraw(self):
        """Redraw both canvases with current viewport state."""
        if not self.img_a:
            return

        zoom = self._get_zoom()
        if zoom is None:
            # Fit mode — use LANCZOS fit for both
            zoom = self._fit_zoom(self.canvas_a)
            iw, ih = self.img_a.size
            cx, cy = iw / 2, ih / 2
        else:
            cx, cy = self._center_x, self._center_y

        self.photo_a = self._render_canvas(self.canvas_a, self.img_a, zoom, cx, cy)
        self.photo_b = self._render_canvas(self.canvas_b, self.img_b, zoom, cx, cy)

        if self.status_label:
            if self._zoom_idx is not None:
                z = self.ZOOM_LEVELS[self._zoom_idx]
                label = f"{z:.0f}×" if z >= 1 else f"{z:.2f}×"
                self.status_label.config(
                    text=f"🔍 {label}  (scroll to zoom, drag to pan, double-click to reset)")
            else:
                self.status_label.config(text="Fit to window  (scroll to zoom)")


class TestTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=COLORS['bg'])
        self.app = app
        self.original_pil = None
        self.result_image_pil = None
        self._cached_model = None
        self._cached_model_path = None

        controls = tk.Frame(self, bg=COLORS['bg'])
        controls.pack(fill='x', padx=20, pady=(20, 0))
        controls.columnconfigure(1, weight=1)
        StyledLabel(controls, text="🧪  Test Model on Image", heading=True).grid(
            row=0, column=0, columnspan=4, sticky='w', pady=(0, 8))

        settings = app.settings
        self.model_path = tk.StringVar(value=settings.get('test_model', 'nnaa.keras'))
        self.image_path = tk.StringVar(value=settings.get('test_image', ''))
        make_path_row(controls, "Keras Model:", self.model_path, 1, browse_type='open')
        make_path_row(controls, "Input Image:", self.image_path, 2, browse_type='open_image')

        btn_frame = tk.Frame(self, bg=COLORS['bg'])
        btn_frame.pack(fill='x', padx=20, pady=(12, 0))
        self.run_btn = StyledButton(btn_frame, text="▶  Run Inference", command=self.run_inference)
        self.run_btn.pack(side='left')
        self.save_btn = StyledButton(btn_frame, text="💾  Save Result", accent=False,
                                     command=self.save_result)
        self.save_btn.pack(side='left', padx=(8, 0))
        self.save_btn.config(state='disabled')
        self.zoom_status = StyledLabel(btn_frame, text="", dim=True)
        self.zoom_status.pack(side='right')
        self.status = StyledLabel(btn_frame, text="", dim=True)
        self.status.pack(side='left', padx=(12, 0))

        # Image display
        img_container = tk.Frame(self, bg=COLORS['bg'])
        img_container.pack(fill='both', expand=True, padx=20, pady=(12, 16))

        left = tk.Frame(img_container, bg=COLORS['bg_secondary'], highlightthickness=1,
                        highlightbackground=COLORS['border'])
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        StyledLabel(left, text="Original", heading=True, bg=COLORS['bg_secondary']).pack(
            anchor='w', padx=8, pady=(6, 2))
        self.canvas_orig = tk.Canvas(left, bg=COLORS['console_bg'], highlightthickness=0)
        self.canvas_orig.pack(fill='both', expand=True, padx=4, pady=(0, 4))

        right = tk.Frame(img_container, bg=COLORS['bg_secondary'], highlightthickness=1,
                         highlightbackground=COLORS['border'])
        right.pack(side='left', fill='both', expand=True, padx=(6, 0))
        StyledLabel(right, text="NNAA Result", heading=True, bg=COLORS['bg_secondary']).pack(
            anchor='w', padx=8, pady=(6, 2))
        self.canvas_result = tk.Canvas(right, bg=COLORS['console_bg'], highlightthickness=0)
        self.canvas_result.pack(fill='both', expand=True, padx=4, pady=(0, 4))

        # Synchronized zoom/pan viewer
        self.viewer = SyncZoomViewer(self.canvas_orig, self.canvas_result,
                                     status_label=self.zoom_status)

    def run_inference(self):
        img_path = self.image_path.get()
        if not img_path or not os.path.isfile(img_path):
            messagebox.showerror("Error", "Please select a valid input image.")
            return
        model_path = self.model_path.get()
        if not model_path or not os.path.isfile(model_path):
            messagebox.showerror("Error", "Please select a valid .keras model.")
            return

        self.app.settings.update({
            'test_model': model_path,
            'test_image': img_path,
        })
        save_settings(self.app.settings)

        self.run_btn.config(state='disabled')
        self.status.config(text="Running inference...", fg=COLORS['warning'])
        threading.Thread(target=self._inference_worker, args=(model_path, img_path), daemon=True).start()

    def _inference_worker(self, model_path, img_path):
        try:
            from PIL import Image
            import numpy as np
            import tensorflow as tf

            # Cache the model — only reload if path changed
            if self._cached_model_path != model_path:
                self._cached_model = tf.keras.models.load_model(model_path)
                self._cached_model_path = model_path
            model = self._cached_model

            img = Image.open(img_path).convert('RGB')  # Handle RGBA/grayscale safely
            r, g, b = [np.float32(ch) for ch in img.split()]

            y = r * 0.299 + g * 0.587 + b * 0.114
            cb = r * -0.1687 + g * -0.3313 + b * 0.5
            cr = r * 0.5 + g * -0.4187 + b * -0.0813

            h, w = img.size[1], img.size[0]
            tensor = y.reshape(1, h, w, 1) / 255.0
            prediction = model(tensor)
            tensor += prediction

            result_y = np.float32(tensor).reshape(h, w)
            result_y = (result_y * 255).round().clip(0, 255)

            r_out = np.uint8((result_y + 1.402 * cr).round().clip(0, 255))
            g_out = np.uint8((result_y - 0.34414 * cb - 0.71414 * cr).round().clip(0, 255))
            b_out = np.uint8((result_y + 1.772 * cb).round().clip(0, 255))

            result_img = Image.merge('RGB', [
                Image.fromarray(r_out),
                Image.fromarray(g_out),
                Image.fromarray(b_out)
            ])

            self.original_pil = img
            self.result_image_pil = result_img
            self.after(0, lambda: self._show_results(img, result_img))

        except Exception as e:
            self.after(0, lambda: self.status.config(text=f"Error: {e}", fg=COLORS['error']))
            self.after(0, lambda: messagebox.showerror("Inference Error", str(e)))
        finally:
            self.after(0, lambda: self.run_btn.config(state='normal'))

    def _show_results(self, orig_pil, result_pil):
        self.viewer.set_images(orig_pil, result_pil)
        self.save_btn.config(state='normal')
        self.status.config(text="Done!", fg=COLORS['success'])

    def save_result(self):
        if self.result_image_pil is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")])
        if path:
            self.result_image_pil.save(path)
            self.status.config(text=f"Saved: {os.path.basename(path)}", fg=COLORS['success'])


# ============================================================================
# Main App
# ============================================================================

class NNAAStudioApp:
    def __init__(self):
        self.settings = load_settings()
        self.root = tk.Tk()
        self.root.title("NNAA Shader Studio")
        self.root.geometry("900x750")
        self.root.minsize(750, 600)
        self.root.configure(bg=COLORS['bg'])

        # Dark title bar on Windows
        try:
            from ctypes import windll, c_int, sizeof, byref
            self.root.update()
            hwnd = windll.user32.GetParent(self.root.winfo_id())
            val = c_int(1)
            windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, byref(val), sizeof(val))
        except Exception:
            pass

        self._build_titlebar()
        self._build_tabs()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        """Save settings on exit."""
        save_settings(self.settings)
        self.root.destroy()

    def _build_titlebar(self):
        header = tk.Frame(self.root, bg=COLORS['bg'], height=56)
        header.pack(fill='x')
        header.pack_propagate(False)

        title_frame = tk.Frame(header, bg=COLORS['bg'])
        title_frame.pack(side='left', padx=20, pady=8)
        tk.Label(title_frame, text="⚡", font=(FONT_FAMILY, 22), bg=COLORS['bg'],
                 fg=COLORS['accent']).pack(side='left', padx=(0, 8))
        tk.Label(title_frame, text="NNAA Shader Studio", font=FONT_TITLE,
                 bg=COLORS['bg'], fg=COLORS['fg_bright']).pack(side='left')
        tk.Label(title_frame, text="Neural Network Anti-Aliasing", font=(FONT_FAMILY, 9),
                 bg=COLORS['bg'], fg=COLORS['fg_dim']).pack(side='left', padx=(12, 0), pady=(6, 0))

        tk.Frame(self.root, bg=COLORS['border'], height=1).pack(fill='x')

    def _build_tabs(self):
        # Tab bar
        tab_bar = tk.Frame(self.root, bg=COLORS['bg_secondary'], height=42)
        tab_bar.pack(fill='x')
        tab_bar.pack_propagate(False)

        self.tab_buttons = []
        self.tab_indicators = []
        self.tab_frames = []
        self.current_tab = 0

        tab_defs = [
            ("🏋  Train", TrainTab),
            ("🔄  Convert", ConvertTab),
            ("🧪  Test", TestTab),
        ]

        self.content = tk.Frame(self.root, bg=COLORS['bg'])

        for i, (label, tab_class) in enumerate(tab_defs):
            tab_container = tk.Frame(tab_bar, bg=COLORS['bg_secondary'])
            tab_container.pack(side='left')

            btn = tk.Label(tab_container, text=label, font=FONT_TAB,
                           bg=COLORS['bg_secondary'], fg=COLORS['tab_inactive'],
                           padx=20, pady=8, cursor='hand2')
            btn.pack()

            # Underline indicator
            indicator = tk.Frame(tab_container, bg=COLORS['bg_secondary'], height=3)
            indicator.pack(fill='x')

            btn.bind('<Button-1>', lambda e, idx=i: self._switch_tab(idx))
            btn.bind('<Enter>', lambda e, b=btn, idx=i: b.config(
                fg=COLORS['tab_active']) if idx != self.current_tab else None)
            btn.bind('<Leave>', lambda e, b=btn, idx=i: b.config(
                fg=COLORS['tab_inactive']) if idx != self.current_tab else None)

            self.tab_buttons.append(btn)
            self.tab_indicators.append(indicator)

            frame = tab_class(self.content, self)
            self.tab_frames.append(frame)

        tk.Frame(self.root, bg=COLORS['border'], height=1).pack(fill='x')
        self.content.pack(fill='both', expand=True)

        self._switch_tab(0)

    def _switch_tab(self, idx):
        self.current_tab = idx
        for i, (btn, indicator, frame) in enumerate(zip(
                self.tab_buttons, self.tab_indicators, self.tab_frames)):
            if i == idx:
                btn.config(fg=COLORS['tab_active'], bg=COLORS['bg'])
                btn.master.config(bg=COLORS['bg'])
                indicator.config(bg=COLORS['tab_active'])
                frame.pack(fill='both', expand=True)
            else:
                btn.config(fg=COLORS['tab_inactive'], bg=COLORS['bg_secondary'])
                btn.master.config(bg=COLORS['bg_secondary'])
                indicator.config(bg=COLORS['bg_secondary'])
                frame.pack_forget()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = NNAAStudioApp()
    app.run()
