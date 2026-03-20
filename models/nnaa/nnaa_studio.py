"""
NNAA Shader Studio - A unified GUI for training, converting, and testing
neural network anti-aliasing models for ReShade.

Requires: tensorflow, numpy, Pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import os
import sys
import traceback

# ============================================================================
# Theme colors
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
}

FONT_FAMILY = 'Segoe UI'
FONT_LABEL = (FONT_FAMILY, 10)
FONT_HEADING = (FONT_FAMILY, 12, 'bold')
FONT_TITLE = (FONT_FAMILY, 18, 'bold')
FONT_CONSOLE = ('Consolas', 9)
FONT_BUTTON = (FONT_FAMILY, 10, 'bold')
FONT_TAB = (FONT_FAMILY, 11, 'bold')

# ============================================================================
# Utility: styled widgets
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
            bg = COLORS['stop_bg']
            hover = COLORS['stop_hover']
        elif accent:
            bg = COLORS['button_bg']
            hover = COLORS['button_hover']
        else:
            bg = COLORS['accent_bg']
            hover = COLORS['border']
        
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
        
        self._bg = bg
        self._hover = hover
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


class ConsoleText(tk.Text):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault('bg', COLORS['console_bg'])
        kwargs.setdefault('fg', COLORS['console_fg'])
        kwargs.setdefault('font', FONT_CONSOLE)
        kwargs.setdefault('relief', 'flat')
        kwargs.setdefault('insertbackground', COLORS['console_fg'])
        kwargs.setdefault('selectbackground', COLORS['accent_bg'])
        kwargs.setdefault('highlightthickness', 1)
        kwargs.setdefault('highlightbackground', COLORS['border'])
        kwargs.setdefault('highlightcolor', COLORS['border'])
        kwargs.setdefault('state', 'disabled')
        kwargs.setdefault('wrap', 'word')
        super().__init__(parent, **kwargs)
        self.tag_configure('success', foreground=COLORS['success'])
        self.tag_configure('error', foreground=COLORS['error'])
        self.tag_configure('warning', foreground=COLORS['warning'])
        self.tag_configure('accent', foreground=COLORS['accent'])

    def append(self, text, tag=None):
        self.config(state='normal')
        if tag:
            self.insert('end', text, tag)
        else:
            self.insert('end', text)
        self.see('end')
        self.config(state='disabled')

    def clear(self):
        self.config(state='normal')
        self.delete('1.0', 'end')
        self.config(state='disabled')


def make_path_row(parent, label_text, var, row, browse_type='folder'):
    """Create a label + entry + browse button row."""
    StyledLabel(parent, text=label_text).grid(row=row, column=0, sticky='w', pady=(8, 2))
    entry = StyledEntry(parent, textvariable=var)
    entry.grid(row=row, column=1, sticky='ew', padx=(8, 4), pady=(8, 2))

    def browse():
        if browse_type == 'folder':
            path = filedialog.askdirectory()
        elif browse_type == 'open':
            path = filedialog.askopenfilename(filetypes=[
                ("Keras Models", "*.keras *.h5"),
                ("All Files", "*.*")
            ])
        elif browse_type == 'open_image':
            path = filedialog.askopenfilename(filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp"),
                ("All Files", "*.*")
            ])
        elif browse_type == 'save':
            path = filedialog.asksaveasfilename(
                defaultextension='.fx',
                filetypes=[("ReShade FX", "*.fx"), ("All Files", "*.*")]
            )
        else:
            path = None
        if path:
            var.set(path)

    StyledButton(parent, text="Browse", accent=False, command=browse).grid(
        row=row, column=2, padx=(0, 4), pady=(8, 2))
    return entry


def make_param_row(parent, label_text, var, row, col_offset=0):
    """Create a label + entry row for a hyperparameter."""
    StyledLabel(parent, text=label_text).grid(
        row=row, column=col_offset, sticky='w', pady=(4, 2), padx=(0, 4))
    entry = StyledEntry(parent, textvariable=var, width=14)
    entry.grid(row=row, column=col_offset + 1, sticky='w', pady=(4, 2), padx=(0, 16))
    return entry


# ============================================================================
# Tab: Train
# ============================================================================

class TrainTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=COLORS['bg'])
        self.app = app
        self.training_thread = None
        self.stop_event = threading.Event()
        self.log_queue = queue.Queue()

        # ── Dataset paths ──
        section = tk.Frame(self, bg=COLORS['bg'])
        section.pack(fill='x', padx=20, pady=(16, 0))
        StyledLabel(section, text="📁  Dataset Configuration", heading=True).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 4))
        section.columnconfigure(1, weight=1)

        self.train_bad = tk.StringVar(value='data/train/bad/1280x720')
        self.train_good = tk.StringVar(value='data/train/fixed/1280x720')
        self.test_bad = tk.StringVar(value='data/test/bad/2560x1440')
        self.test_good = tk.StringVar(value='data/test/fixed/2560x1440')

        make_path_row(section, "Train — No AA:", self.train_bad, 1)
        make_path_row(section, "Train — With AA:", self.train_good, 2)
        make_path_row(section, "Test — No AA:", self.test_bad, 3)
        make_path_row(section, "Test — With AA:", self.test_good, 4)

        # ── Hyperparameters ──
        params = tk.Frame(self, bg=COLORS['bg'])
        params.pack(fill='x', padx=20, pady=(16, 0))
        StyledLabel(params, text="⚙  Hyperparameters", heading=True).grid(
            row=0, column=0, columnspan=6, sticky='w', pady=(0, 4))

        self.lr = tk.StringVar(value='0.00001')
        self.batch_size = tk.StringVar(value='16')
        self.test_batch = tk.StringVar(value='4')
        self.epochs_per_run = tk.StringVar(value='5')

        make_param_row(params, "Learning Rate:", self.lr, 1, 0)
        make_param_row(params, "Train Batch:", self.batch_size, 1, 2)
        make_param_row(params, "Test Batch:", self.test_batch, 1, 4)
        make_param_row(params, "Epochs/Run:", self.epochs_per_run, 2, 0)

        # ── Model output ──
        model_frame = tk.Frame(self, bg=COLORS['bg'])
        model_frame.pack(fill='x', padx=20, pady=(16, 0))
        model_frame.columnconfigure(1, weight=1)
        StyledLabel(model_frame, text="💾  Model Output", heading=True).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 4))

        self.model_name = tk.StringVar(value='nnaa')
        self.model_dir = tk.StringVar(value='..')
        make_param_row(model_frame, "Model Name:", self.model_name, 1, 0)
        StyledLabel(model_frame, text="Output Dir:").grid(row=2, column=0, sticky='w', pady=(4, 2))
        StyledEntry(model_frame, textvariable=self.model_dir).grid(
            row=2, column=1, sticky='ew', padx=(8, 4), pady=(4, 2))
        StyledButton(model_frame, text="Browse", accent=False,
                     command=lambda: self.model_dir.set(
                         filedialog.askdirectory() or self.model_dir.get()
                     )).grid(row=2, column=2, padx=(0, 4), pady=(4, 2))

        # ── Buttons ──
        btn_frame = tk.Frame(self, bg=COLORS['bg'])
        btn_frame.pack(fill='x', padx=20, pady=(16, 0))

        self.start_btn = StyledButton(btn_frame, text="▶  Start Training", command=self.start_training)
        self.start_btn.pack(side='left', padx=(0, 8))
        self.stop_btn = StyledButton(btn_frame, text="■  Stop", danger=True, command=self.stop_training)
        self.stop_btn.pack(side='left')
        self.stop_btn.config(state='disabled')
        
        self.status_label = StyledLabel(btn_frame, text="", dim=True)
        self.status_label.pack(side='right')

        # ── Console ──
        console_frame = tk.Frame(self, bg=COLORS['bg'])
        console_frame.pack(fill='both', expand=True, padx=20, pady=(12, 16))
        StyledLabel(console_frame, text="📋  Training Log", heading=True).pack(anchor='w', pady=(0, 4))
        
        self.console = ConsoleText(console_frame, height=12)
        scrollbar = tk.Scrollbar(console_frame, command=self.console.yview,
                                 bg=COLORS['bg_secondary'], troughcolor=COLORS['console_bg'],
                                 highlightthickness=0, bd=0)
        self.console.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.console.pack(fill='both', expand=True)

        self.console.append("Welcome to NNAA Shader Studio!\n", 'accent')
        self.console.append("Configure your dataset paths and click Start Training.\n")

        self._poll_log()

    def _poll_log(self):
        """Pull messages from the log queue and display them."""
        while not self.log_queue.empty():
            msg, tag = self.log_queue.get_nowait()
            self.console.append(msg, tag)
        self.after(100, self._poll_log)

    def log(self, msg, tag=None):
        self.log_queue.put((msg, tag))

    def start_training(self):
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.stop_event.clear()
        self.status_label.config(text="Training...", fg=COLORS['success'])
        self.console.clear()
        self.log("Starting training...\n", 'accent')

        self.training_thread = threading.Thread(target=self._train_worker, daemon=True)
        self.training_thread.start()

    def stop_training(self):
        self.stop_event.set()
        self.log("\n⏹  Stop requested. Finishing current epoch...\n", 'warning')

    def _train_worker(self):
        try:
            self.log("Importing TensorFlow... ", None)
            import tensorflow as tf
            import numpy as np
            self.log("OK\n", 'success')

            # Import the dataset class from nnaa_train
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, script_dir)
            from nnaa_train import NnaaDataset

            lr = float(self.lr.get())
            batch_size = int(self.batch_size.get())
            test_batch = int(self.test_batch.get())
            epochs = int(self.epochs_per_run.get())
            model_name = self.model_name.get()
            models_path = self.model_dir.get()

            base_dir = self.train_bad.get()
            target_dir = self.train_good.get()
            test_base = self.test_bad.get()
            test_target = self.test_good.get()

            model_directory = os.path.join(models_path, model_name)
            model_path = os.path.join(model_directory, model_name) + ".keras"

            self.log(f"Model path: {model_path}\n")
            self.log(f"Learning rate: {lr}\n")
            self.log(f"Batch size: {batch_size} (train), {test_batch} (test)\n")
            self.log(f"Epochs per run: {epochs}\n\n")

            if not os.path.isdir(model_directory):
                os.makedirs(model_directory, exist_ok=True)
                self.log(f"Created directory: {model_directory}\n")

            loss_fn = tf.keras.losses.MeanSquaredError()

            if os.path.isfile(model_path):
                self.log("Loading existing model... ", None)
                model = tf.keras.models.load_model(model_path)
                self.log("OK\n", 'success')
            else:
                self.log("Creating new model... ", None)
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
                self.log("OK\n", 'success')

            if model.optimizer.learning_rate != lr:
                model.optimizer.learning_rate = lr
                self.log(f"Updated learning rate to {lr}\n", 'warning')

            # Model summary
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            self.log('\n'.join(summary_lines) + '\n\n', None)

            self.log("Loading dataset... ", None)
            train_dataset = NnaaDataset(base_dir, target_dir, batch_size, use_cache=True)
            test_dataset = NnaaDataset(test_base, test_target, test_batch, use_cache=True)
            self.log(f"OK ({len(train_dataset)} train batches, {len(test_dataset)} test batches)\n", 'success')

            best_error = float('inf')
            best_path = os.path.join(model_directory, "bestError.npy")
            if os.path.isfile(best_path):
                best_error = np.load(best_path).item()
            self.log(f"Best error so far: {best_error}\n\n")

            run = 0
            while not self.stop_event.is_set():
                run += 1
                self.log(f"━━━ Run {run} ━━━\n", 'accent')

                # Train
                history = model.fit(train_dataset, epochs=epochs, verbose=0)
                for epoch_idx, loss_val in enumerate(history.history['loss']):
                    self.log(f"  Epoch {epoch_idx + 1}/{epochs} — loss: {loss_val:.8f}\n")

                if self.stop_event.is_set():
                    break

                # Evaluate
                eval_result = model.evaluate(test_dataset, verbose=0)
                self.log(f"  Eval — loss: {eval_result[0]:.8f}, mse: {eval_result[1]:.8f}\n")

                if eval_result[0] < best_error:
                    best_error = eval_result[0]
                    np.save(best_path, best_error)
                    model.save(model_path)
                    self.log(f"  ★ New best! Saved model (error: {best_error:.8f})\n", 'success')
                else:
                    self.log(f"  No improvement (best: {best_error:.8f})\n", 'warning')
                self.log('\n')

            self.log("Training stopped.\n", 'accent')

        except Exception as e:
            self.log(f"\n✗ Error: {e}\n", 'error')
            self.log(traceback.format_exc() + '\n', 'error')
        finally:
            self.after(0, self._training_finished)

    def _training_finished(self):
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Idle", fg=COLORS['fg_dim'])


# ============================================================================
# Tab: Convert
# ============================================================================

class ConvertTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=COLORS['bg'])
        self.app = app

        # ── Input/Output ──
        section = tk.Frame(self, bg=COLORS['bg'])
        section.pack(fill='x', padx=20, pady=(20, 0))
        section.columnconfigure(1, weight=1)
        StyledLabel(section, text="🔄  Keras → ReShade FX Converter", heading=True).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))

        self.model_path = tk.StringVar(value='nnaa.keras')
        self.output_path = tk.StringVar(value='out_nnaa.fx')
        make_path_row(section, "Keras Model:", self.model_path, 1, browse_type='open')
        make_path_row(section, "Output .fx:", self.output_path, 2, browse_type='save')

        # ── Button ──
        btn_frame = tk.Frame(self, bg=COLORS['bg'])
        btn_frame.pack(fill='x', padx=20, pady=(16, 0))
        self.convert_btn = StyledButton(btn_frame, text="⚡  Convert to Shader",
                                        command=self.do_convert)
        self.convert_btn.pack(side='left')
        self.status = StyledLabel(btn_frame, text="", dim=True)
        self.status.pack(side='left', padx=(12, 0))

        # ── Console ──
        console_frame = tk.Frame(self, bg=COLORS['bg'])
        console_frame.pack(fill='both', expand=True, padx=20, pady=(12, 16))
        StyledLabel(console_frame, text="📋  Conversion Log", heading=True).pack(anchor='w', pady=(0, 4))
        self.console = ConsoleText(console_frame, height=16)
        scrollbar = tk.Scrollbar(console_frame, command=self.console.yview,
                                 bg=COLORS['bg_secondary'], troughcolor=COLORS['console_bg'],
                                 highlightthickness=0, bd=0)
        self.console.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.console.pack(fill='both', expand=True)

        self.console.append("Select a .keras model file and output path, then click Convert.\n")

    def do_convert(self):
        self.convert_btn.config(state='disabled')
        self.status.config(text="Converting...", fg=COLORS['warning'])
        self.console.clear()
        threading.Thread(target=self._convert_worker, daemon=True).start()

    def _convert_worker(self):
        try:
            self.console.append("Loading model...\n", 'accent')
            
            # Import the converter
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, script_dir)
            import convert

            model_path = self.model_path.get()
            output_path = self.output_path.get()

            if not os.path.isfile(model_path):
                self.console.append(f"✗ Model file not found: {model_path}\n", 'error')
                return

            layers = convert.load_model_weights(model_path)

            self.console.append("Extracted layers:\n")
            for i, layer in enumerate(layers):
                shapes = [w.shape for w in layer['weights']]
                self.console.append(f"  [{i}] {layer['name']} ({layer['type']}): {shapes}\n")

            conv0_kernel, conv0_bias = layers[0]['weights']
            prelu0_alpha = layers[1]['weights'][0]
            conv1_kernel, conv1_bias = layers[2]['weights']
            prelu1_alpha = layers[3]['weights'][0]
            conv2_kernel, conv2_bias = layers[4]['weights']
            prelu2_alpha = layers[5]['weights'][0]
            conv3_kernel, conv3_bias = layers[6]['weights']
            prelu3_alpha = layers[7]['weights'][0]
            final_kernel, final_bias = layers[8]['weights']

            self.console.append("\nGenerating shader...\n", 'accent')

            parts = []
            parts.append(convert.generate_header())

            self.console.append("  Layer conv2d (8x8)... ")
            parts.append(convert.generate_first_conv_layer(conv0_kernel, conv0_bias, prelu0_alpha))
            self.console.append("OK\n", 'success')

            self.console.append("  Layer conv2d_1 (3x3)... ")
            parts.append('\n')
            parts.append(convert.generate_mid_conv_layer(1, conv1_kernel, conv1_bias, prelu1_alpha))
            self.console.append("OK\n", 'success')

            self.console.append("  Layer conv2d_2 (3x3)... ")
            parts.append('\n')
            parts.append(convert.generate_mid_conv_layer(2, conv2_kernel, conv2_bias, prelu2_alpha))
            self.console.append("OK\n", 'success')

            self.console.append("  Layer conv2d_3 + final (fused)... ")
            parts.append('\n')
            parts.append(convert.generate_mid_conv_layer(3, conv3_kernel, conv3_bias, prelu3_alpha,
                                                         is_last_hidden=True,
                                                         final_kernel=final_kernel,
                                                         final_bias=final_bias))
            self.console.append("OK\n", 'success')

            parts.append(convert.generate_technique())

            shader_code = '\n'.join(parts)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(shader_code)

            lines = shader_code.count('\n')
            size = len(shader_code)
            self.console.append(f"\n✓ Shader saved to: {output_path}\n", 'success')
            self.console.append(f"  {size:,} bytes, {lines:,} lines\n", 'success')
            self.after(0, lambda: self.status.config(text="Done!", fg=COLORS['success']))

        except Exception as e:
            self.console.append(f"\n✗ Error: {e}\n", 'error')
            self.console.append(traceback.format_exc() + '\n', 'error')
            self.after(0, lambda: self.status.config(text="Failed", fg=COLORS['error']))
        finally:
            self.after(0, lambda: self.convert_btn.config(state='normal'))


# ============================================================================
# Tab: Test
# ============================================================================

class TestTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=COLORS['bg'])
        self.app = app
        self.original_photo = None
        self.result_photo = None
        self.result_image_pil = None

        # ── Controls ──
        controls = tk.Frame(self, bg=COLORS['bg'])
        controls.pack(fill='x', padx=20, pady=(20, 0))
        controls.columnconfigure(1, weight=1)
        StyledLabel(controls, text="🧪  Test Model on Image", heading=True).grid(
            row=0, column=0, columnspan=4, sticky='w', pady=(0, 8))

        self.model_path = tk.StringVar(value='nnaa.keras')
        self.image_path = tk.StringVar()
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
        self.status = StyledLabel(btn_frame, text="", dim=True)
        self.status.pack(side='left', padx=(12, 0))

        # ── Image display ──
        img_container = tk.Frame(self, bg=COLORS['bg'])
        img_container.pack(fill='both', expand=True, padx=20, pady=(12, 16))

        # Before
        left = tk.Frame(img_container, bg=COLORS['bg_secondary'], highlightthickness=1,
                        highlightbackground=COLORS['border'])
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        StyledLabel(left, text="Original", heading=True, bg=COLORS['bg_secondary']).pack(
            anchor='w', padx=8, pady=(6, 2))
        self.canvas_orig = tk.Canvas(left, bg=COLORS['console_bg'], highlightthickness=0)
        self.canvas_orig.pack(fill='both', expand=True, padx=4, pady=(0, 4))

        # After
        right = tk.Frame(img_container, bg=COLORS['bg_secondary'], highlightthickness=1,
                         highlightbackground=COLORS['border'])
        right.pack(side='left', fill='both', expand=True, padx=(6, 0))
        StyledLabel(right, text="NNAA Result", heading=True, bg=COLORS['bg_secondary']).pack(
            anchor='w', padx=8, pady=(6, 2))
        self.canvas_result = tk.Canvas(right, bg=COLORS['console_bg'], highlightthickness=0)
        self.canvas_result.pack(fill='both', expand=True, padx=4, pady=(0, 4))

    def _fit_image(self, pil_img, canvas):
        """Resize image to fit canvas and display it."""
        from PIL import ImageTk
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 100)
        ch = max(canvas.winfo_height(), 100)
        iw, ih = pil_img.size
        ratio = min(cw / iw, ch / ih)
        new_w = max(1, int(iw * ratio))
        new_h = max(1, int(ih * ratio))
        resized = pil_img.resize((new_w, new_h))
        photo = ImageTk.PhotoImage(resized)
        canvas.delete('all')
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor='center')
        return photo  # must keep reference

    def run_inference(self):
        img_path = self.image_path.get()
        if not img_path or not os.path.isfile(img_path):
            messagebox.showerror("Error", "Please select a valid input image.")
            return
        model_path = self.model_path.get()
        if not model_path or not os.path.isfile(model_path):
            messagebox.showerror("Error", "Please select a valid .keras model.")
            return

        self.run_btn.config(state='disabled')
        self.status.config(text="Running inference...", fg=COLORS['warning'])
        threading.Thread(target=self._inference_worker, args=(model_path, img_path), daemon=True).start()

    def _inference_worker(self, model_path, img_path):
        try:
            from PIL import Image
            import numpy as np
            import tensorflow as tf

            model = tf.keras.models.load_model(model_path)

            img = Image.open(img_path)
            channels = img.split()
            r = np.float32(channels[0])
            g = np.float32(channels[1])
            b = np.float32(channels[2])

            y = r * 0.299 + g * 0.587 + b * 0.114
            cb = r * -0.1687 + g * -0.3313 + b * 0.5
            cr = r * 0.5 + g * -0.4187 + b * -0.0813

            tensor = y.reshape(1, channels[0].size[1], channels[0].size[0], 1) / 255.0
            prediction = model(tensor)
            tensor += prediction

            result_y = np.float32(tensor).reshape(channels[0].size[1], channels[0].size[0])
            result_y = (result_y * 255).round().clip(0, 255)

            r_out = np.uint8((result_y + 1.402 * cr).round().clip(0, 255))
            g_out = np.uint8((result_y - 0.34414 * cb - 0.71414 * cr).round().clip(0, 255))
            b_out = np.uint8((result_y + 1.772 * cb).round().clip(0, 255))

            result_img = Image.merge('RGB', [
                Image.fromarray(r_out),
                Image.fromarray(g_out),
                Image.fromarray(b_out)
            ])

            self.result_image_pil = result_img

            # Display on main thread
            self.after(0, lambda: self._show_results(img, result_img))

        except Exception as e:
            self.after(0, lambda: self.status.config(text=f"Error: {e}", fg=COLORS['error']))
            self.after(0, lambda: messagebox.showerror("Inference Error", str(e)))
        finally:
            self.after(0, lambda: self.run_btn.config(state='normal'))

    def _show_results(self, orig_pil, result_pil):
        self.original_photo = self._fit_image(orig_pil, self.canvas_orig)
        self.result_photo = self._fit_image(result_pil, self.canvas_result)
        self.save_btn.config(state='normal')
        self.status.config(text="Done!", fg=COLORS['success'])

    def save_result(self):
        if self.result_image_pil is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")]
        )
        if path:
            self.result_image_pil.save(path)
            self.status.config(text=f"Saved: {os.path.basename(path)}", fg=COLORS['success'])


# ============================================================================
# Main App
# ============================================================================

class NNAAStudioApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("NNAA Shader Studio")
        self.root.geometry("900x720")
        self.root.minsize(750, 550)
        self.root.configure(bg=COLORS['bg'])

        # Try to set dark title bar on Windows
        try:
            from ctypes import windll
            self.root.update()
            hwnd = windll.user32.GetParent(self.root.winfo_id())
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                                                 byref := (c_int := __import__('ctypes').c_int)(1),
                                                 __import__('ctypes').sizeof(c_int))
        except Exception:
            pass

        self._build_titlebar()
        self._build_tabs()

    def _build_titlebar(self):
        """Build the custom title/header area."""
        header = tk.Frame(self.root, bg=COLORS['bg'], height=56)
        header.pack(fill='x', padx=0, pady=0)
        header.pack_propagate(False)

        # App title
        title_frame = tk.Frame(header, bg=COLORS['bg'])
        title_frame.pack(side='left', padx=20, pady=8)
        tk.Label(title_frame, text="⚡", font=(FONT_FAMILY, 22), bg=COLORS['bg'],
                 fg=COLORS['accent']).pack(side='left', padx=(0, 8))
        tk.Label(title_frame, text="NNAA Shader Studio", font=FONT_TITLE,
                 bg=COLORS['bg'], fg=COLORS['fg_bright']).pack(side='left')

        # Subtitle
        tk.Label(title_frame, text="Neural Network Anti-Aliasing", font=(FONT_FAMILY, 9),
                 bg=COLORS['bg'], fg=COLORS['fg_dim']).pack(side='left', padx=(12, 0), pady=(6, 0))

        # Separator
        sep = tk.Frame(self.root, bg=COLORS['border'], height=1)
        sep.pack(fill='x')

    def _build_tabs(self):
        """Build the tab navigation and content area."""
        # Custom tab bar
        tab_bar = tk.Frame(self.root, bg=COLORS['bg_secondary'], height=42)
        tab_bar.pack(fill='x')
        tab_bar.pack_propagate(False)

        self.tab_buttons = []
        self.tab_frames = []
        self.current_tab = 0

        tab_defs = [
            ("🏋  Train", TrainTab),
            ("🔄  Convert", ConvertTab),
            ("🧪  Test", TestTab),
        ]

        # Content container
        self.content = tk.Frame(self.root, bg=COLORS['bg'])
        self.content.pack(fill='both', expand=True)

        for i, (label, tab_class) in enumerate(tab_defs):
            btn = tk.Label(tab_bar, text=label, font=FONT_TAB,
                           bg=COLORS['bg_secondary'], fg=COLORS['tab_inactive'],
                           padx=20, pady=8, cursor='hand2')
            btn.pack(side='left')
            btn.bind('<Button-1>', lambda e, idx=i: self._switch_tab(idx))
            btn.bind('<Enter>', lambda e, b=btn: b.config(
                fg=COLORS['tab_active'] if b != self.tab_buttons[self.current_tab]
                else COLORS['tab_active']))
            btn.bind('<Leave>', lambda e, b=btn, idx=i: b.config(
                fg=COLORS['tab_active'] if idx == self.current_tab
                else COLORS['tab_inactive']))
            self.tab_buttons.append(btn)

            frame = tab_class(self.content, self)
            self.tab_frames.append(frame)

        # Tab indicator line
        sep2 = tk.Frame(self.root, bg=COLORS['border'], height=1)
        sep2.pack(fill='x')
        # Re-pack content after separator
        self.content.pack_forget()
        self.content.pack(fill='both', expand=True)

        self._switch_tab(0)

    def _switch_tab(self, idx):
        self.current_tab = idx
        for i, (btn, frame) in enumerate(zip(self.tab_buttons, self.tab_frames)):
            if i == idx:
                btn.config(fg=COLORS['tab_active'], bg=COLORS['bg'])
                frame.pack(fill='both', expand=True)
            else:
                btn.config(fg=COLORS['tab_inactive'], bg=COLORS['bg_secondary'])
                frame.pack_forget()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = NNAAStudioApp()
    app.run()
