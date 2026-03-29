import os
from os.path import isfile, isdir, join
from multiprocessing import freeze_support
import concurrent.futures
from PIL import Image
import random
import tensorflow as tf
import numpy as np


def extract_luma(img_path):
    """Extract luma (Y) channel from an image, normalized to [0, 1]."""
    img = Image.open(img_path).convert('RGB')
    r, g, b = [np.float32(ch) for ch in img.split()]
    y = (r * 0.299 + g * 0.587 + b * 0.114) / 255.0
    img.close()
    return y.reshape(y.shape[0], y.shape[1], 1)


def _load_pair(args):
    """Load a (base, target) luma pair. Used for parallel loading."""
    base_path, target_path = args
    x = extract_luma(base_path)
    y = extract_luma(target_path)
    return x, y - x  # input, residual target


class NnaaDataset(tf.keras.utils.PyDataset):
    """
    Dataset that loads paired aliased/clean images for NNAA training.
    
    Improvements over the original:
      - Intersection-based file matching (not union)
      - Random patch cropping for memory efficiency
      - Data augmentation (horizontal + vertical flips)
      - Reshuffling each epoch
      - Parallel image loading
      - float32 precision (not float16)
    """

    def __init__(self, bases_dir, targets_dir, batch_size, 
                 use_cache=False, patch_size=0, augment=False, **kwargs):
        """
        Args:
            bases_dir: Directory of aliased (no AA) images
            targets_dir: Directory of clean (AA) images
            batch_size: Number of samples per batch
            use_cache: If True, cache all data in RAM after first load
            patch_size: If > 0, extract random patches of this size.
                        Must be divisible by 2 (for stride-2 layers).
                        Recommended: 128 for training, 0 for evaluation.
            augment: If True, apply random flips (H + V) for 4x data.
        """
        super().__init__(**kwargs)
        
        # Find matching filenames using intersection (not union)
        base_files = set(f for f in os.listdir(bases_dir) if isfile(join(bases_dir, f)))
        target_files = set(f for f in os.listdir(targets_dir) if isfile(join(targets_dir, f)))
        self.img_names = sorted(base_files & target_files)
        
        if not self.img_names:
            raise ValueError(
                f"No matching files between:\n  {bases_dir}\n  {targets_dir}\n"
                f"Found {len(base_files)} base files, {len(target_files)} target files."
            )

        random.shuffle(self.img_names)

        self.bases_dir = bases_dir
        self.targets_dir = targets_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.augment = augment
        self._epoch = 0

        self.cache_built = False
        self._cached_images = None  # Will store list of (input, target) pairs

        if use_cache:
            self._build_cache()

    def _build_cache(self):
        """Load all images into RAM using parallel I/O."""
        pairs = [(join(self.bases_dir, n), join(self.targets_dir, n)) 
                 for n in self.img_names]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            self._cached_images = list(executor.map(_load_pair, pairs))
        
        self.cache_built = True

    def __len__(self):
        return len(self.img_names) // self.batch_size

    def on_epoch_end(self):
        """Reshuffle data after each epoch for better generalization."""
        self._epoch += 1
        if self.cache_built:
            random.shuffle(self._cached_images)
        else:
            random.shuffle(self.img_names)

    def _random_patch(self, x, y):
        """Extract a random patch from input and target (same location)."""
        ps = self.patch_size
        h, w = x.shape[0], x.shape[1]
        
        if h <= ps or w <= ps:
            return x, y
        
        # Ensure patch coordinates are even (divisible by 2) for stride-2 layers
        top = random.randrange(0, h - ps, 2)
        left = random.randrange(0, w - ps, 2)
        
        return x[top:top+ps, left:left+ps, :], y[top:top+ps, left:left+ps, :]

    def _augment(self, x, y):
        """Apply random horizontal and vertical flips."""
        if random.random() > 0.5:
            x = np.flip(x, axis=1)  # horizontal flip
            y = np.flip(y, axis=1)
        if random.random() > 0.5:
            x = np.flip(x, axis=0)  # vertical flip
            y = np.flip(y, axis=0)
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    def __getitem__(self, idx):
        inputs = []
        targets = []
        r_idx = idx * self.batch_size

        for i in range(r_idx, r_idx + self.batch_size):
            if self.cache_built:
                x, y = self._cached_images[i]
            else:
                img_name = self.img_names[i]
                x, y = _load_pair((
                    join(self.bases_dir, img_name),
                    join(self.targets_dir, img_name)
                ))

            # Random patch cropping
            if self.patch_size > 0:
                x, y = self._random_patch(x, y)

            # Data augmentation
            if self.augment:
                x, y = self._augment(x, y)

            inputs.append(x)
            targets.append(y)

        return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


# ============================================================================
# CLI training script
# ============================================================================

if __name__ == "__main__":
    freeze_support()

    # ── Configuration ──
    base_dir_path = "data/train/bad/1280x720"
    target_dir_path = "data/train/fixed/1280x720"
    base_dir_path_test = "data/test/bad/2560x1440"
    target_dir_path_test = "data/test/fixed/2560x1440"

    model_name = "nnaa"
    models_path = ".."
    lr = 0.00001
    patience = 20          # Stop after N runs with no improvement
    patch_size = 128       # Random crop size (0 = full image)
    epochs_per_run = 5

    # ── Model ──
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

    model_directory = join(models_path, model_name)
    os.makedirs(model_directory, exist_ok=True)
    model_path = join(model_directory, model_name) + ".keras"

    loss_fn = tf.keras.losses.MeanSquaredError()

    if isfile(model_path):
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Creating new model...")
        model = tf.keras.Model(input_layer, output, name=model_name)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=loss_fn, metrics=['mean_squared_error'])

    model.summary()

    if model.optimizer.learning_rate != lr:
        model.optimizer.learning_rate = lr
        print("Learning rate updated")

    print(f"Learning rate: {lr}")
    print(f"Patch size: {patch_size if patch_size > 0 else 'full image'}")
    print(f"Patience: {patience} runs\n")

    # ── Dataset ──
    train_dataset = NnaaDataset(base_dir_path, target_dir_path, 16,
                                use_cache=True, patch_size=patch_size, augment=True)
    test_dataset = NnaaDataset(base_dir_path_test, target_dir_path_test, 4,
                               use_cache=True, patch_size=0, augment=False)

    print(f"Training: {len(train_dataset)} batches ({len(train_dataset.img_names)} images)")
    print(f"Testing:  {len(test_dataset)} batches ({len(test_dataset.img_names)} images)\n")

    # ── LR scheduler: reduce LR when loss plateaus ──
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

    # ── Training loop ──
    best_error = float('inf')
    best_path = join(model_directory, "bestError.npy")
    if isfile(best_path):
        best_error = np.load(best_path).item()
    print(f"Best error: {best_error}\n")

    no_improve_count = 0
    run = 0

    while True:
        run += 1
        print(f"\n━━━ Run {run} ━━━")

        model.fit(train_dataset, epochs=epochs_per_run, callbacks=[lr_scheduler])

        eval_result = model.evaluate(test_dataset, verbose=2)
        current_lr = float(model.optimizer.learning_rate)
        print(f"  LR: {current_lr:.2e}")

        if eval_result[0] < best_error:
            best_error = eval_result[0]
            np.save(best_path, best_error)
            model.save(model_path)
            print(f"  ★ New best! Saved (error: {best_error:.8f})")
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"  No improvement ({no_improve_count}/{patience}, best: {best_error:.8f})")

        if patience > 0 and no_improve_count >= patience:
            print(f"\nEarly stopping: no improvement for {patience} runs.")
            break
