#!/usr/bin/env python
"""
train_cyclegan_target12.py
==========================

Train a 3D CycleGAN target-site harmonization baseline (``cyclegan_target12``)
on OpenBHB CAT12 VBM gray-matter maps.

    Domain A = train subjects with original_site != 12.0   (non-target sites)
    Domain B = train subjects with original_site == 12.0   (reference site)

The generator/discriminator architectures and the LSGAN + cycle-consistency
losses are taken unchanged from ``3d_cyclegan_mri_harmonization`` (imported, not
copied).  We add an optional identity loss (both domains are CAT12 VBM, so
identity mapping is meaningful and helps preserve anatomy).

The main artifact is ``generator_AtoB.h5`` -- applied at inference to OOD images
to map them toward the reference-site style.

This script REQUIRES a GPU and is meant to run on the user's server.  It never
runs on the laptop.  See ``README_openbhb.md`` for exact commands.
"""

import argparse
import json
import os
import sys
from time import time

import numpy as np

# Local bridge (same directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openbhb_bridge as ob  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data / paths
    p.add_argument("--dataloader-dir", required=True,
                   help="Directory containing the user's DataLoader.py (OpenBHBDataset).")
    p.add_argument("--openbhb-root", default=None,
                   help="OpenBHB data root (contains images/metadata.tsv). "
                        "Default: OpenBHBDataset's built-in default.")
    p.add_argument("--repo-harmo-dir", required=True,
                   help="Path to 3d_cyclegan_mri_harmonization/harmonization "
                        "(for model_architectures.py).")
    p.add_argument("--dest-dir", required=True,
                   help="Output dir for checkpoints, norm_stats.json and stats.json.")
    # Optimization schedule (repo defaults)
    p.add_argument("--epochs-base", type=int, default=150,
                   help="Epochs at constant LR (repo default 150).")
    p.add_argument("--epochs-decay", type=int, default=150,
                   help="Epochs of linear LR decay to 0 (repo default 150).")
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--init-lr", type=float, default=2e-4)
    p.add_argument("--lambda-cyc-init", type=float, default=200.0)
    p.add_argument("--lambda-cyc-end", type=float, default=100.0)
    p.add_argument("--identity-frac", type=float, default=0.5,
                   help="Identity loss weight = frac * current cycle lambda. "
                        "Set 0 to disable identity loss.")
    p.add_argument("--disc-n-batchs", type=int, default=2)
    p.add_argument("--disc-buf-size", type=int, default=50)
    p.add_argument("--n-steps-disc", type=int, default=1)
    # Data handling
    p.add_argument("--no-augment", action="store_true",
                   help="Disable random-shift augmentation.")
    p.add_argument("--shift-max", type=int, default=5)
    p.add_argument("--norm-scale", type=float, default=None,
                   help="Override the VBM normalization scale S (else computed "
                        "as the median of nonzero voxels over Domain B).")
    p.add_argument("--scale-sample", type=int, default=60,
                   help="How many Domain-B subjects to sample when computing S.")
    # Environment
    p.add_argument("--gpu-mem-limit", type=int, default=None,
                   help="Optional per-GPU logical memory cap in MB.")
    p.add_argument("--no-mixed-precision", action="store_true")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Allow running without a GPU (debug only; very slow).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def configure_tf(args):
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus and not args.allow_cpu:
        raise SystemExit(
            "No GPU detected. Training a 3D CycleGAN on CPU is not practical. "
            "Run on the server, or pass --allow-cpu for a tiny debug run.")
    if gpus and args.gpu_mem_limit:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_mem_limit)])
    if gpus and not args.no_mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[tf] mixed_float16 enabled")
    else:
        print("[tf] float32 (mixed precision off)")
    tf.random.set_seed(args.seed)
    return tf


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.dest_dir, exist_ok=True)

    tf = configure_tf(args)
    from tensorflow import (where as tf_where, concat as tf_concat, Variable as tf_Variable,
                            GradientTape, function as tf_function, TensorSpec,
                            tensor_scatter_nd_update, expand_dims as tf_expand_dims,
                            range as tf_range)
    from tensorflow.random import shuffle as tf_shuffle
    from tensorflow.math import reduce_mean, abs as tf_abs, square
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import mixed_precision

    # --- import repo architectures (unchanged) --------------------------------
    sys.path.insert(0, os.path.abspath(args.repo_harmo_dir))
    from model_architectures import Generator, Discriminator  # noqa: E402

    # --- build dataset + domains ---------------------------------------------
    print("\n=== Building OpenBHBDataset and domains ===")
    ds = ob.load_openbhb_dataset(args.dataloader_dir, args.openbhb_root)
    A_idx, B_idx = ob.get_domain_indices(ds, ob.TARGET_SITE)
    if len(A_idx) == 0 or len(B_idx) == 0:
        raise SystemExit(
            f"Empty domain: |A|={len(A_idx)}, |B|={len(B_idx)}. "
            f"Site {ob.TARGET_SITE} must be present in the TRAIN split.")

    # --- normalization scale --------------------------------------------------
    if args.norm_scale is not None:
        scale = float(args.norm_scale)
        print(f"[norm] using provided S = {scale:.6f}")
    else:
        scale = ob.compute_norm_scale(ds, B_idx, max_subjects=args.scale_sample)

    # --- REQUIRED PRINTS ------------------------------------------------------
    native_shape = ob.load_raw_volume(ds, int(A_idx[0])).shape
    print("\n=== Training configuration ===")
    print(f"  number of A_train subjects (original_site != 12): {len(A_idx)}")
    print(f"  number of B_train subjects (original_site == 12): {len(B_idx)}")
    print(f"  native image shape:                               {native_shape}")
    print(f"  network image shape (padded):                     {ob.NET_SHAPE} + (1,)")
    ob.report_intensity_range(ds, B_idx, scale)
    ob.report_intensity_range(ds, A_idx, scale)
    print(f"  steps (batches) per epoch:                        {args.steps_per_epoch}")
    print(f"  batch size:                                       1")
    total_epochs = args.epochs_base + args.epochs_decay
    print(f"  total epochs:                                     {total_epochs}")
    print(f"  identity loss:                                    "
          f"{'OFF' if args.identity_frac == 0 else f'ON (frac={args.identity_frac})'}")

    # Persist the normalization so export uses an identical transform.
    ob.save_norm_stats(os.path.join(args.dest_dir, "norm_stats.json"), scale,
                       extra={"n_A_train": int(len(A_idx)),
                              "n_B_train": int(len(B_idx)),
                              "native_shape": list(native_shape)})

    # --- input pipelines ------------------------------------------------------
    augment = not args.no_augment
    dsA = ob.make_tf_dataset(ds, A_idx, scale, augment=augment,
                             shift_max=args.shift_max, seed=args.seed)
    dsB = ob.make_tf_dataset(ds, B_idx, scale, augment=augment,
                             shift_max=args.shift_max, seed=args.seed + 1)
    iterA = iter(dsA)
    iterB = iter(dsB)

    IMAGE_SHAPE = list(ob.NET_SHAPE) + [1]
    DISC_N_BATCHS = args.disc_n_batchs
    DISC_BUF_SIZE = args.disc_buf_size
    USE_ID = args.identity_frac > 0
    ID_FRAC = float(args.identity_frac)

    # --- models ---------------------------------------------------------------
    print("\n=== Instantiating models ===")
    generator_AtoB = Generator()
    generator_BtoA = Generator()
    discriminator_A = Discriminator()
    discriminator_B = Discriminator()

    # --- LR schedule + optimizers (repo logic) --------------------------------
    # Repo behaviour: constant LR for `epochs_base` epochs, then linear decay to 0
    # over `epochs_decay` epochs.  We set the LR each epoch via the optimizer's
    # public setter (on the *inner* Adam, before any LossScaleOptimizer wrapper),
    # rather than reading a Python callable inside a @tf.function -- the latter
    # would bake the LR at trace time and never decay.
    DECAY_STEP = 0 if args.epochs_decay == 0 else args.init_lr / args.epochs_decay

    def lr_for_epoch(e):
        s = max(e - args.epochs_base, 0)
        return float(args.init_lr - s * DECAY_STEP)

    use_lso = not args.no_mixed_precision and bool(tf.config.list_physical_devices("GPU"))
    inner_opts = []  # underlying Adam optimizers, used only to update the LR

    def make_opt():
        inner = Adam(learning_rate=float(args.init_lr), beta_1=0.5, beta_2=0.999)
        inner_opts.append(inner)
        return mixed_precision.LossScaleOptimizer(inner, dynamic=True) if use_lso else inner

    generator_AtoB_optimizer = make_opt()
    generator_BtoA_optimizer = make_opt()
    discriminator_A_optimizer = make_opt()
    discriminator_B_optimizer = make_opt()
    scaled = use_lso

    def set_lr(value):
        for o in inner_opts:
            o.learning_rate = value

    def _scale_loss(opt, loss):
        return opt.get_scaled_loss(loss) if scaled else loss

    def _unscale_grads(opt, grads):
        return opt.get_unscaled_gradients(grads) if scaled else grads

    # --- discriminator trainer (repo logic) -----------------------------------
    class DiscriminatorTrainer:
        def __init__(self, discriminator, generator, optimizer):
            self.discriminator = discriminator
            self.generator = generator
            self.optimizer = optimizer

        def translate_images(self, images):
            transformed = self.generator(images, training=False)
            return tf_where(images > -1, transformed, -1)

        def init_buffer(self, images_list):
            images_list = [self.translate_images(im) for im in images_list]
            self.buffer = tf_Variable(tf_concat(images_list, 0), dtype="float32",
                                      trainable=False)

        @tf_function(input_signature=(TensorSpec(shape=[DISC_N_BATCHS * 2] + IMAGE_SHAPE, dtype="float32"),
                                      TensorSpec(shape=[DISC_N_BATCHS] + IMAGE_SHAPE, dtype="float32")),
                     jit_compile=True)
        def train(self, images1, images2):
            indices = tf_shuffle(tf_range(DISC_BUF_SIZE))[:DISC_N_BATCHS]
            fakes1 = self.buffer.sparse_read(indices)
            fakes2 = self.translate_images(images2)
            fake_images = tf_concat([fakes1, fakes2], axis=0)

            replace_indices = tf_shuffle(tf_range(DISC_BUF_SIZE))[:DISC_N_BATCHS]
            self.buffer.assign(tensor_scatter_nd_update(
                self.buffer, indices=tf_expand_dims(replace_indices, axis=1), updates=fakes2))

            with GradientTape() as tape:
                disc_real = self.discriminator(images1, training=True)
                disc_fakes = self.discriminator(fake_images, training=True)
                disc_loss = reduce_mean(square(disc_real - 1)) + reduce_mean(square(disc_fakes))
                disc_loss_scaled = _scale_loss(self.optimizer, disc_loss)
            grads = tape.gradient(disc_loss_scaled, self.discriminator.trainable_variables)
            grads = _unscale_grads(self.optimizer, grads)
            self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
            return disc_loss

    discA_trainer = DiscriminatorTrainer(discriminator_A, generator_BtoA, discriminator_A_optimizer)
    discB_trainer = DiscriminatorTrainer(discriminator_B, generator_AtoB, discriminator_B_optimizer)

    # --- generator training step (repo LSGAN + cycle, + optional identity) ----
    @tf_function(input_signature=(TensorSpec(shape=[1] + IMAGE_SHAPE, dtype="float32"),
                                  TensorSpec(shape=[1] + IMAGE_SHAPE, dtype="float32"),
                                  TensorSpec(shape=(), dtype="float32")),
                 jit_compile=True)
    def train_generators(imagesA, imagesB, lambda_cyc):
        maskA = imagesA > -1
        maskB = imagesB > -1
        with GradientTape(persistent=True) as tape:
            fakesA = tf_where(maskB, generator_BtoA(imagesB, training=True), -1)
            fakesB = tf_where(maskA, generator_AtoB(imagesA, training=True), -1)

            # adversarial (LSGAN)
            disc_fakesA = discriminator_A(fakesA, training=False)
            disc_fakesB = discriminator_B(fakesB, training=False)
            gen_AtoB_adv_loss = reduce_mean(square(disc_fakesB - 1))
            gen_BtoA_adv_loss = reduce_mean(square(disc_fakesA - 1))

            # cycle consistency
            cycledA = tf_where(maskA, generator_BtoA(fakesB, training=True), -1)
            cycledB = tf_where(maskB, generator_AtoB(fakesA, training=True), -1)
            cycle_loss_aba = reduce_mean(tf_abs(imagesA - cycledA))
            cycle_loss_bab = reduce_mean(tf_abs(imagesB - cycledB))
            sum_cycle_loss = cycle_loss_aba + cycle_loss_bab

            gen_AtoB_loss = gen_AtoB_adv_loss + lambda_cyc * sum_cycle_loss
            gen_BtoA_loss = gen_BtoA_adv_loss + lambda_cyc * sum_cycle_loss

            # identity: G_AtoB(B) ~ B ; G_BtoA(A) ~ A
            if USE_ID:
                lambda_id = ID_FRAC * lambda_cyc
                idB = tf_where(maskB, generator_AtoB(imagesB, training=True), -1)
                idA = tf_where(maskA, generator_BtoA(imagesA, training=True), -1)
                id_loss_AtoB = reduce_mean(tf_abs(imagesB - idB))
                id_loss_BtoA = reduce_mean(tf_abs(imagesA - idA))
                gen_AtoB_loss = gen_AtoB_loss + lambda_id * id_loss_AtoB
                gen_BtoA_loss = gen_BtoA_loss + lambda_id * id_loss_BtoA
            else:
                id_loss_AtoB = tf.constant(0.0)
                id_loss_BtoA = tf.constant(0.0)

            gen_AtoB_loss_s = _scale_loss(generator_AtoB_optimizer, gen_AtoB_loss)
            gen_BtoA_loss_s = _scale_loss(generator_BtoA_optimizer, gen_BtoA_loss)

        g1 = tape.gradient(gen_AtoB_loss_s, generator_AtoB.trainable_variables)
        g1 = _unscale_grads(generator_AtoB_optimizer, g1)
        generator_AtoB_optimizer.apply_gradients(zip(g1, generator_AtoB.trainable_variables))
        g2 = tape.gradient(gen_BtoA_loss_s, generator_BtoA.trainable_variables)
        g2 = _unscale_grads(generator_BtoA_optimizer, g2)
        generator_BtoA_optimizer.apply_gradients(zip(g2, generator_BtoA.trainable_variables))
        del tape
        return (gen_AtoB_adv_loss, gen_BtoA_adv_loss, cycle_loss_aba, cycle_loss_bab,
                id_loss_AtoB, id_loss_BtoA)

    # --- lambda schedule ------------------------------------------------------
    def get_lambda(epoch):
        if total_epochs <= 1:
            return args.lambda_cyc_init
        step = (args.lambda_cyc_init - args.lambda_cyc_end) / (total_epochs - 1)
        return args.lambda_cyc_init - epoch * step

    # --- init discriminator buffers ------------------------------------------
    print("\n=== Initializing discriminator buffers ===")
    discA_trainer.init_buffer([next(iterB) for _ in range(DISC_BUF_SIZE)])
    discB_trainer.init_buffer([next(iterA) for _ in range(DISC_BUF_SIZE)])

    @tf_function(input_signature=(TensorSpec(shape=(), dtype="float32"),), jit_compile=False)
    def train_step(lambda_cyc):
        for _ in range(args.n_steps_disc):
            imagesA = tf_concat([next(iterA) for _ in range(DISC_N_BATCHS * 2)], axis=0)
            imagesB = tf_concat([next(iterB) for _ in range(DISC_N_BATCHS)], axis=0)
            discA_loss = discA_trainer.train(imagesA, imagesB)
            imagesB = tf_concat([next(iterB) for _ in range(DISC_N_BATCHS * 2)], axis=0)
            imagesA = tf_concat([next(iterA) for _ in range(DISC_N_BATCHS)], axis=0)
            discB_loss = discB_trainer.train(imagesB, imagesA)
        imagesA = next(iterA)
        imagesB = next(iterB)
        (gAtoB_adv, gBtoA_adv, cyc_aba, cyc_bab, id_ab, id_ba) = \
            train_generators(imagesA, imagesB, lambda_cyc)
        return {"discA_loss": discA_loss, "discB_loss": discB_loss,
                "gen_AtoB_adv_loss": gAtoB_adv, "gen_BtoA_adv_loss": gBtoA_adv,
                "cycle_loss_aba": cyc_aba, "cycle_loss_bab": cyc_bab,
                "id_loss_AtoB": id_ab, "id_loss_BtoA": id_ba}

    # --- training loop --------------------------------------------------------
    print("\n=== Training ===")
    t_start = time()
    lambda_cyc = tf_Variable(0, dtype="float32")
    record = {}
    for epoch in range(total_epochs):
        set_lr(lr_for_epoch(epoch))
        tmp = {}
        for step in range(args.steps_per_epoch):
            lambda_cyc.assign(get_lambda(epoch))
            res = train_step(lambda_cyc)
            for k, v in res.items():
                tmp[k] = tmp.get(k, 0.0) + float(v.numpy())
            log = f"  step {step + 1}/{args.steps_per_epoch} epoch {epoch + 1}/{total_epochs} | "
            log += ", ".join(f"{k}={float(v.numpy()):.4f}" for k, v in res.items())
            print(log + " " * 8, end="\r")
        for k in tmp:
            record.setdefault(k, []).append(tmp[k] / args.steps_per_epoch)
        msg = f"Epoch {epoch + 1}/{total_epochs} (lr={lr_for_epoch(epoch):.2e}, "
        msg += f"lambda_cyc={get_lambda(epoch):.1f}) -> "
        msg += ", ".join(f"{k}:{record[k][-1]:.4f}" for k in record)
        print(msg + " " * 8)

        # Periodic checkpoint so a long run is recoverable.
        if (epoch + 1) % 25 == 0 or (epoch + 1) == total_epochs:
            generator_AtoB.save_weights(os.path.join(args.dest_dir, "generator_AtoB.h5"))
            generator_BtoA.save_weights(os.path.join(args.dest_dir, "generator_BtoA.h5"))

    # --- save final artifacts -------------------------------------------------
    generator_AtoB.save_weights(os.path.join(args.dest_dir, "generator_AtoB.h5"))
    generator_BtoA.save_weights(os.path.join(args.dest_dir, "generator_BtoA.h5"))
    discriminator_A.save_weights(os.path.join(args.dest_dir, "discriminator_A.h5"))
    discriminator_B.save_weights(os.path.join(args.dest_dir, "discriminator_B.h5"))
    with open(os.path.join(args.dest_dir, "stats.json"), "w") as f:
        json.dump(record, f)
    print("\n=== Done ===")
    print(f"Artifacts saved in: {args.dest_dir}")
    print("  generator_AtoB.h5  <- export this for harmonization")
    print("  generator_BtoA.h5, discriminator_A.h5, discriminator_B.h5")
    print("  norm_stats.json, stats.json")
    print(f"Elapsed: {time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
