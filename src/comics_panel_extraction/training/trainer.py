from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
import keras

from comics_panel_extraction.model.unetpp import build_unetpp
from comics_panel_extraction.training.config import TrainingConfig
from comics_panel_extraction.training.loss import RegionBCEDiceLoss
from comics_panel_extraction.training.optimizer import WarmupCosineDecaySchedule

class Trainer:
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        steps_per_epoch: Optional[int] = None,
    ):
        self.config = config or TrainingConfig()
        keras.utils.set_random_seed(self.config.seed)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = build_unetpp(input_shape=self.config.input_shape)
        self.loss_fn = RegionBCEDiceLoss()

        total_steps = max(1, (steps_per_epoch or 100) * self.config.epochs)
        warmup_steps = max(1, (steps_per_epoch or 100) * self.config.warmup_epochs)
        self.lr_schedule = WarmupCosineDecaySchedule(
            base_learning_rate=self.config.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
        )
        self.optimizer = keras.optimizers.AdamW(
            learning_rate=self.lr_schedule,
            weight_decay=self.config.weight_decay,
        )

    @tf.function
    def _compiled_train_step(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            preds = self.model(x, training=True)
            loss = self.loss_fn(y, preds)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    @tf.function
    def _compiled_eval_step(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        preds = self.model(x, training=False)
        return self.loss_fn(y, preds)

    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> float:
        return float(self._compiled_train_step(x, y).numpy())

    def eval_step(self, x: tf.Tensor, y: tf.Tensor) -> float:
        return float(self._compiled_eval_step(x, y).numpy())

    def train_dataset(
        self,
        train_ds: tf.data.Dataset,
        val_ds: Optional[tf.data.Dataset] = None,
        steps_per_epoch: int = 100,
        val_steps: int = 10,
    ) -> Dict[str, Any]:
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": [], "lr": []}

        # One iterator for the whole run so each pass over the data gets a new shuffle order.
        train_iter = iter(train_ds)
        for epoch in range(self.config.epochs):
            epoch_losses = []
            for _ in range(steps_per_epoch):
                batch = next(train_iter, None)
                if batch is None:
                    train_iter = iter(train_ds)
                    batch = next(train_iter, None)
                    if batch is None:
                        break
                x, y = batch
                loss_val = self.train_step(x, y)
                epoch_losses.append(loss_val)

            mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            history["train_loss"].append(mean_train_loss)

            current_step = (epoch + 1) * steps_per_epoch
            current_lr = float(self.lr_schedule(current_step).numpy())
            history["lr"].append(current_lr)

            if val_ds is not None:
                val_losses = []
                for v_step, (vx, vy) in enumerate(val_ds):
                    if v_step >= val_steps:
                        break
                    v_loss = self.eval_step(vx, vy)
                    val_losses.append(v_loss)
                mean_val_loss = float(np.mean(val_losses)) if val_losses else mean_train_loss
                history["val_loss"].append(mean_val_loss)
            else:
                mean_val_loss = mean_train_loss

            improved = mean_val_loss < best_val_loss
            if improved:
                best_val_loss = mean_val_loss
                self.model.save(str(self.output_dir / "best_model.keras"))

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - train_loss: {mean_train_loss:.4f}"
                f" - val_loss: {mean_val_loss:.4f} - lr: {current_lr:.2e}"
                f"{' - saved best' if improved else ''}",
                flush=True,
            )

        self.model.save(str(self.output_dir / "latest_model.keras"))
        return {
            "best_loss": best_val_loss,
            "best_val_loss": best_val_loss,
            "history": history,
        }
