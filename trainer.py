"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import shutil
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model weights updated as:
        ema_param = decay * ema_param + (1 - decay) * param

    Usage: call ``update()`` after each optimizer step; use ``apply()`` /
    ``restore()`` around evaluation to temporarily swap in EMA weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        # Store EMA shadow params on CPU to save GPU memory
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.data.clone().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters after an optimizer step."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.cpu() * (1.0 - self.decay)
                )

    def apply(self, model: nn.Module) -> None:
        """Swap in EMA weights for evaluation."""
        self._backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device))

    def restore(self, model: nn.Module) -> None:
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        query_div_weight: float = 0.01,
        ema_decay: float = 0.999,
        dig_mode: str = 'all',
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.query_div_weight: float = query_div_weight
        assert dig_mode in ('all', 'random'), \
            f"dig_mode must be 'all' or 'random', got {dig_mode!r}"
        self.dig_mode: str = dig_mode

        # EMA
        self.ema: Optional[ModelEMA] = None
        if ema_decay > 0.0:
            self.ema = ModelEMA(model, decay=ema_decay)
            logging.info(f"EMA enabled with decay={ema_decay}")

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                     f"query_div_weight={query_div_weight}")

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            # No new best anticipated: leave disk untouched. The previous
            # best_model dir (with its model.pt + sidecars) remains valid.
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        # Point EarlyStopping at the canonical best-model location for this
        # step. Only done on the likely-new-best branch so that a skipped
        # save never leaks the unused path into EarlyStopping state.
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # Remove stale best dirs first so EarlyStopping's write is the only
        # I/O needed when a new best is confirmed.
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        # Write sidecar files only when EarlyStopping actually confirmed a
        # new best and wrote model.pt. If the score tripped our heuristic
        # but EarlyStopping internally declined to save, skip to avoid
        # creating an empty (sidecar-only) checkpoint directory.
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0
            div_loss_sum = 0.0
            # Per-reveal-ratio accumulators for DIG mode (filled lazily)
            dig_loss_sums: Optional[list] = None

            for step, batch in train_pbar:
                loss, dig_losses, div_loss_val = self._train_step(batch)
                total_step += 1
                loss_sum += loss
                div_loss_sum += div_loss_val

                # --- TensorBoard ---
                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)
                    if self.query_div_weight > 0:
                        self.writer.add_scalar('Loss/train_div', div_loss_val, total_step)
                    if dig_losses is not None:
                        K = len(dig_losses)
                        for k, dl in enumerate(dig_losses):
                            ratio = (k + 1) / K
                            self.writer.add_scalar(
                                f'Loss/train_dig_r{ratio:.2f}', dl, total_step)

                # --- tqdm postfix ---
                postfix: Dict[str, str] = {"loss": f"{loss:.4f}"}
                if self.query_div_weight > 0:
                    postfix["div"] = f"{div_loss_val:.4f}"
                if dig_losses is not None:
                    K = len(dig_losses)
                    for k, dl in enumerate(dig_losses):
                        ratio = (k + 1) / K
                        postfix[f"r{ratio:.2f}"] = f"{dl:.4f}"
                train_pbar.set_postfix(postfix)

                # --- DIG per-ratio accumulator ---
                if dig_losses is not None:
                    if dig_loss_sums is None:
                        dig_loss_sums = [0.0] * len(dig_losses)
                    for k, dl in enumerate(dig_losses):
                        dig_loss_sums[k] += dl

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                    self._handle_validation_result(total_step, val_auc, val_logloss)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            n_steps = len(self.train_loader)
            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / n_steps}")
            if self.query_div_weight > 0:
                logging.info(f"Epoch {epoch}, Average Div Loss: {div_loss_sum / n_steps:.4f} "
                             f"(weighted: {self.query_div_weight * div_loss_sum / n_steps:.4f})")
            if dig_loss_sums is not None:
                K = len(dig_loss_sums)
                parts = " | ".join(
                    f"r{(k+1)/K:.2f}={dig_loss_sums[k]/n_steps:.4f}"
                    for k in range(K)
                )
                logging.info(f"Epoch {epoch}, DIG Average Loss: {parts}")

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            self._handle_validation_result(total_step, val_auc, val_logloss)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        seq_hours: Dict[str, torch.Tensor] = {}
        seq_weekdays: Dict[str, torch.Tensor] = {}
        seq_diff_hours: Dict[str, torch.Tensor] = {}
        seq_diff_days: Dict[str, torch.Tensor] = {}
        seq_diff_weeks: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            # New temporal fields (optional; absent in old checkpoints or disabled datasets)
            if f'{domain}_seq_hour' in device_batch:
                seq_hours[domain] = device_batch[f'{domain}_seq_hour']
            if f'{domain}_seq_weekday' in device_batch:
                seq_weekdays[domain] = device_batch[f'{domain}_seq_weekday']
            if f'{domain}_diff_hours' in device_batch:
                seq_diff_hours[domain] = device_batch[f'{domain}_diff_hours']
            if f'{domain}_diff_days' in device_batch:
                seq_diff_days[domain] = device_batch[f'{domain}_diff_days']
            if f'{domain}_diff_weeks' in device_batch:
                seq_diff_weeks[domain] = device_batch[f'{domain}_diff_weeks']
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            req_hour=device_batch.get('req_hour'),
            req_weekday=device_batch.get('req_weekday'),
            req_week_of_year=device_batch.get('req_week_of_year'),
            seq_hours=seq_hours if seq_hours else None,
            seq_weekdays=seq_weekdays if seq_weekdays else None,
            seq_diff_hours=seq_diff_hours if seq_diff_hours else None,
            seq_diff_days=seq_diff_days if seq_diff_days else None,
            seq_diff_weeks=seq_diff_weeks if seq_diff_weeks else None,
        )

    def _compute_loss(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute loss from logits and labels."""
        if self.loss_type == 'focal':
            return sigmoid_focal_loss(logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
        return F.binary_cross_entropy_with_logits(logits, label)

    def _train_step(self, batch: Dict[str, Any]) -> Tuple[float, Optional[list], float]:
        """Run a single training step and return ``(avg_loss, dig_losses, div_loss)``.

        ``dig_losses`` is a list of per-reveal-ratio task losses when
        sid_mode='fid_order', otherwise ``None``.
        ``div_loss`` is the scalar query diversity regularisation loss value
        (before weighting); 0.0 when query_div_weight == 0 or num_queries <= 1.

        r6 (sid_mode='fid_order'): runs dig_steps forward passes with
        reveal_ratio = 1/K, 2/K, …, 1.  The final loss is the mean over
        all K passes, which forces the model to predict from partial feature
        sets (coarse-to-fine regularisation, DIG-style).
        """
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)

        # Check if model is in fid_order mode
        use_dig = (
            hasattr(self.model, 'sid_mode')
            and self.model.sid_mode == 'fid_order'
        )

        dig_losses: Optional[list] = None
        div_loss_tensor = torch.tensor(0.0, device=self.device)
        if use_dig:
            K = self.model.dig_steps

            if self.dig_mode == 'random':
                # Random+Full DIG: always run reveal_ratio=1.0 (full-feature, main
                # task), then randomly sample ONE coarse step from {1/K,...,(K-1)/K}
                # as a regularisation auxiliary loss.
                # Total cost = 2 forward passes (vs K in 'all' mode).
                # The full-feature step is the anchor; the coarse step prevents
                # the model from ignoring low-granularity representations.
                # Both steps call backward() immediately to release activation graphs.

                # --- Step 1: full-feature forward (always) ---
                logits, step_div_loss = self.model(model_input, reveal_ratio=1.0)
                logits = logits.squeeze(-1)
                full_loss = self._compute_loss(logits, label)
                div_loss_tensor = step_div_loss
                if self.query_div_weight > 0:
                    full_loss = full_loss + self.query_div_weight * step_div_loss
                full_loss.backward()

                # --- Step 2: one random coarse step from {1/K,...,(K-1)/K} ---
                coarse_idx = int(torch.randint(K - 1, (1,)).item())  # 0..K-2
                coarse_ratio = (coarse_idx + 1) / K
                logits_c, _ = self.model(model_input, reveal_ratio=coarse_ratio)
                logits_c = logits_c.squeeze(-1)
                coarse_loss = self._compute_loss(logits_c, label)
                # Weight coarse loss relative to full loss (half weight to keep
                # the main task dominant)
                (0.5 * coarse_loss).backward()

                dig_losses = [full_loss.item(), coarse_loss.item()]
                loss = full_loss.item()
            else:
                # DIG training: multiple forward passes with increasing reveal_ratio.
                # reveal_ratios = [1/K, 2/K, ..., K/K=1.0]
                #
                # Loss weighting: linearly increasing weights so that the full-feature
                # step contributes more than coarse-only steps.  Fine-grained fids
                # (high vocab_size) only appear in later steps, so upweighting them
                # compensates for the asymmetric training frequency.
                #   weights[i] = (i+1) / sum(1..K) = (i+1) / (K*(K+1)/2)
                #
                # div_loss is only added at the last step (reveal_ratio=1.0): at
                # coarse steps the Q tokens are mostly zeroed-out, so pairwise
                # cosine similarity is spuriously high and would inject noisy
                # gradients into the query projection parameters.
                #
                # Memory-efficient gradient accumulation: each sub-step calls
                # loss.backward() immediately so only ONE forward pass graph is
                # live in GPU memory at a time (instead of K graphs all kept until
                # the final backward).  Optimizer step is deferred to after the loop.
                weight_sum = K * (K + 1) / 2  # sum of 1..K
                dig_losses = []
                total_loss_val = 0.0
                for step_idx in range(K):
                    reveal_ratio = (step_idx + 1) / K
                    step_weight = (step_idx + 1) / weight_sum  # linearly increasing
                    logits, step_div_loss = self.model(model_input, reveal_ratio=reveal_ratio)
                    logits = logits.squeeze(-1)
                    step_loss = self._compute_loss(logits, label)
                    # Accumulate div_loss only from the last (full-feature) step
                    if step_idx == K - 1:
                        div_loss_tensor = step_div_loss
                        if self.query_div_weight > 0:
                            step_loss = step_loss + self.query_div_weight * step_div_loss
                    weighted = step_weight * step_loss
                    # backward immediately to free this step's activation graph;
                    # retain_graph=False (default) drops the graph right away.
                    weighted.backward()
                    total_loss_val += weighted.item()
                    dig_losses.append(step_loss.item())
                loss = total_loss_val  # scalar float, used only for logging
        else:
            logits, div_loss_tensor = self.model(model_input)  # (B, 1), scalar
            logits = logits.squeeze(-1)  # (B,)
            loss = self._compute_loss(logits, label)

            # Add query diversity regularisation loss (method B)
            if self.query_div_weight > 0:
                loss = loss + self.query_div_weight * div_loss_tensor

            loss.backward()
            loss = loss.item()

        # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
        # with certain tensor shapes in this project.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        # Update EMA shadow parameters after optimizer step
        if self.ema is not None:
            self.ema.update(self.model)

        loss_val = loss if isinstance(loss, float) else loss.item()
        return loss_val, dig_losses, div_loss_tensor.item()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        When EMA is enabled, validation is always run with EMA weights.
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        # Swap in EMA weights for evaluation
        if self.ema is not None:
            self.ema.apply(self.model)
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        # Filter NaN predictions (may appear if gradients explode).
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        # Binary logloss (same NaN filtering).
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        # Restore original weights after evaluation
        if self.ema is not None:
            self.ema.restore(self.model)

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)

        return logits, label
