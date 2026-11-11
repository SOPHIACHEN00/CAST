import sys
sys.path.append("./uni2ts/src")

import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # Optional: required only by Moirai attacks.
    hf_hub_download = None

try:
    from pmdarima import auto_arima
except ImportError:  # Optional: required only by ARIMA attacks.
    auto_arima = None

try:
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from uni2ts.model.moirai.module import decode_distr_output
except ImportError:  # Optional: required only by Moirai attacks.
    MoiraiForecast = None
    MoiraiModule = None
    decode_distr_output = None

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:  # pragma: no cover - safetensors is expected but keep a fallback
    load_safetensors_file = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _extract_moirai_module_kwargs(raw_config):
    candidates = [
        raw_config,
        raw_config.get("module_kwargs") if isinstance(raw_config, dict) else None,
        raw_config.get("model_config") if isinstance(raw_config, dict) else None,
        raw_config.get("module_config") if isinstance(raw_config, dict) else None,
    ]
    required = {
        "distr_output",
        "d_model",
        "num_layers",
        "patch_sizes",
        "max_seq_len",
        "attn_dropout_p",
        "dropout_p",
    }

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if required.issubset(candidate.keys()):
            module_kwargs = dict(candidate)
            if isinstance(module_kwargs.get("distr_output"), dict):
                module_kwargs["distr_output"] = decode_distr_output(module_kwargs["distr_output"])
            if isinstance(module_kwargs.get("patch_sizes"), list):
                module_kwargs["patch_sizes"] = tuple(module_kwargs["patch_sizes"])
            return module_kwargs

    raise KeyError(f"Could not find MoiraiModule kwargs in config keys: {list(raw_config.keys()) if isinstance(raw_config, dict) else type(raw_config)}")


def _resolve_moirai_source(size):
    local_path = PROJECT_ROOT / f"moirai-1.1-R-{size}"
    if local_path.exists():
        print(f"[Moirai][LOAD] using local model path: {local_path}")
        return str(local_path)
    return f"Salesforce/moirai-1.1-R-{size}"


def _load_moirai_module(repo_or_path):
    repo_or_path = str(repo_or_path)
    try:
        return MoiraiModule.from_pretrained(repo_or_path)
    except TypeError as exc:
        # Newer hub/code combinations sometimes fail to hydrate init kwargs for MoiraiModule.
        msg = str(exc)
        required_markers = [
            "distr_output",
            "d_model",
            "num_layers",
            "patch_sizes",
            "max_seq_len",
            "attn_dropout_p",
            "dropout_p",
        ]
        if not all(marker in msg for marker in required_markers):
            raise

        print(f"[Moirai][LOAD] direct from_pretrained failed for {repo_or_path}; trying manual config+weights fallback")

        if Path(repo_or_path).exists():
            config_path = Path(repo_or_path) / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Missing config.json under local Moirai path: {repo_or_path}")

            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)
            module_kwargs = _extract_moirai_module_kwargs(raw_config)
            module = MoiraiModule(**module_kwargs)

            safetensor_path = Path(repo_or_path) / "model.safetensors"
            pytorch_bin_path = Path(repo_or_path) / "pytorch_model.bin"

            if safetensor_path.exists():
                if load_safetensors_file is None:
                    raise ImportError("safetensors is not available but model.safetensors is present")
                state_dict = load_safetensors_file(str(safetensor_path), device="cpu")
            elif pytorch_bin_path.exists():
                state_dict = torch.load(str(pytorch_bin_path), map_location="cpu")
            else:
                raise FileNotFoundError(
                    f"Neither model.safetensors nor pytorch_model.bin found under {repo_or_path}"
                )

            module.load_state_dict(state_dict, strict=True)
            return module

        config_path = hf_hub_download(repo_or_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        module_kwargs = _extract_moirai_module_kwargs(raw_config)
        module = MoiraiModule(**module_kwargs)

        state_dict = None
        safetensor_error = None
        try:
            weight_path = hf_hub_download(repo_or_path, "model.safetensors")
            if load_safetensors_file is None:
                raise ImportError("safetensors is not available")
            state_dict = load_safetensors_file(weight_path, device="cpu")
        except Exception as inner_exc:
            safetensor_error = inner_exc

        if state_dict is None:
            try:
                weight_path = hf_hub_download(repo_or_path, "pytorch_model.bin")
                state_dict = torch.load(weight_path, map_location="cpu")
            except Exception:
                raise RuntimeError(
                    f"Failed to load Moirai weights for {repo_or_path} via both safetensors and pytorch_model.bin"
                ) from safetensor_error

        module.load_state_dict(state_dict, strict=True)
        return module


def _avg_history_np(history, k):
    window = history[-min(len(history), max(1, int(k))):]
    return sum(window) / float(len(window))


def _flatten_grad_list_to_np(global_gradient):
    return torch.cat([g.reshape(-1).detach().cpu() for g in global_gradient]).numpy().astype(np.float32)


def _log_fallback(tag, round_num, mode, reason, extra=""):
    suffix = f" {extra}" if extra else ""
    print(f"[{tag}][FALLBACK] round={round_num} mode={mode} reason={reason}{suffix}")


def _is_constant_series(series, atol=1e-12):
    arr = np.asarray(series, dtype=np.float32)
    if arr.size <= 1:
        return True
    return bool(np.all(np.abs(arr - arr[0]) <= float(atol)))


def _constant_series_model(series):
    arr = np.asarray(series, dtype=np.float32)
    if arr.size == 0:
        return ("constant", 0.0)
    return ("constant", float(np.mean(arr)))


def zeros_hessian_provider(model, device):
    return [torch.zeros_like(p, device=device) for p in model.parameters()]


@torch.no_grad()
def frobenius_scaled_identity_provider(model, device, scale=1e-3):
    return [scale * torch.ones_like(p, device=device) for p in model.parameters()]


def hutchinson_hvp_provider(model, device, loss_fn, batch):
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model, batch)
    grad1 = torch.autograd.grad(loss, list(model.parameters()), create_graph=True, retain_graph=True)
    v = [torch.sign(torch.randn_like(g)) for g in grad1]
    dot = sum((g * vv).sum() for g, vv in zip(grad1, v))
    hvp = torch.autograd.grad(dot, list(model.parameters()), retain_graph=False, create_graph=False)
    return [h for h in hvp]


class _DKFRLiteGenerator(nn.Module):
    """
    NAYER-style lightweight generator for already-preprocessed benchmark inputs.

    It maps random labels plus a noisy layer to the exact feature space consumed
    by the current MLP benchmark models, so no dataset preprocessing is changed.
    """
    def __init__(
        self,
        input_dim,
        num_classes,
        label_dim=32,
        noise_dim=32,
        hidden_dim=128,
        feature_min=None,
        feature_max=None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.noise_dim = int(noise_dim)

        self.label_embedding = nn.Embedding(self.num_classes, int(label_dim))
        self.noisy_layer = nn.Linear(int(label_dim) + self.noise_dim, int(hidden_dim))
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.input_dim),
        )

        if feature_min is None:
            feature_min = torch.full((self.input_dim,), -1.0)
        if feature_max is None:
            feature_max = torch.full((self.input_dim,), 1.0)
        self.register_buffer("feature_min", feature_min.detach().float().view(1, -1))
        self.register_buffer("feature_max", feature_max.detach().float().view(1, -1))
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, labels):
        labels = labels.long().view(-1)
        z = torch.randn(labels.size(0), self.noise_dim, device=labels.device)
        label_vec = self.label_embedding(labels)
        h = self.noisy_layer(torch.cat([label_vec, z], dim=1))
        raw = self.net(h)
        span = (self.feature_max - self.feature_min).clamp_min(1e-6)
        return torch.sigmoid(raw) * span + self.feature_min


class _DKFRMNISTImageGenerator(nn.Module):
    """
    Label-conditioned image generator for MNIST's existing flattened,
    normalized preprocessing: x = (pixel_0_1 - 0.1307) / 0.3081.
    """
    def __init__(
        self,
        num_classes,
        label_dim=32,
        noise_dim=64,
        hidden_dim=128,
        mnist_mean=0.1307,
        mnist_std=0.3081,
    ):
        super().__init__()
        self.input_dim = 28 * 28
        self.num_classes = int(num_classes)
        self.noise_dim = int(noise_dim)
        self.mnist_mean = float(mnist_mean)
        self.mnist_std = float(mnist_std)

        self.label_embedding = nn.Embedding(self.num_classes, int(label_dim))
        self.project = nn.Sequential(
            nn.Linear(int(label_dim) + self.noise_dim, int(hidden_dim) * 7 * 7),
            nn.BatchNorm1d(int(hidden_dim) * 7 * 7),
            nn.ReLU(),
        )
        self.image_net = nn.Sequential(
            nn.ConvTranspose2d(int(hidden_dim), 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, labels):
        labels = labels.long().view(-1)
        z = torch.randn(labels.size(0), self.noise_dim, device=labels.device)
        label_vec = self.label_embedding(labels)
        h = self.project(torch.cat([label_vec, z], dim=1))
        h = h.view(labels.size(0), -1, 7, 7)
        pixels = self.image_net(h).view(labels.size(0), -1)
        return (pixels - self.mnist_mean) / self.mnist_std


class _DKFREmbeddingGenerator(nn.Module):
    """
    Label-conditioned generator for continuous encoder embeddings such as IMDB
    DistilBERT CLS vectors. It uses global embedding mean/std shape, not minmax
    pixel/tabular bounds.
    """
    def __init__(
        self,
        input_dim,
        num_classes,
        label_dim=32,
        noise_dim=64,
        hidden_dim=256,
        feature_mean=None,
        feature_std=None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.noise_dim = int(noise_dim)
        self.label_embedding = nn.Embedding(self.num_classes, int(label_dim))
        self.net = nn.Sequential(
            nn.Linear(int(label_dim) + self.noise_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.input_dim),
            nn.Tanh(),
        )
        if feature_mean is None:
            feature_mean = torch.zeros(self.input_dim)
        if feature_std is None:
            feature_std = torch.ones(self.input_dim)
        self.register_buffer("feature_mean", feature_mean.detach().float().view(1, -1))
        self.register_buffer("feature_std", feature_std.detach().float().view(1, -1).clamp_min(1e-6))
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, labels):
        labels = labels.long().view(-1)
        z = torch.randn(labels.size(0), self.noise_dim, device=labels.device)
        label_vec = self.label_embedding(labels)
        normalized = self.net(torch.cat([label_vec, z], dim=1))
        return self.feature_mean + normalized * (2.0 * self.feature_std)


class DKFRLiteAttack:
    """
    Data-free KD free-rider baseline adapted to this repo's benchmark inputs.

    Teacher: current global model received from the server.
    Student/local: attacker's saved local model from the previous global round.
    Generator: produces synthetic feature tensors in the current preprocessed
    input space. The implementation intentionally avoids any access to other
    clients' local gradients or data.
    """
    def __init__(
        self,
        base_model,
        dataset,
        input_dim,
        num_classes,
        feature_min=None,
        feature_max=None,
        feature_mean=None,
        feature_std=None,
        synth_batch_size=64,
        generator_steps=5,
        kd_steps=5,
        lr_g=1e-3,
        lr_student=1e-3,
        temperature=2.0,
        lambda_cls_g=1.0,
        lambda_kl_g=1.0,
        alpha_kl=1.0,
        alpha_cls=1.0,
        alpha_feat=0.1,
        label_dim=32,
        noise_dim=32,
        hidden_dim=128,
        grad_clip=5.0,
        max_update_norm=0.0,
        kl_cap=0.0,
        student_kl_cap=0.0,
        student_loss_cap=0.0,
        teacher_conf_min=0.0,
        guard_reset_patience=3,
        mnist_tv_weight=0.0,
        mnist_pixel_reg_weight=0.0,
        mnist_logit_reg_weight=0.0,
    ):
        self.dataset = str(dataset).lower()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.synth_batch_size = int(synth_batch_size)
        self.generator_steps = int(generator_steps)
        self.kd_steps = int(kd_steps)
        self.lr_g = float(lr_g)
        self.lr_student = float(lr_student)
        self.temperature = float(temperature)
        self.lambda_cls_g = float(lambda_cls_g)
        self.lambda_kl_g = float(lambda_kl_g)
        self.alpha_kl = float(alpha_kl)
        self.alpha_cls = float(alpha_cls)
        self.alpha_feat = float(alpha_feat)
        self.grad_clip = float(grad_clip)
        self.max_update_norm = float(max_update_norm)
        self.kl_cap = float(kl_cap)
        self.student_kl_cap = float(student_kl_cap)
        self.student_loss_cap = float(student_loss_cap)
        self.teacher_conf_min = float(teacher_conf_min)
        self.guard_reset_patience = int(guard_reset_patience)
        self.mnist_tv_weight = float(mnist_tv_weight)
        self.mnist_pixel_reg_weight = float(mnist_pixel_reg_weight)
        self.mnist_logit_reg_weight = float(mnist_logit_reg_weight)
        self.guard_streak = 0
        self.global_gradient_history = []

        feature_min = None if feature_min is None else feature_min.detach().cpu().float()
        feature_max = None if feature_max is None else feature_max.detach().cpu().float()
        feature_mean = None if feature_mean is None else feature_mean.detach().cpu().float()
        feature_std = None if feature_std is None else feature_std.detach().cpu().float()

        if self.dataset == "mnist":
            self.generator = _DKFRMNISTImageGenerator(
                num_classes=self.num_classes,
                label_dim=label_dim,
                noise_dim=max(noise_dim, 64),
                hidden_dim=max(hidden_dim, 64),
            )
            self.generator_kind = "mnist_image"
        elif self.dataset == "imdb":
            self.generator = _DKFREmbeddingGenerator(
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                label_dim=label_dim,
                noise_dim=max(noise_dim, 64),
                hidden_dim=max(hidden_dim, 256),
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
            self.generator_kind = "embedding"
        else:
            self.generator = _DKFRLiteGenerator(
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                label_dim=label_dim,
                noise_dim=noise_dim,
                hidden_dim=hidden_dim,
                feature_min=feature_min,
                feature_max=feature_max,
            )
            self.generator_kind = "tabular_minmax"
        self.local_state = copy.deepcopy(base_model.state_dict())

    def _is_multiclass(self):
        return self.num_classes > 2

    def _clone_model(self, model, device):
        cloned = copy.deepcopy(model).to(device)
        return cloned

    def _set_requires_grad(self, model, enabled):
        for param in model.parameters():
            param.requires_grad_(enabled)

    def _params_are_finite(self, module):
        return all(torch.isfinite(param).all().item() for param in module.parameters())

    def _clip_gradients(self, parameters):
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(parameters), max_norm=self.grad_clip)

    def _zero_fake_gradient(self, model, device):
        return [torch.zeros_like(param, device=device) for param in model.parameters()]

    def _clone_state_to_cpu(self, module):
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    def _restore_state(self, module, state):
        module.load_state_dict({k: v.to(next(module.parameters()).device) for k, v in state.items()})

    def _sample_labels(self, device):
        return torch.randint(
            low=0,
            high=self.num_classes,
            size=(self.synth_batch_size,),
            device=device,
        )

    def _classification_loss(self, outputs, labels):
        if self._is_multiclass():
            return F.cross_entropy(outputs, labels.long().view(-1))
        probs = outputs.view(-1, 1).clamp(1e-6, 1.0 - 1e-6)
        targets = labels.float().view(-1, 1)
        return F.binary_cross_entropy(probs, targets)

    def _kl_loss(self, teacher_outputs, student_outputs):
        temp = max(self.temperature, 1e-6)
        if self._is_multiclass():
            teacher_probs = F.softmax(teacher_outputs / temp, dim=1)
            student_log_probs = F.log_softmax(student_outputs / temp, dim=1)
            return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp ** 2)

        t = teacher_outputs.view(-1, 1).clamp(1e-6, 1.0 - 1e-6)
        s = student_outputs.view(-1, 1).clamp(1e-6, 1.0 - 1e-6)
        return (t * (t.log() - s.log()) + (1.0 - t) * ((1.0 - t).log() - (1.0 - s).log())).mean()

    def _generator_kl_term(self, kl_loss):
        if self.kl_cap > 0:
            return self._smooth_cap_loss(kl_loss, self.kl_cap)
        return kl_loss

    def _student_kl_term(self, kl_loss):
        if self.student_kl_cap > 0:
            return self._smooth_cap_loss(kl_loss, self.student_kl_cap)
        return kl_loss

    def _student_loss_term(self, loss):
        if self.student_loss_cap > 0:
            return self._smooth_cap_loss(loss, self.student_loss_cap)
        return loss

    def _smooth_cap_loss(self, loss, cap):
        cap_tensor = torch.as_tensor(float(cap), device=loss.device, dtype=loss.dtype)
        return torch.where(loss <= cap_tensor, loss, cap_tensor + torch.log1p(loss - cap_tensor))

    def _maybe_reset_generator_after_guard(self):
        if self.guard_reset_patience > 0 and self.guard_streak >= self.guard_reset_patience:
            self.generator.reset_parameters()
            self.guard_streak = 0
            return True
        return False

    def _mnist_image_prior_loss(self, x_hat):
        if self.dataset != "mnist":
            return torch.zeros((), device=x_hat.device)
        if self.mnist_tv_weight <= 0 and self.mnist_pixel_reg_weight <= 0:
            return torch.zeros((), device=x_hat.device)

        pixels = (x_hat.view(-1, 1, 28, 28) * 0.3081 + 0.1307).clamp(0.0, 1.0)
        reg = torch.zeros((), device=x_hat.device)
        if self.mnist_tv_weight > 0:
            tv_h = (pixels[:, :, 1:, :] - pixels[:, :, :-1, :]).abs().mean()
            tv_w = (pixels[:, :, :, 1:] - pixels[:, :, :, :-1]).abs().mean()
            reg = reg + self.mnist_tv_weight * (tv_h + tv_w)
        if self.mnist_pixel_reg_weight > 0:
            mean_loss = (pixels.mean() - 0.1307).pow(2)
            std_loss = (pixels.std(unbiased=False) - 0.3081).pow(2)
            reg = reg + self.mnist_pixel_reg_weight * (mean_loss + std_loss)
        return reg

    def _mnist_logit_regularizer(self, teacher_outputs, student_outputs):
        if self.dataset != "mnist" or self.mnist_logit_reg_weight <= 0:
            return torch.zeros((), device=student_outputs.device)
        student_reg = student_outputs.float().pow(2).mean()
        teacher_reg = teacher_outputs.detach().float().pow(2).mean()
        return self.mnist_logit_reg_weight * (student_reg + 0.1 * teacher_reg)

    def _features(self, model, x):
        if hasattr(model, "fc1") and hasattr(model, "relu"):
            h = model.fc1(x)
            return model.relu(h)
        return model(x)

    def _feature_loss(self, teacher, student, x):
        teacher_feat = self._features(teacher, x).detach()
        student_feat = self._features(student, x)
        if teacher_feat.shape != student_feat.shape:
            return torch.zeros((), device=x.device)
        return 1.0 - F.cosine_similarity(
            teacher_feat.view(teacher_feat.size(0), -1),
            student_feat.view(student_feat.size(0), -1),
            dim=1,
            eps=1e-8,
        ).mean()

    def _train_generator(self, teacher, student, device):
        self.generator.to(device)
        self.generator.train(True)
        teacher.eval()
        student.eval()
        self._set_requires_grad(teacher, False)
        self._set_requires_grad(student, False)
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g)
        last_loss = None

        for _ in range(max(1, self.generator_steps)):
            safe_state = self._clone_state_to_cpu(self.generator)
            labels = self._sample_labels(device)
            x_hat = self.generator(labels)
            teacher_outputs = teacher(x_hat)
            cls_loss = self._classification_loss(teacher_outputs, labels)
            if self.lambda_kl_g != 0:
                student_outputs = student(x_hat)
                kl_loss = self._kl_loss(teacher_outputs, student_outputs)
                kl_term = self._generator_kl_term(kl_loss)
            else:
                kl_term = torch.zeros((), device=device)
            prior_loss = self._mnist_image_prior_loss(x_hat)
            loss = self.lambda_cls_g * cls_loss - self.lambda_kl_g * kl_term + prior_loss
            if not torch.isfinite(loss).all().item():
                self._restore_state(self.generator, safe_state)
                return None

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._clip_gradients(self.generator.parameters())
            optimizer.step()
            if not self._params_are_finite(self.generator):
                self._restore_state(self.generator, safe_state)
                return None
            last_loss = float(loss.detach().item())

        return last_loss

    def _train_student(self, teacher, student, device):
        self.generator.to(device)
        self.generator.eval()
        teacher.eval()
        student.train(True)
        self._set_requires_grad(teacher, False)
        self._set_requires_grad(student, True)
        optimizer = torch.optim.Adam(student.parameters(), lr=self.lr_student)
        last_loss = None

        for _ in range(max(1, self.kd_steps)):
            safe_state = self._clone_state_to_cpu(student)
            labels = self._sample_labels(device)
            with torch.no_grad():
                x_hat = self.generator(labels)
                teacher_outputs = teacher(x_hat)
            student_outputs = student(x_hat)

            kd_teacher_outputs = teacher_outputs
            kd_student_outputs = student_outputs
            if self._is_multiclass() and self.teacher_conf_min > 0:
                teacher_probs = F.softmax(teacher_outputs / max(self.temperature, 1e-6), dim=1)
                confidence = teacher_probs.max(dim=1).values
                mask = confidence >= self.teacher_conf_min
                if mask.any().item():
                    kd_teacher_outputs = teacher_outputs[mask]
                    kd_student_outputs = student_outputs[mask]
                else:
                    kd_teacher_outputs = None
                    kd_student_outputs = None

            if kd_teacher_outputs is None:
                kd_loss = torch.zeros((), device=device)
            else:
                kd_loss = self._kl_loss(kd_teacher_outputs, kd_student_outputs)
            kd_term = self._student_kl_term(kd_loss)

            if self.dataset == "mnist" and self._is_multiclass():
                cls_labels = teacher_outputs.detach().argmax(dim=1)
            else:
                cls_labels = labels
            cls_loss = self._classification_loss(student_outputs, cls_labels)
            feat_loss = self._feature_loss(teacher, student, x_hat)
            logit_reg = self._mnist_logit_regularizer(teacher_outputs, student_outputs)
            raw_loss = self.alpha_kl * kd_term + self.alpha_cls * cls_loss + self.alpha_feat * feat_loss + logit_reg
            loss = self._student_loss_term(raw_loss)
            if not torch.isfinite(loss).all().item():
                self._restore_state(student, safe_state)
                return None

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._clip_gradients(student.parameters())
            optimizer.step()
            if not self._params_are_finite(student):
                self._restore_state(student, safe_state)
                return None
            last_loss = float(loss.detach().item())

        return last_loss

    def get_fake_gradient(self, round_num, device, model):
        generator_safe_state = self._clone_state_to_cpu(self.generator)
        teacher = self._clone_model(model, device)
        teacher.load_state_dict(model.state_dict())
        teacher.eval()

        student = self._clone_model(model, device)
        student.load_state_dict(self.local_state)
        before_state = copy.deepcopy(model.state_dict())

        gen_loss = self._train_generator(teacher, student, device)
        kd_loss = self._train_student(teacher, student, device)

        guarded = gen_loss is None or kd_loss is None

        fake_gradient = []
        if guarded:
            self._restore_state(self.generator, generator_safe_state)
            fake_gradient = self._zero_fake_gradient(model, device)
        else:
            for name, param in model.named_parameters():
                update = student.state_dict()[name].to(device=device, dtype=param.dtype) - before_state[name].to(device=device, dtype=param.dtype)
                fake_gradient.append(update.detach().clone())

            if any(not torch.isfinite(g).all().item() for g in fake_gradient):
                guarded = True
                self._restore_state(self.generator, generator_safe_state)
                fake_gradient = self._zero_fake_gradient(model, device)

        # Per professor's corrected flow: next local/student starts from the
        # current received global model, not from other clients' hidden updates.
        self.local_state = copy.deepcopy(before_state)

        update_norm = torch.sqrt(sum((g.float() ** 2).sum() for g in fake_gradient)).item()
        clipped = False
        if (not guarded) and self.max_update_norm > 0 and update_norm > self.max_update_norm:
            scale = self.max_update_norm / max(update_norm, 1e-12)
            fake_gradient = [(g * scale).detach().clone() for g in fake_gradient]
            update_norm = torch.sqrt(sum((g.float() ** 2).sum() for g in fake_gradient)).item()
            clipped = True

        reset_text = ""
        if guarded:
            self.guard_streak += 1
            if self._maybe_reset_generator_after_guard():
                reset_text = " reset_generator=1"
        else:
            self.guard_streak = 0

        gen_loss_text = "nan_guard" if gen_loss is None else f"{gen_loss:.6f}"
        kd_loss_text = "nan_guard" if kd_loss is None else f"{kd_loss:.6f}"
        guard_text = " guard=zero" if guarded else ""
        clip_text = f" clipped=max_update_norm({self.max_update_norm:.6f})" if clipped else ""
        _log_fallback(
            "DKFR-LITE",
            round_num,
            "data_free_kd",
            "teacher_current_global_student_previous_global",
            extra=f"generator={self.generator_kind} g_loss={gen_loss_text} kd_loss={kd_loss_text} update_norm={update_norm:.6f}{guard_text}{clip_text}{reset_text}",
        )
        return fake_gradient

    def record_global_gradient(self, global_gradient):
        self.global_gradient_history.append([g.detach().clone().to(g.device) for g in global_gradient])


class SyntheticReplacementAttack:
    """
    Lee-style synthetic local-data replacement baseline.

    The attacker builds a fixed synthetic shard once per client and then trains
    normally on that shard in fake rounds. MNIST and IMDB use same-label MixUp in
    the current model input space; tabular/relational data use light feature
    perturbation as a simple modality-specific analogue.
    """
    def __init__(
        self,
        dataset,
        mixup_alpha=0.4,
        noise_std=0.02,
        tabular_noise_std=0.05,
        tabular_flip_prob=0.02,
        seed=0,
    ):
        self.dataset = str(dataset).lower()
        self.mixup_alpha = float(mixup_alpha)
        self.noise_std = float(noise_std)
        self.tabular_noise_std = float(tabular_noise_std)
        self.tabular_flip_prob = float(tabular_flip_prob)
        self.seed = int(seed)
        self._cache = {}
        self.global_gradient_history = []

    def get_synthetic_data(self, client_id, X, y):
        key = (int(client_id), tuple(X.shape), str(X.device), str(X.dtype))
        if key not in self._cache:
            self._cache[key] = self._synthesize(X, y, int(client_id))
        return self._cache[key]

    def record_global_gradient(self, global_gradient):
        self.global_gradient_history.append([g.detach().clone().to(g.device) for g in global_gradient])

    def _make_generator(self, X, client_id):
        gen = torch.Generator(device=X.device)
        gen.manual_seed(self.seed + 1009 * (client_id + 1))
        return gen

    def _synthesize(self, X, y, client_id):
        if X.numel() == 0:
            return X.detach().clone(), y.detach().clone()

        if self.dataset in {"mnist", "imdb"}:
            X_fake = self._same_label_mixup(X, y, client_id)
            if self.dataset == "mnist":
                mnist_min = (0.0 - 0.1307) / 0.3081
                mnist_max = (1.0 - 0.1307) / 0.3081
                X_fake = X_fake.clamp(mnist_min, mnist_max)
        else:
            X_fake = self._tabular_perturb(X, client_id)

        return X_fake.detach(), y.detach().clone()

    def _same_label_mixup(self, X, y, client_id):
        gen = self._make_generator(X, client_id)
        X_base = X.detach().clone().float()
        y_flat = y.detach().view(-1)
        X_fake = X_base.clone()

        for label in torch.unique(y_flat).tolist():
            idx = torch.nonzero(y_flat == label, as_tuple=False).view(-1)
            if idx.numel() < 2:
                continue
            perm = idx[torch.randperm(idx.numel(), generator=gen, device=X.device)]
            if self.mixup_alpha > 0:
                beta = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
                lam = beta.sample((idx.numel(),)).to(device=X.device, dtype=X_base.dtype)
            else:
                lam = torch.full((idx.numel(),), 0.5, device=X.device, dtype=X_base.dtype)
            lam = lam.view(-1, *([1] * (X_base.dim() - 1)))
            X_fake[idx] = lam * X_base[idx] + (1.0 - lam) * X_base[perm]

        if self.noise_std > 0:
            scale = X_base.detach().std(dim=0, keepdim=True).clamp_min(1e-6)
            noise = torch.randn(X_fake.shape, generator=gen, device=X.device, dtype=X_base.dtype)
            X_fake = X_fake + self.noise_std * scale * noise

        return X_fake.to(dtype=X.dtype)

    def _tabular_perturb(self, X, client_id):
        gen = self._make_generator(X, client_id)
        X_base = X.detach().clone().float()
        X_fake = X_base.clone()

        rounded = torch.isclose(X_base, X_base.round(), atol=1e-6)
        binary_like = rounded & ((X_base.round() == 0) | (X_base.round() == 1))
        binary_cols = binary_like.all(dim=0) if X_base.dim() == 2 else torch.zeros(0, dtype=torch.bool, device=X.device)

        if X_base.dim() == 2 and binary_cols.numel() == X_base.shape[1] and binary_cols.any():
            mask = torch.rand((X_base.shape[0], int(binary_cols.sum().item())), generator=gen, device=X.device) < self.tabular_flip_prob
            cols = torch.nonzero(binary_cols, as_tuple=False).view(-1)
            selected = X_fake[:, cols]
            X_fake[:, cols] = torch.where(mask, 1.0 - selected, selected)

        cont_cols = ~binary_cols if X_base.dim() == 2 and binary_cols.numel() == X_base.shape[1] else None
        if cont_cols is None or cont_cols.any():
            if cont_cols is None:
                scale = X_base.std().clamp_min(1e-6)
                noise = torch.randn(X_fake.shape, generator=gen, device=X.device, dtype=X_base.dtype)
                X_fake = X_fake + self.tabular_noise_std * scale * noise
            else:
                cols = torch.nonzero(cont_cols, as_tuple=False).view(-1)
                scale = X_base[:, cols].std(dim=0, keepdim=True).clamp_min(1e-6)
                noise = torch.randn((X_base.shape[0], cols.numel()), generator=gen, device=X.device, dtype=X_base.dtype)
                X_fake[:, cols] = X_fake[:, cols] + self.tabular_noise_std * scale * noise

        return X_fake.to(dtype=X.dtype)


# Attacking Method 1: Lin et al. random weights / update attack.
class RandomAttack:
    """Coordinate-wise ``U[-R, R]`` fake update.

    The runner aggregates model increments, rather than absolute model
    parameters.  Lin et al.'s bounded random value is therefore returned
    directly as an increment.  ``radius`` is intentionally explicit: the
    paper leaves R to the attacker's prior scale knowledge.
    """

    def __init__(self, radius):
        if radius is None or float(radius) <= 0.0:
            raise ValueError("RandomAttack requires a positive Lin radius R")
        self.radius = float(radius)
        self.global_gradient_history = []

    def get_fake_gradient(self, round_num, device, model):
        _log_fallback(
            "LIN_RANDOM",
            round_num,
            "uniform",
            "always_random",
            extra=f"R={self.radius:.6e}",
        )
        return [
            torch.empty_like(param, device=device).uniform_(-self.radius, self.radius)
            for param in model.parameters()
        ]

    def record_global_gradient(self, global_gradient):
        pass

# Project baseline: replay recent global aggregate increments.
class ReplayAttack:
    """History-replay baseline; this is not Fraboni plain free-riding."""

    def __init__(self, k, start_attack_round):
        self.k = int(k)
        self.start_attack_round = int(start_attack_round)
        self.global_gradient_history = []

    def get_fake_gradient(self, round_num, device, model):
        if len(self.global_gradient_history) == 0:
            _log_fallback("REPLAY", round_num, "zero", "no_history")
            return [torch.zeros_like(param, device=device) for param in model.parameters()]
        used = self.global_gradient_history[-min(len(self.global_gradient_history), self.k):]
        denom = float(len(used))
        _log_fallback("REPLAY", round_num, "avg", "history_average", extra=f"used={len(used)}")
        return [sum(t[i] for t in used) / denom for i in range(len(self.global_gradient_history[0]))]

    def record_global_gradient(self, global_gradient):
        self.global_gradient_history.append([g.detach().clone().to(g.device) for g in global_gradient])


# Compatibility only for direct imports from the isolated module.  The updated
# runner exposes the unambiguous CLI name ``replay``.
GradientReplayAttack = ReplayAttack


class FraboniDisguisedAttack:
    """Fraboni et al.'s disguised free-rider in a delta-update pipeline.

    The paper sends ``theta_global + epsilon``.  This repository sends
    ``theta_client - theta_global`` and the server adds the resulting update,
    so the exactly corresponding packet is ``epsilon``.  No global parameter
    tensor is added here; doing so would mix absolute-parameter and delta
    units.

    ``fixed`` uses an explicitly supplied, externally calibrated sigma.
    ``history_rms`` replaces sigma after the first observed server update with
    a per-layer RMS proxy.  The latter is a documented project adaptation, not
    the paper's client-SGD calibration.
    """

    def __init__(
        self,
        noise_std,
        calibration_mode="history_rms",
        calibration_window=5,
        min_noise_std=1e-12,
    ):
        if noise_std is None or float(noise_std) <= 0.0:
            raise ValueError("FraboniDisguisedAttack requires a positive noise_std")
        if calibration_mode not in {"fixed", "history_rms"}:
            raise ValueError(
                "calibration_mode must be 'fixed' or 'history_rms', "
                f"got {calibration_mode!r}"
            )
        if int(calibration_window) < 1:
            raise ValueError("calibration_window must be at least 1")

        self.noise_std = float(noise_std)
        self.calibration_mode = str(calibration_mode)
        self.calibration_window = int(calibration_window)
        self.min_noise_std = float(min_noise_std)
        self.global_gradient_history = []
        self._layer_rms_history = []

    def _layer_noise_std(self, layer_index):
        if self.calibration_mode == "fixed" or not self.global_gradient_history:
            return self.noise_std

        used = self._layer_rms_history[-self.calibration_window:]
        rms_values = np.asarray([layer_rms[layer_index] for layer_rms in used], dtype=np.float64)
        history_rms = float(np.sqrt(np.mean(rms_values * rms_values)))
        return max(history_rms, self.min_noise_std)

    def get_fake_gradient(self, round_num, device, model):
        stds = []
        fake_gradient = []
        for layer_index, parameter in enumerate(model.parameters()):
            sigma = self._layer_noise_std(layer_index)
            stds.append(sigma)
            fake_gradient.append(torch.randn_like(parameter, device=device) * sigma)

        summary_sigma = float(np.mean(stds)) if stds else 0.0
        print(
            "[FRABONI_DISGUISED] "
            f"round={round_num} mode={self.calibration_mode} "
            f"mean_sigma={summary_sigma:.6e}",
            flush=True,
        )
        return fake_gradient

    def record_global_gradient(self, global_gradient):
        # Only retain per-layer scalars.  Keeping five full MNIST/IMDB update
        # tensors (and stacking them during calibration) would defeat the
        # layerwise memory-engineering path.
        layer_rms = [
            float(torch.sqrt(torch.mean(gradient.detach().float() ** 2)).cpu())
            for gradient in global_gradient
        ]
        self._layer_rms_history.append(layer_rms)
        self.global_gradient_history.append(None)


class ScaledDeltaAttack:
    """
    Scaled-delta free-rider attack from the advanced free-rider family.

    It predicts the next aggregate update by rescaling the most recent global
    update with the observed norm decay between the last two updates.
    """
    def __init__(
        self,
        random_round=2,
        lambda_h=0.9,
        hessian_provider=None,
        eps=1e-12,
        max_scale=10.0,
    ):
        self.random_round = int(random_round)
        self.lambda_h = float(lambda_h)
        self.hessian_provider = hessian_provider
        self.eps = float(eps)
        self.max_scale = None if max_scale is None else float(max_scale)
        self.global_gradient_history = []

    def record_global_gradient(self, global_gradient):
        self.global_gradient_history.append([g.detach().clone().to(g.device) for g in global_gradient])

    def _flatten_gradient(self, gradient, device):
        return torch.cat([
            g.detach().to(device=device).reshape(-1).float()
            for g in gradient
        ])

    def _unflatten_to_model(self, flat_gradient, model, device):
        fake_gradient, start = [], 0
        for param in model.parameters():
            n = param.numel()
            fake_gradient.append(
                flat_gradient[start:start + n].view_as(param).to(device=device, dtype=param.dtype)
            )
            start += n
        return fake_gradient

    def _avg_fallback(self, device, model):
        if len(self.global_gradient_history) == 0:
            return [torch.randn_like(param, device=device) for param in model.parameters()]
        flat_history = [self._flatten_gradient(g, device) for g in self.global_gradient_history]
        avg_vec = sum(flat_history) / float(len(flat_history))
        return self._unflatten_to_model(avg_vec, model, device)

    def _hessian_fuse(self, predicted_gradient, model, device):
        provider = self.hessian_provider or (
            lambda m, d: [torch.zeros_like(p, device=d) for p in m.parameters()]
        )
        h_term = provider(model, device)
        return [
            self.lambda_h * pg + (1.0 - self.lambda_h) * hh
            for pg, hh in zip(predicted_gradient, h_term)
        ]

    def _scaled_delta_flat(self, device):
        prev = self._flatten_gradient(self.global_gradient_history[-1], device)
        prev_prev = self._flatten_gradient(self.global_gradient_history[-2], device)
        scale = torch.linalg.vector_norm(prev) / (torch.linalg.vector_norm(prev_prev) + self.eps)
        if self.max_scale is not None:
            scale = torch.clamp(scale, min=0.0, max=self.max_scale)
        return scale * prev, float(scale.detach().cpu())

    def get_fake_gradient(self, round_num, device, model):
        if len(self.global_gradient_history) < 2:
            _log_fallback("SCALED_DELTA", round_num, "avg", "insufficient_history")
            return self._avg_fallback(device, model)

        if round_num < self.random_round:
            _log_fallback("SCALED_DELTA", round_num, "avg", "warmup")
            return self._avg_fallback(device, model)

        fake_flat, scale = self._scaled_delta_flat(device)
        print(f"[SCALED_DELTA] round={round_num} scale={scale:.6e}")
        return self._hessian_fuse(self._unflatten_to_model(fake_flat, model, device), model, device)


class AdvancedScaledDeltaAttack(ScaledDeltaAttack):
    """
    Advanced scaled-delta attack with geometry-calibrated orthogonal noise.

    The scaled-delta vector approximates the next average update. The added
    orthogonal noise simulates client-specific update deviation so the forged
    update is less trivially separated by cosine, L2-norm, or STD features.
    """
    def __init__(
        self,
        num_clients,
        random_round=2,
        c_const=1.0,
        lambda_est=None,
        noise_fraction=1.0,
        max_noise_ratio=5.0,
        lambda_h=0.9,
        hessian_provider=None,
        eps=1e-12,
        max_scale=10.0,
    ):
        super().__init__(
            random_round=random_round,
            lambda_h=lambda_h,
            hessian_provider=hessian_provider,
            eps=eps,
            max_scale=max_scale,
        )
        self.num_clients = max(1, int(num_clients))
        self.c_const = float(c_const)
        self.lambda_est = None if lambda_est is None else float(lambda_est)
        self.noise_fraction = float(noise_fraction)
        self.max_noise_ratio = None if max_noise_ratio is None else float(max_noise_ratio)

    def _estimate_lambda(self, device):
        if self.lambda_est is not None:
            return max(0.0, self.lambda_est)
        if len(self.global_gradient_history) < 2:
            return 0.0
        first = self._flatten_gradient(self.global_gradient_history[0], device)
        latest = self._flatten_gradient(self.global_gradient_history[-1], device)
        first_norm = float(torch.linalg.vector_norm(first).detach().cpu())
        latest_norm = float(torch.linalg.vector_norm(latest).detach().cpu())
        if first_norm <= self.eps or latest_norm <= self.eps:
            return 0.0
        steps = max(1, len(self.global_gradient_history) - 1)
        # Update norms usually decay as exp(-lambda * t).
        return max(0.0, -np.log(latest_norm / (first_norm + self.eps)) / float(steps))

    def _estimate_cosine(self, round_num, device):
        lam = self._estimate_lambda(device)
        c2 = max(self.c_const * self.c_const, self.eps)
        exponent = float(np.clip(2.0 * lam * max(0, int(round_num)), -50.0, 50.0))
        cos_est = c2 / (c2 + float(np.exp(exponent)))
        return float(np.clip(cos_est, 0.0, 1.0)), lam

    def _orthogonal_noise(self, base, target_norm):
        noise = torch.randn_like(base)
        if 0.0 < self.noise_fraction < 1.0:
            mask = (torch.rand_like(base) < self.noise_fraction).float()
            noise = noise * mask

        base_norm_sq = torch.dot(base, base) + self.eps
        noise = noise - (torch.dot(noise, base) / base_norm_sq) * base
        noise_norm = torch.linalg.vector_norm(noise)
        if float(noise_norm.detach().cpu()) <= self.eps:
            noise = torch.randn_like(base)
            noise = noise - (torch.dot(noise, base) / base_norm_sq) * base
            noise_norm = torch.linalg.vector_norm(noise)
        return noise * (target_norm / (noise_norm + self.eps))

    def get_fake_gradient(self, round_num, device, model):
        if len(self.global_gradient_history) < 2:
            _log_fallback("ADVANCED_SCALED_DELTA", round_num, "avg", "insufficient_history")
            return self._avg_fallback(device, model)

        if round_num < self.random_round:
            _log_fallback("ADVANCED_SCALED_DELTA", round_num, "avg", "warmup")
            return self._avg_fallback(device, model)

        base_flat, scale = self._scaled_delta_flat(device)
        base_norm = torch.linalg.vector_norm(base_flat)
        if float(base_norm.detach().cpu()) <= self.eps:
            fake_flat = base_flat
            noise_ratio = 0.0
            cos_est, lam = self._estimate_cosine(round_num, device)
        else:
            cos_est, lam = self._estimate_cosine(round_num, device)
            n = float(self.num_clients)
            denom = n + (n * n - n) * cos_est
            r_sq = (n * n) / max(denom, self.eps)
            noise_ratio = float(np.sqrt(max(r_sq - 1.0, 0.0)))
            if self.max_noise_ratio is not None:
                noise_ratio = min(noise_ratio, self.max_noise_ratio)
            phi_norm = base_norm * noise_ratio
            fake_flat = base_flat + self._orthogonal_noise(base_flat, phi_norm)

        print(
            "[ADVANCED_SCALED_DELTA] "
            f"round={round_num} scale={scale:.6e} "
            f"lambda={lam:.6e} cos_est={cos_est:.6e} noise_ratio={noise_ratio:.6e}"
        )
        return self._hessian_fuse(self._unflatten_to_model(fake_flat, model, device), model, device)


class WangLocalDeviationAttack:
    """Public-warm-start Adam attack with a local/global-deviation proxy.

    Wang et al. explicitly trains the initial model on public data before the
    first upload.  The paper does not operationalise its later data-free
    gradient ``g_t``.  For this project's delta-update API, the confirmed
    Wang-inspired adaptation therefore stores the previous fake local model
    ``F_(t-1)`` and uses ``F_(t-1) - G_t`` as the observable local-parameter
    deviation.  Adam updates that deviation and the attacker submits
    ``F_t - G_t``.  It never uses an attacker's private shard on fake rounds.

    The runner must call :meth:`initialize_public_warm_start` at round 0 after
    fitting a copy of the received global model on the disjoint public pool.
    This class deliberately omits history-norm matching, clipping and Hessian
    fusion: those are not part of the Wang update rule and would make the
    paper-comparison path a hybrid method.
    """

    def __init__(self, adam_lr=8e-3, beta1=0.9, beta2=0.999, adam_eps=1e-8):
        if float(adam_lr) <= 0.0:
            raise ValueError("adam_lr must be positive")
        if not 0.0 <= float(beta1) < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= float(beta2) < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if float(adam_eps) <= 0.0:
            raise ValueError("adam_eps must be positive")

        self.adam_lr = float(adam_lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.adam_eps = float(adam_eps)

        self._fake_local_parameters = None
        self._m = None
        self._v = None
        self._t = 0
        self.last_deviation_norm = None
        # Kept only for the common attacker interface / runner diagnostics.  We
        # intentionally store markers instead of full GPU gradients because
        # this method never consumes aggregate-update history.
        self.global_gradient_history = []

    @staticmethod
    def _parameter_list(model):
        return [parameter.detach() for parameter in model.parameters()]

    @staticmethod
    def _validate_compatible_parameters(reference, candidate, tag):
        if len(reference) != len(candidate):
            raise ValueError(
                f"{tag}: parameter count mismatch {len(candidate)} != {len(reference)}"
            )
        for index, (ref, value) in enumerate(zip(reference, candidate)):
            if ref.shape != value.shape:
                raise ValueError(
                    f"{tag}: parameter {index} shape mismatch {tuple(value.shape)} != {tuple(ref.shape)}"
                )

    def initialize_public_warm_start(self, global_model, public_local_model):
        """Save the public-trained local state and return its model delta."""
        global_parameters = self._parameter_list(global_model)
        public_parameters = self._parameter_list(public_local_model)
        self._validate_compatible_parameters(
            global_parameters,
            public_parameters,
            "public warm start",
        )

        self._fake_local_parameters = [
            parameter.detach().clone() for parameter in public_parameters
        ]
        self._m = [torch.zeros_like(parameter) for parameter in public_parameters]
        self._v = [torch.zeros_like(parameter) for parameter in public_parameters]
        self._t = 0

        delta = [
            local_parameter - global_parameter.to(
                device=local_parameter.device,
                dtype=local_parameter.dtype,
            )
            for local_parameter, global_parameter in zip(
                self._fake_local_parameters,
                global_parameters,
            )
        ]
        self.last_deviation_norm = float(
            torch.sqrt(sum(torch.sum(value.float() * value.float()) for value in delta)).detach().cpu()
        )
        print(
            "[WANG_LOCAL_DEVIATION] "
            f"round=0 source=in_domain_held_out_public_pool "
            f"warm_start_delta_norm={self.last_deviation_norm:.6e}",
            flush=True,
        )
        return delta

    def get_fake_gradient(self, round_num, device, model):
        if self._fake_local_parameters is None or self._m is None or self._v is None:
            raise RuntimeError(
                "WangLocalDeviationAttack requires initialize_public_warm_start at round 0 "
                "before later fake updates."
            )

        global_parameters = self._parameter_list(model)
        self._validate_compatible_parameters(
            global_parameters,
            self._fake_local_parameters,
            "local/global deviation",
        )

        deviations = [
            local_parameter.to(device=device, dtype=global_parameter.dtype)
            - global_parameter.to(device=device, dtype=global_parameter.dtype)
            for local_parameter, global_parameter in zip(
                self._fake_local_parameters,
                global_parameters,
            )
        ]
        self._t += 1
        next_m, next_v = [], []
        next_fake_local_parameters, submitted_delta = [], []
        for local_parameter, global_parameter, deviation, previous_m, previous_v in zip(
            self._fake_local_parameters,
            global_parameters,
            deviations,
            self._m,
            self._v,
        ):
            previous_m = previous_m.to(device=device, dtype=deviation.dtype)
            previous_v = previous_v.to(device=device, dtype=deviation.dtype)
            moment = self.beta1 * previous_m + (1.0 - self.beta1) * deviation
            variance = self.beta2 * previous_v + (1.0 - self.beta2) * (deviation * deviation)
            moment_hat = moment / (1.0 - self.beta1 ** self._t)
            variance_hat = variance / (1.0 - self.beta2 ** self._t)
            # Algorithm-2 is F_t <- F_(t-1) - Adam(g_t).  The packet in this
            # repository is then the updated local model relative to the
            # currently received global model: F_t - G_t.
            local_step = -self.adam_lr * moment_hat / (torch.sqrt(variance_hat) + self.adam_eps)
            next_local = local_parameter.detach().to(
                device=device,
                dtype=local_step.dtype,
            ).clone() + local_step
            delta = next_local - global_parameter.detach().to(
                device=device,
                dtype=local_step.dtype,
            )
            next_m.append(moment.detach().clone())
            next_v.append(variance.detach().clone())
            next_fake_local_parameters.append(next_local.detach().clone())
            submitted_delta.append(delta.detach().clone())

        if not all(torch.isfinite(update).all() for update in submitted_delta):
            raise FloatingPointError("Wang local-deviation Adam produced a non-finite fake update")

        self._m = next_m
        self._v = next_v
        self._fake_local_parameters = next_fake_local_parameters
        self.last_deviation_norm = float(
            torch.sqrt(sum(torch.sum(value.float() * value.float()) for value in deviations)).detach().cpu()
        )
        update_norm = float(
            torch.sqrt(sum(torch.sum(value.float() * value.float()) for value in submitted_delta)).detach().cpu()
        )
        print(
            "[WANG_LOCAL_DEVIATION] "
            f"round={round_num} t={self._t} "
            f"deviation_norm={self.last_deviation_norm:.6e} "
            f"update_norm={update_norm:.6e} lr={self.adam_lr:.6e}",
            flush=True,
        )
        return submitted_delta

    def record_global_gradient(self, global_gradient):
        # The method uses the received global *parameters* through ``model``;
        # keep only an O(rounds) marker to preserve the shared attacker API.
        self.global_gradient_history.append(None)


# The isolated module keeps this alias for code importing the old class name,
# but FLCE_updated exposes the semantically accurate ``adam_local_deviation``
# method name.
AdamUpdatePredictionAttack = WangLocalDeviationAttack

# Attacking Method 3: Offline ARIMA
class OfflineARIMAAttack:
    def __init__(self, k, random_round, model_params, var_window=3, var_threshold=5e-3,
                 lambda_h=0.9, hessian_provider=None):
        self.k = int(k)
        self.random_round = int(random_round)
        self.var_window = int(var_window)
        self.var_threshold = float(var_threshold)
        self.global_gradient_history = []

        self.param_shapes = [p.shape for p in model_params]
        self.param_sizes = [p.numel() for p in model_params]
        self.param_indices = self._build_param_indices()

        self.lambda_h = float(lambda_h)
        self.hessian_provider = hessian_provider

        self.arima_started = False
        self.arima_start_round = None
        self.offline_model_trained = False
        self.offline_model = {}
        self._fixed_origin_round = None
        self._fit_stats = {"success": 0, "constant": 0, "failed": 0}
        self.last_source = "uninitialized"

    def _build_param_indices(self):
        idx = []
        s = 0
        for size in self.param_sizes:
            e = s + size
            idx.append((s, e))
            s = e
        return idx

    def record_gradient_history(self, flat_gradient):
        self.global_gradient_history.append(flat_gradient)

    def record_global_gradient(self, global_gradient):
        self.record_gradient_history(_flatten_grad_list_to_np(global_gradient))

    def _avg_fallback(self, device, model):
        avg_vec = _avg_history_np(self.global_gradient_history, self.k)
        avg_vec = torch.as_tensor(avg_vec, device=device)
        fake_gradient, start = [], 0
        for p in model.parameters():
            n = p.numel()
            fake_gradient.append(avg_vec[start:start + n].view_as(p).to(dtype=p.dtype))
            start += n
        return fake_gradient

    def _hessian_fuse(self, predicted_gradient, model, device):
        provider = self.hessian_provider or (lambda m, d: [torch.zeros_like(p, device=d) for p in m.parameters()])
        h_term = provider(model, device)
        return [self.lambda_h * pg + (1.0 - self.lambda_h) * hh for pg, hh in zip(predicted_gradient, h_term)]

    def check_arima_ready(self):
        needed = self.k + self.var_window
        if len(self.global_gradient_history) < needed:
            return False
        var_list = []
        start = 0
        for size, shape in zip(self.param_sizes, self.param_shapes):
            series = []
            for round_grad in self.global_gradient_history[-(self.k + self.var_window):]:
                param_grad = round_grad[start:start + size].reshape(shape)
                series.append(param_grad)
            start += size
            window_vars = []
            # ``k + var_window`` observations form ``var_window + 1``
            # overlapping windows of length ``k``.  Their adjacent
            # differences therefore number exactly ``var_window`` and, most
            # importantly, include the newest observation.
            for i in range(self.var_window + 1):
                window = series[i:i + self.k]
                window_vars.append(np.var(np.stack(window)))
            var_list.append(window_vars)
        var_diff = np.abs(np.diff(np.mean(var_list, axis=0)))
        print(f"[ARIMA][OFFLINE] var diff: {var_diff}")
        return bool(np.all(var_diff < self.var_threshold))

    def train_offline_arima_model(self, fit_round=None):
        print("[ARIMA][OFFLINE] Starting offline auto_arima training...")
        self.offline_model = {}
        successful_dims = 0
        constant_dims = 0
        failed_dims = 0
        for param_index, size in enumerate(self.param_sizes):
            st, _ = self.param_indices[param_index]
            param_series_list = []
            for i in range(size):
                series = np.array([g[st + i] for g in self.global_gradient_history], dtype=np.float32)
                if _is_constant_series(series):
                    param_series_list.append(_constant_series_model(series))
                    constant_dims += 1
                    continue
                try:
                    m = auto_arima(
                        series,
                        start_p=0, start_q=0, max_p=2, max_q=2,
                        d=None, max_d=1, seasonal=False, stationary=False,
                        error_action='ignore', suppress_warnings=True, stepwise=True, maxiter=20
                    )
                    param_series_list.append(m)
                    successful_dims += 1
                except Exception:
                    param_series_list.append(None)
                    failed_dims += 1
            self.offline_model[param_index] = param_series_list
        self.offline_model_trained = True
        self._fixed_origin_round = (
            int(fit_round) if fit_round is not None else len(self.global_gradient_history)
        )
        self._fit_stats = {
            "success": successful_dims,
            "constant": constant_dims,
            "failed": failed_dims,
        }
        print(
            "[ARIMA][OFFLINE][FIT] "
            f"origin_round={self._fixed_origin_round} "
            f"observations={len(self.global_gradient_history)} "
            f"success={successful_dims} constant={constant_dims} failed={failed_dims}"
        )

    def get_fake_gradient(self, round_num, device, model):
        if len(self.global_gradient_history) == 0:
            self.last_source = "random_no_history"
            _log_fallback("ARIMA][OFFLINE", round_num, "random", "no_history")
            return [torch.randn_like(p, device=device) for p in model.parameters()]

        # A frozen offline model must not inspect readiness or fit during warmup.
        if round_num < self.random_round:
            self.last_source = "avg_warmup"
            _log_fallback("ARIMA][OFFLINE", round_num, "avg", "warmup", extra=f"used={min(len(self.global_gradient_history), self.k)}")
            return self._avg_fallback(device, model)

        if (not self.arima_started) and self.check_arima_ready():
            self.arima_started = True
            self.arima_start_round = int(round_num)
            self.train_offline_arima_model(fit_round=round_num)
            print(f"[ARIMA][OFFLINE] round={round_num} SWITCHED TO AUTO-ARIMA")

        if self.arima_started and self.offline_model_trained:
            origin_round = (
                self._fixed_origin_round
                if self._fixed_origin_round is not None
                else self.arima_start_round
            )
            horizon = max(1, int(round_num) - int(origin_round) + 1)
            predicted_gradient = []
            fallback_vec = _avg_history_np(self.global_gradient_history, self.k)
            forecast_success = 0
            forecast_constant = 0
            forecast_fallback = 0
            forecast_failed = 0
            for param_index, param in enumerate(model.parameters()):
                predicted_param = torch.zeros_like(param, device=device)
                model_list = self.offline_model.get(param_index, [])
                for i in range(param.numel()):
                    m = model_list[i] if i < len(model_list) else None
                    if isinstance(m, tuple) and len(m) >= 2 and m[0] == "constant":
                        predicted_param.view(-1)[i] = torch.as_tensor(m[1], device=device, dtype=param.dtype)
                        forecast_constant += 1
                    elif m is not None:
                        try:
                            forecast = m.predict(n_periods=horizon)
                            predicted_param.view(-1)[i] = torch.as_tensor(forecast[-1], device=device, dtype=param.dtype)
                            forecast_success += 1
                        except Exception:
                            flat_index = self.param_indices[param_index][0] + i
                            predicted_param.view(-1)[i] = torch.as_tensor(
                                fallback_vec[flat_index], device=device, dtype=param.dtype
                            )
                            forecast_fallback += 1
                            forecast_failed += 1
                    else:
                        flat_index = self.param_indices[param_index][0] + i
                        predicted_param.view(-1)[i] = torch.as_tensor(
                            fallback_vec[flat_index], device=device, dtype=param.dtype
                        )
                        forecast_fallback += 1
                        forecast_failed += 1
                predicted_gradient.append(predicted_param)
            print(
                "[ARIMA][OFFLINE][FORECAST] "
                f"round={round_num} origin_round={origin_round} horizon={horizon} "
                f"success={forecast_success} constant={forecast_constant} "
                f"fallback={forecast_fallback} failed={forecast_failed}"
            )
            self.last_source = "arima_fixed_origin"
            return self._hessian_fuse(predicted_gradient, model, device)

        self.last_source = "avg_not_ready"
        _log_fallback("ARIMA][OFFLINE", round_num, "avg", "not_ready", extra=f"used={min(len(self.global_gradient_history), self.k)}")
        return self._avg_fallback(device, model)

# Attacking Method 4: Adaptive ARIMA
class AdaptiveARIMAAttack:
    supports_round_aware_recording = True

    def __init__(self, k, random_round, model_params,
                 var_window=3, var_threshold=5e-3,
                 lambda_h=0.9, hessian_provider=None,
                 err_ema_beta=0.9, err_threshold=5e-3,
                 min_rounds_after_start=2, retrain_interval=10,
                 trim_history=True):
        self.k = int(k)
        self.random_round = int(random_round)
        self.var_window = int(var_window)
        self.var_threshold = float(var_threshold)
        self.global_gradient_history = []

        self.param_shapes = [p.shape for p in model_params]
        self.param_sizes = [p.numel() for p in model_params]
        self.param_indices = self._build_param_indices()

        self.lambda_h = float(lambda_h)
        self.hessian_provider = hessian_provider

        self.arima_started = False
        self.arima_start_round = None
        self.offline_model_trained = False
        self.offline_model = {}

        self.err_ema_beta = float(err_ema_beta)
        self.err_threshold = float(err_threshold)
        self.min_rounds_after_start = int(min_rounds_after_start)
        self.retrain_interval = int(retrain_interval)
        self.trim_history = bool(trim_history)

        self._err_ema = None
        self._since_start = 0
        self._last_pred_flat = None
        self._last_pred_round = None
        self._total_observations = 0
        self._observations_since_reset = None
        self._last_full_refit_observation = None
        self._fixed_origin_round = None
        self._fit_generation = 0
        self._aligned_observations = 0
        self._skipped_alignments = 0
        self._fit_stats = {"success": 0, "constant": 0, "failed": 0}
        self.last_source = "uninitialized"

    def _build_param_indices(self):
        idx = []
        s = 0
        for size in self.param_sizes:
            e = s + size
            idx.append((s, e))
            s = e
        return idx

    def _avg_fallback(self, device, model):
        if len(self.global_gradient_history) == 0:
            return [torch.randn_like(p, device=device) for p in model.parameters()]
        avg_vec = _avg_history_np(self.global_gradient_history, self.k)
        avg_vec = torch.as_tensor(avg_vec, device=device)
        fake_gradient, start = [], 0
        for p in model.parameters():
            n = p.numel()
            fake_gradient.append(avg_vec[start:start + n].view_as(p).to(dtype=p.dtype))
            start += n
        return fake_gradient

    def _hessian_fuse(self, predicted_gradient, model, device):
        provider = self.hessian_provider or (lambda m, d: [torch.zeros_like(p, device=d) for p in m.parameters()])
        h_term = provider(model, device)
        return [self.lambda_h * pg + (1.0 - self.lambda_h) * hh for pg, hh in zip(predicted_gradient, h_term)]

    def check_arima_ready(self):
        needed = self.k + self.var_window
        if len(self.global_gradient_history) < needed:
            return False
        if (
            self._observations_since_reset is not None
            and self._observations_since_reset < needed
        ):
            return False
        var_list = []
        start = 0
        for size, shape in zip(self.param_sizes, self.param_shapes):
            series = []
            for round_grad in self.global_gradient_history[-(self.k + self.var_window):]:
                param_grad = round_grad[start:start + size].reshape(shape)
                series.append(param_grad)
            start += size
            window_vars = []
            # Evaluate every available overlapping window so the latest
            # observation participates in readiness.  With ``k +
            # var_window`` samples this yields ``var_window`` differences.
            for i in range(self.var_window + 1):
                window = series[i:i + self.k]
                window_vars.append(np.var(np.stack(window)))
            var_list.append(window_vars)
        var_diff = np.abs(np.diff(np.mean(var_list, axis=0)))
        print(f"[ARIMA][ADAPT] var diff: {var_diff}")
        return bool(np.all(var_diff < self.var_threshold))

    def train_offline_arima_model(
        self,
        reason="initial",
        history_limit=None,
        fit_round=None,
    ):
        print("[ARIMA][ADAPT] Starting offline auto_arima training...")
        self.offline_model = {}
        fit_history = self.global_gradient_history
        if history_limit is not None:
            fit_history = fit_history[-max(1, int(history_limit)):]
        successful_dims = 0
        constant_dims = 0
        failed_dims = 0
        for param_index, size in enumerate(self.param_sizes):
            st, _ = self.param_indices[param_index]
            param_series_list = []
            for i in range(size):
                series = np.array([g[st + i] for g in fit_history], dtype=np.float32)
                if _is_constant_series(series):
                    constant_model = _constant_series_model(series)
                    param_series_list.append(
                        (constant_model[0], constant_model[1], int(series.size))
                    )
                    constant_dims += 1
                    continue
                try:
                    m = auto_arima(
                        series,
                        start_p=0, start_q=0, max_p=2, max_q=2,
                        d=None, max_d=1, seasonal=False, stationary=False,
                        error_action='ignore', suppress_warnings=True, stepwise=True, maxiter=20
                    )
                    param_series_list.append(m)
                    successful_dims += 1
                except Exception:
                    param_series_list.append(None)
                    failed_dims += 1
            self.offline_model[param_index] = param_series_list
        self.offline_model_trained = True
        self._last_full_refit_observation = self._total_observations
        default_origin = (
            self._total_observations
            if self._total_observations > 0
            else len(self.global_gradient_history)
        )
        self._fixed_origin_round = (
            int(fit_round) if fit_round is not None else int(default_origin)
        )
        self._fit_generation += 1
        self._fit_stats = {
            "success": successful_dims,
            "constant": constant_dims,
            "failed": failed_dims,
        }
        print(
            "[ARIMA][ADAPT][FIT] "
            f"reason={reason} generation={self._fit_generation} "
            f"observation={self._total_observations} "
            f"origin_round={self._fixed_origin_round} "
            f"fit_observations={len(fit_history)} "
            f"success={successful_dims} constant={constant_dims} failed={failed_dims}"
        )

    def _clear_prediction_cache(self):
        self._last_pred_flat = None
        self._last_pred_round = None

    def _reset_arima_state(self):
        self.arima_started = False
        self.arima_start_round = None
        self.offline_model_trained = False
        self.offline_model = {}
        self._clear_prediction_cache()
        self._err_ema = None
        self._since_start = 0
        self._observations_since_reset = 0
        self._last_full_refit_observation = None
        self._fixed_origin_round = None
        if self.trim_history:
            keep = max(self.k + self.var_window, self.k)
            if len(self.global_gradient_history) > keep:
                self.global_gradient_history = self.global_gradient_history[-keep:]
        print(
            "[ARIMA][ADAPT] RESET completed; "
            f"requires {self.k + self.var_window} new observations before readiness."
        )

    def record_gradient_history(self, flat_gradient):
        self.global_gradient_history.append(flat_gradient)
        self._total_observations += 1
        if self._observations_since_reset is not None:
            self._observations_since_reset += 1

    def record_global_gradient(self, global_gradient, round_num=None):
        flat = _flatten_grad_list_to_np(global_gradient)
        self.record_gradient_history(flat)

        try:
            observed_round = None if round_num is None else int(round_num)
        except (TypeError, ValueError):
            observed_round = None

        pending_prediction = (
            self._last_pred_flat is not None and self._last_pred_round is not None
        )
        predicted_round = self._last_pred_round
        alignment = "no_pending_prediction"
        mae = None
        should_reset = False

        if pending_prediction:
            if observed_round is not None and observed_round == int(predicted_round):
                alignment = "aligned"
                self._aligned_observations += 1
                self._since_start += 1
                mae = float(np.mean(np.abs(flat - self._last_pred_flat)))
                if self._err_ema is None:
                    self._err_ema = mae
                else:
                    beta = self.err_ema_beta
                    self._err_ema = beta * self._err_ema + (1.0 - beta) * mae
                should_reset = bool(
                    self._since_start >= self.min_rounds_after_start
                    and self._err_ema > self.err_threshold
                )
            else:
                self._skipped_alignments += 1
                alignment = (
                    "skipped_legacy_round_none"
                    if round_num is None
                    else f"skipped_round_mismatch:{observed_round}!={predicted_round}"
                )

        active_model = self.arima_started and self.offline_model_trained
        periodic_refit = False
        if should_reset:
            print(
                "[ARIMA][ADAPT][OBSERVE] "
                f"round={observed_round} pred_round={predicted_round} alignment={alignment} "
                f"mae={mae:.6e} ema={self._err_ema:.6e} "
                "reset=1 "
                f"aligned_total={self._aligned_observations} "
                f"skipped_total={self._skipped_alignments}"
            )
            self._clear_prediction_cache()
            self._reset_arima_state()
            return

        if (
            active_model
            and self.retrain_interval > 0
            and self._last_full_refit_observation is not None
            and (
                self._total_observations - self._last_full_refit_observation
                >= self.retrain_interval
            )
        ):
            periodic_refit = True
            next_prediction_round = (
                int(observed_round) + 1
                if observed_round is not None
                else int(self._total_observations)
            )
            self.train_offline_arima_model(
                reason=f"periodic@observation={self._total_observations}",
                history_limit=self._observations_since_reset,
                fit_round=next_prediction_round,
            )

        # Every pending prediction is consumed by the next observation, even
        # when the caller omitted/mismatched the round. This prevents stale
        # predictions from being reused on later honest rounds.
        self._clear_prediction_cache()
        print(
            "[ARIMA][ADAPT][OBSERVE] "
            f"round={observed_round} pred_round={predicted_round} alignment={alignment} "
            f"mae={'n/a' if mae is None else f'{mae:.6e}'} "
            f"ema={'n/a' if self._err_ema is None else f'{self._err_ema:.6e}'} "
            f"periodic_refit={int(periodic_refit)} "
            f"new_since_reset={self._observations_since_reset} "
            f"aligned_total={self._aligned_observations} "
            f"skipped_total={self._skipped_alignments}"
        )

    def get_fake_gradient(self, round_num, device, model):
        if len(self.global_gradient_history) == 0:
            self.last_source = "random_no_history"
            _log_fallback("ARIMA][ADAPT", round_num, "random", "no_history")
            return [torch.randn_like(p, device=device) for p in model.parameters()]

        # Readiness/fitting is forbidden during warmup, including after reset.
        if round_num < self.random_round:
            self.last_source = "avg_warmup"
            _log_fallback("ARIMA][ADAPT", round_num, "avg", "warmup", extra=f"used={min(len(self.global_gradient_history), self.k)}")
            return self._avg_fallback(device, model)

        cooldown = self._observations_since_reset
        cooldown_needed = self.k + self.var_window
        if cooldown is not None and cooldown < cooldown_needed:
            self.last_source = "avg_cooldown"
            _log_fallback(
                "ARIMA][ADAPT",
                round_num,
                "avg",
                "cooldown",
                extra=(
                    f"used={min(len(self.global_gradient_history), self.k)} "
                    f"new_since_reset={cooldown}/{cooldown_needed}"
                ),
            )
            return self._avg_fallback(device, model)

        if (not self.arima_started) and self.check_arima_ready():
            self.arima_started = True
            self.arima_start_round = int(round_num)
            self.train_offline_arima_model(
                reason=f"start@round={round_num}",
                history_limit=self._observations_since_reset,
                fit_round=round_num,
            )
            print(f"[ARIMA][ADAPT] round={round_num} SWITCHED TO ADAPTIVE")

        if self.arima_started and self.offline_model_trained:
            if self._last_pred_flat is not None or self._last_pred_round is not None:
                print(
                    "[ARIMA][ADAPT][ALIGN] overwriting an unconsumed prediction "
                    f"pred_round={self._last_pred_round} with round={round_num}"
                )
                self._clear_prediction_cache()
            predicted_gradient = []
            origin_round = (
                self._fixed_origin_round
                if self._fixed_origin_round is not None
                else (
                    self.arima_start_round
                    if self.arima_start_round is not None
                    else int(round_num)
                )
            )
            horizon = max(1, int(round_num) - int(origin_round) + 1)
            fallback_vec = _avg_history_np(self.global_gradient_history, self.k)
            forecast_success = 0
            forecast_constant = 0
            forecast_fallback = 0
            forecast_failed = 0
            for param_index, param in enumerate(model.parameters()):
                predicted_param = torch.zeros_like(param, device=device)
                model_list = self.offline_model.get(param_index, [])
                for i in range(param.numel()):
                    m = model_list[i] if i < len(model_list) else None
                    if isinstance(m, tuple) and len(m) >= 2 and m[0] == "constant":
                        predicted_param.view(-1)[i] = torch.as_tensor(m[1], device=device, dtype=param.dtype)
                        forecast_constant += 1
                    elif m is not None:
                        try:
                            forecast = m.predict(n_periods=horizon)
                            predicted_param.view(-1)[i] = torch.as_tensor(forecast[-1], device=device, dtype=param.dtype)
                            forecast_success += 1
                        except Exception:
                            flat_index = self.param_indices[param_index][0] + i
                            predicted_param.view(-1)[i] = torch.as_tensor(
                                fallback_vec[flat_index], device=device, dtype=param.dtype
                            )
                            forecast_fallback += 1
                            forecast_failed += 1
                    else:
                        flat_index = self.param_indices[param_index][0] + i
                        predicted_param.view(-1)[i] = torch.as_tensor(
                            fallback_vec[flat_index], device=device, dtype=param.dtype
                        )
                        forecast_fallback += 1
                        forecast_failed += 1
                predicted_gradient.append(predicted_param)

            submitted_gradient = self._hessian_fuse(
                predicted_gradient, model, device
            )
            self._last_pred_round = int(round_num)
            self._last_pred_flat = _flatten_grad_list_to_np(
                submitted_gradient
            )
            print(
                "[ARIMA][ADAPT][FORECAST] "
                f"round={round_num} origin_round={origin_round} horizon={horizon} "
                f"success={forecast_success} "
                f"constant={forecast_constant} fallback={forecast_fallback} "
                f"failed={forecast_failed}"
            )
            self.last_source = "arima_adaptive_fixed_origin"
            return submitted_gradient

        extra = f"used={min(len(self.global_gradient_history), self.k)}"
        if cooldown is not None:
            extra += f" new_since_reset={cooldown}/{self.k + self.var_window}"
        self.last_source = "avg_not_ready"
        _log_fallback("ARIMA][ADAPT", round_num, "avg", "not_ready", extra=extra)
        return self._avg_fallback(device, model)

# Attacking Method 5: Offline MOIRAI
class OfflineMoiraiAttack:
    def __init__(self, k, random_round, history_length=5, model_type="moirai",
                 size="small", patch_size="auto",
                 lambda_h=0.9, hessian_provider=None):
        self.history_length = int(history_length)
        self.k = int(k)
        self.model_type = model_type
        self.size = size
        self.patch_size = patch_size
        self.random_round = int(random_round)
        self.global_gradient_history = []

        self.lambda_h = float(lambda_h)
        self.hessian_provider = hessian_provider

        self.param_shapes = None
        self.param_sizes = None
        self.D = None
        self.model = None
        self._predictor = None
        self.last_source = "uninitialized"

        assert self.model_type == "moirai", "Only 'moirai' type is supported."

    def record_gradient_history(self, flat_gradient):
        self.global_gradient_history.append(flat_gradient)

    def record_global_gradient(self, global_gradient):
        self.record_gradient_history(_flatten_grad_list_to_np(global_gradient))

    def _lazy_init(self, model):
        if self.D is not None:
            return
        self.param_shapes = [p.shape for p in model.parameters()]
        self.param_sizes = [p.numel() for p in model.parameters()]
        self.D = int(sum(self.param_sizes))
        self.model = MoiraiForecast(
            module=_load_moirai_module(_resolve_moirai_source(self.size)),
            prediction_length=1,
            context_length=self.history_length,
            patch_size=self.patch_size,
            num_samples=1,
            target_dim=self.D,                                                                                                                                                                                                                                                             
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._predictor = self.model.create_predictor(batch_size=1)

    def _avg_fallback(self, device, model):
        if len(self.global_gradient_history) == 0:
            return [torch.randn_like(p, device=device) for p in model.parameters()]
        avg_vec = _avg_history_np(self.global_gradient_history, self.k)
        avg_vec = torch.as_tensor(avg_vec, device=device)
        fake_gradient, start = [], 0
        for p in model.parameters():
            n = p.numel()
            fake_gradient.append(avg_vec[start:start + n].view_as(p).to(dtype=p.dtype))
            start += n
        return fake_gradient

    def _hessian_fuse(self, predicted_gradient, model, device):
        provider = self.hessian_provider or (lambda m, d: [torch.zeros_like(pp, device=d) for pp in m.parameters()])
        h_term = provider(model, device)
        return [self.lambda_h * pg + (1.0 - self.lambda_h) * hh for pg, hh in zip(predicted_gradient, h_term)]

    def get_fake_gradient(self, round_num, device, model):
        self._lazy_init(model)

        if len(self.global_gradient_history) == 0:
            self.last_source = "random_no_history"
            _log_fallback("Moirai][OFFLINE", round_num, "random", "no_history")
            return [torch.randn_like(p, device=device) for p in model.parameters()]

        if len(self.global_gradient_history) < self.history_length or round_num < self.random_round:
            self.last_source = "avg_warmup_or_short_history"
            _log_fallback("Moirai][OFFLINE", round_num, "avg", "warmup_or_short_history", extra=f"used={min(len(self.global_gradient_history), self.k)}")
            return self._avg_fallback(device, model)

        print(f"[Moirai][OFFLINE] round={round_num} USING MOIRAI PREDICTION")
        gradient_series = np.stack(self.global_gradient_history[-self.history_length:], axis=0)
        input_data = [{
            "start": pd.Period("2000-01-01", freq="D"),
            "target": gradient_series.T,
        }]
        forecast = next(self._predictor.predict(input_data))
        pred_vec = torch.as_tensor(forecast.samples[0, 0], device=device)

        fake_gradient, start = [], 0
        for p in model.parameters():
            n = p.numel()
            fake_gradient.append(pred_vec[start:start + n].view_as(p).to(dtype=p.dtype))
            start += n
        self.last_source = "moirai_offline"
        return self._hessian_fuse(fake_gradient, model, device)

# Attacking Method 6: Online MOIRAI
class OnlineMoiraiAttack:
    supports_round_aware_recording = True

    def __init__(self, k, random_round, history_length=5, model_type="moirai",
                 size="small", patch_size="auto",
                 lambda_h=0.9, hessian_provider=None,
                 online_tuning=True, tune_lr=1e-3, tune_steps=1, tune_start_round=None,
                 residual_clip=5.0, adapt_interval=10, residual_decay=0.9,
                 err_ema_beta=0.9, err_threshold=5e-3, min_rounds_after_start=2,
                 trim_history=True):
        self.history_length = int(history_length)
        self.k = int(k)
        self.model_type = model_type
        self.size = size
        self.patch_size = patch_size
        self.random_round = int(random_round)
        self.global_gradient_history = []
        self.lambda_h = float(lambda_h)
        self.hessian_provider = hessian_provider

        self.param_shapes = None
        self.param_sizes = None
        self.D = None
        self.model = None
        self._predictor = None

        self.online_tuning = bool(online_tuning)
        self.tune_lr = float(tune_lr)
        self.tune_steps = int(max(1, tune_steps))
        self.tune_start_round = self.random_round if tune_start_round is None else int(tune_start_round)
        self.residual_clip = residual_clip
        self.adapt_interval = int(adapt_interval)
        self.residual_decay = float(residual_decay)
        self.trim_history = bool(trim_history)

        self.err_ema_beta = float(err_ema_beta)
        self.err_threshold = float(err_threshold)
        self.min_rounds_after_start = int(min_rounds_after_start)
        self._err_ema = None
        self._since_start = 0

        self.residual = None
        self._last_pred_round = None
        self._last_base_pred_flat = None
        self._last_pred_flat = None
        self._aligned_observations = 0
        self._skipped_alignments = 0
        # Count only residual calibration updates that actually completed.
        # Using the global round here would make decay depend on the lazy
        # attack cadence (and can make some fixed-gap schedules never decay).
        self._residual_update_count = 0
        self.last_source = "uninitialized"

        assert self.model_type == "moirai", "Only 'moirai' type is supported."

    def _lazy_init(self, model):
        if self.D is not None:
            return
        self.param_shapes = [p.shape for p in model.parameters()]
        self.param_sizes = [p.numel() for p in model.parameters()]
        self.D = int(sum(self.param_sizes))
        self.model = MoiraiForecast(
            module=_load_moirai_module(_resolve_moirai_source(self.size)),
            prediction_length=1,
            context_length=self.history_length,
            patch_size=self.patch_size,
            num_samples=1,
            target_dim=self.D,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._predictor = self.model.create_predictor(batch_size=1)
        if self.residual is None:
            self.residual = np.zeros((self.D,), dtype=np.float32)

    def _lazy_init_from_global_grad(self, global_gradient):
        if self.D is not None:
            return
        self.param_shapes = [g.shape for g in global_gradient]
        self.param_sizes = [g.numel() for g in global_gradient]
        self.D = int(sum(self.param_sizes))
        self.model = MoiraiForecast(
            module=_load_moirai_module(_resolve_moirai_source(self.size)),
            prediction_length=1,
            context_length=self.history_length,
            patch_size=self.patch_size,
            num_samples=1,
            target_dim=self.D,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._predictor = self.model.create_predictor(batch_size=1)
        if self.residual is None:
            self.residual = np.zeros((self.D,), dtype=np.float32)

    def _avg_fallback(self, device, model):
        if len(self.global_gradient_history) == 0:
            return [torch.randn_like(p, device=device) for p in model.parameters()]
        avg_vec = _avg_history_np(self.global_gradient_history, self.k)
        avg_vec = torch.as_tensor(avg_vec, device=device)
        fake_gradient, start = [], 0
        for p in model.parameters():
            n = p.numel()
            fake_gradient.append(avg_vec[start:start + n].view_as(p).to(dtype=p.dtype))
            start += n
        return fake_gradient

    def _hessian_fuse(self, predicted_gradient, model, device):
        provider = self.hessian_provider or (lambda m, d: [torch.zeros_like(pp, device=d) for pp in m.parameters()])
        h_term = provider(model, device)
        return [self.lambda_h * pg + (1.0 - self.lambda_h) * hh for pg, hh in zip(predicted_gradient, h_term)]

    def _soft_adjust(self):
        if self.residual is not None:
            self.residual *= self.residual_decay

    def _clear_prediction_cache(self):
        self._last_pred_round = None
        self._last_base_pred_flat = None
        self._last_pred_flat = None

    def _hard_reset(self):
        if self.residual is not None:
            self.residual.fill(0.0)
        self._err_ema = None
        self._since_start = 0
        self._residual_update_count = 0
        self._clear_prediction_cache()
        if self.trim_history:
            keep = max(self.k + self.history_length, self.k)
            if len(self.global_gradient_history) > keep:
                self.global_gradient_history = self.global_gradient_history[-keep:]
        print("[Moirai][OnlineTune] RESET completed residual_updates=0.")

    def record_gradient_history(self, flat_gradient):
        self.global_gradient_history.append(flat_gradient)

    def record_global_gradient(self, global_gradient, round_num=None):
        flat = _flatten_grad_list_to_np(global_gradient)
        self.record_gradient_history(flat)

        pending_prediction = (
            self._last_pred_round is not None
            and self._last_base_pred_flat is not None
            and self._last_pred_flat is not None
        )
        if not pending_prediction:
            # Honest/fallback rounds only extend the observable global history.
            return

        try:
            observed_round = None if round_num is None else int(round_num)
        except (TypeError, ValueError):
            observed_round = None
        predicted_round = int(self._last_pred_round)

        if observed_round is None or observed_round != predicted_round:
            self._skipped_alignments += 1
            reason = (
                "legacy_round_none"
                if round_num is None
                else f"round_mismatch:{observed_round}!={predicted_round}"
            )
            print(
                "[Moirai][OnlineTune][OBSERVE] "
                f"round={observed_round} pred_round={predicted_round} "
                f"alignment=skipped reason={reason}; history_only=1 "
                f"aligned_total={self._aligned_observations} "
                f"skipped_total={self._skipped_alignments}"
            )
            self._clear_prediction_cache()
            return

        self._aligned_observations += 1
        self._since_start += 1
        mae = float(np.mean(np.abs(flat - self._last_pred_flat)))
        if self._err_ema is None:
            self._err_ema = mae
        else:
            beta = self.err_ema_beta
            self._err_ema = beta * self._err_ema + (1.0 - beta) * mae

        if (
            self._since_start >= self.min_rounds_after_start
            and self._err_ema > self.err_threshold
        ):
            print(
                "[Moirai][OnlineTune][OBSERVE] "
                f"round={observed_round} pred_round={predicted_round} alignment=aligned "
                f"mae={mae:.6e} ema={self._err_ema:.6e} reset=1 "
                f"aligned_total={self._aligned_observations} "
                f"skipped_total={self._skipped_alignments}"
            )
            self._hard_reset()
            return

        residual_updated = False
        decay_applied = False
        if (
            self.online_tuning
            and observed_round >= self.tune_start_round
            and len(self.global_gradient_history) >= self.history_length + 1
        ):
            if self.residual is None:
                self.residual = np.zeros((self.D,), dtype=np.float32)

            # Calibrate the exact Hessian-fused submission cached for this
            # round. Never draw a second stochastic Moirai sample here.
            for step in range(self.tune_steps):
                if step == 0:
                    # The first step is exactly the error of the cached final
                    # prediction submitted for this round.
                    err = self._last_pred_flat - flat
                else:
                    err = (
                        self._last_base_pred_flat
                        + self.lambda_h * self.residual
                    ) - flat
                # The residual is added before Hessian fusion, so the actual
                # submitted vector changes by lambda_h * residual.
                self.residual -= self.tune_lr * self.lambda_h * err
                if self.residual_clip is not None:
                    np.clip(
                        self.residual,
                        -float(self.residual_clip),
                        float(self.residual_clip),
                        out=self.residual,
                    )
            residual_updated = True
            self._residual_update_count += 1
            if (
                self.adapt_interval > 0
                and self._residual_update_count % self.adapt_interval == 0
            ):
                # Decay applies to future predictions; it must not change the
                # residual before calibrating the just-observed submission.
                # The interval is measured in successful calibration updates,
                # not global rounds, so lazy schedules have identical meaning.
                self._soft_adjust()
                decay_applied = True

        residual_norm = (
            float(np.linalg.norm(self.residual)) if self.residual is not None else 0.0
        )
        print(
            "[Moirai][OnlineTune][OBSERVE] "
            f"round={observed_round} pred_round={predicted_round} alignment=aligned "
            f"mae={mae:.6e} ema={self._err_ema:.6e} "
            f"residual_updated={int(residual_updated)} "
            f"residual_updates={self._residual_update_count} "
            f"decay_applied={int(decay_applied)} "
            f"residual_norm={residual_norm:.6e} "
            f"aligned_total={self._aligned_observations} "
            f"skipped_total={self._skipped_alignments}"
        )
        self._clear_prediction_cache()

    def get_fake_gradient(self, round_num, device, model):
        self._lazy_init(model)

        if len(self.global_gradient_history) == 0:
            self.last_source = "random_no_history"
            _log_fallback("Moirai][OnlineTune", round_num, "random", "no_history")
            return [torch.randn_like(p, device=device) for p in model.parameters()]

        if len(self.global_gradient_history) < self.history_length or round_num < self.random_round:
            self.last_source = "avg_warmup_or_short_history"
            _log_fallback("Moirai][OnlineTune", round_num, "avg", "warmup_or_short_history", extra=f"used={min(len(self.global_gradient_history), self.k)}")
            return self._avg_fallback(device, model)

        print(f"[Moirai][OnlineTune] round={round_num} USING MOIRAI PREDICTION")
        gradient_series = np.stack(self.global_gradient_history[-self.history_length:], axis=0)
        input_data = [{
            "start": pd.Period("2000-01-01", freq="D"),
            "target": gradient_series.T,
        }]
        forecast = next(self._predictor.predict(input_data))
        base_pred_np = forecast.samples[0, 0].astype(np.float32)
        pred_np = base_pred_np.copy()
        if self.residual is not None:
            pred_np = pred_np + self.residual

        if (
            self._last_pred_round is not None
            or self._last_base_pred_flat is not None
            or self._last_pred_flat is not None
        ):
            print(
                "[Moirai][OnlineTune][ALIGN] overwriting an unconsumed prediction "
                f"pred_round={self._last_pred_round} with round={round_num}"
            )
            self._clear_prediction_cache()
        pred_vec = torch.as_tensor(pred_np, device=device)
        fake_gradient, start = [], 0
        for p in model.parameters():
            n = p.numel()
            fake_gradient.append(pred_vec[start:start + n].view_as(p).to(dtype=p.dtype))
            start += n
        submitted_gradient = self._hessian_fuse(fake_gradient, model, device)
        submitted_flat = _flatten_grad_list_to_np(submitted_gradient)
        self._last_pred_round = int(round_num)
        self._last_pred_flat = submitted_flat.copy()
        self._last_base_pred_flat = (
            submitted_flat - self.lambda_h * self.residual
        ).astype(np.float32, copy=False)
        self.last_source = "moirai_online_residual"
        return submitted_gradient


# Backward compatibility for existing entry scripts.
ARIMAAttack = OfflineARIMAAttack
MoiraiAttack = OfflineMoiraiAttack
