"""Full definition of a GPT Language Model, all of it in this single file.

Based on implementation by Andrej Karpathy, with some modifications.
Copyright (c) 2022 Andrej Karpathy

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias.

    PyTorch doesn't support simply bias=False.
    """

    def __init__(self, ndim: int, bias: int) -> None:
        """Initialize LayerNorm with given dimensionality and bias option."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Apply LayerNorm to the input tensor."""
        return F.layer_norm(
            input_tensor,
            self.weight.shape,
            self.weight,
            self.bias,
            1e-5,
        )


class CausalSelfAttention(nn.Module):
    """Causal self-attention module."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize the causal self-attention module."""
        super().__init__()
        if config.n_embd % config.n_head != 0:
            error = (
                f"n_embd ({config.n_embd}) must be"
                f" divisible by n_head ({config.n_head})"
            )
            raise ValueError(error)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            error = (
                "CausalSelfAttention requires PyTorch 2.0"
                " or later for flash attention support. "
                "Please upgrade your PyTorch version."
            )
            raise RuntimeError(error)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the causal self-attention module."""
        B, T, C = (  # noqa: N806
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch
        # and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1,
            2,
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1,
            2,
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1,
            2,
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend:
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        # efficient attention using Flash Attention CUDA kernels
        y: Tensor = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y  # noqa: RET504


class MLP(nn.Module):
    """MLP module."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize the MLP module."""
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the MLP module."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x  # noqa: RET504


class Block(nn.Module):
    """Transformer block with self-attention and MLP."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize the transformer block."""
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the transformer block."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x  # noqa: RET504


@dataclass
class GPTConfig:
    """Configuration for the GPT model."""

    block_size: int = 256
    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    bias: bool = False


@dataclass
class ExtraConfig:
    """Extra configuration for the GPT model."""

    ncolumns: int = 0
    max_blocks: int = 0


class NanoGPT(nn.Module):
    """GPT module."""

    def __init__(self, config: GPTConfig, extra_config: ExtraConfig) -> None:
        """Initialize the GPT module with the given configuration."""
        super().__init__()
        self.config = config
        self.extra_config = extra_config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            },
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (  # type: ignore[union-attr]
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

        # report number of parameters
        # print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model.

        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()  # type: ignore[operator,union-attr]
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass of the GPT model."""
        device = idx.device
        b, t = idx.size()
        if t > self.config.block_size:
            error = (
                f"Cannot forward sequence of length {t},"
                f" block size is only {self.config.block_size}"
            )
            raise RuntimeError(error)

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer.wte(idx)  # type: ignore[operator]
        # position embeddings of shape (t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # type: ignore[operator]
        x = self.transformer.drop(tok_emb + pos_emb)  # type: ignore[operator]
        for block in self.transformer.h:  # type: ignore[union-attr]
            x = block(x)
        x = self.transformer.ln_f(x)  # type: ignore[operator]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # inference-time mini-optimization:
            # only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :],
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """Configure optimizers for the model."""
        # start with all of the candidate parameters
        param_dict = dict(self.named_parameters())
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed,
        # otherwise no i.e. all weight tensors in matmuls + embeddings decay,
        # all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]  # noqa: PLR2004
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]  # noqa: PLR2004
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # num_decay_params = sum(p.numel() for p in decay_params)
        # num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(
        #     "num decayed parameter tensors:"
        #     f" {len(decay_params)}, with {num_decay_params:,} parameters",
        # )
        # print(
        #     f"num non-decayed parameter tensors:"
        #     f" {len(nodecay_params)}, with {num_nodecay_params:,} parameters",
        # )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args,
        )

    def estimate_mfu(self, fwdbwd_per_iter: float, dt: float) -> float:
        """Estimate model flops utilization (MFU) in A100 bfloat16 peak FLOPS."""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()  # noqa: N806
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size  # noqa: N806
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        end_tensor: Tensor,
        column_mask_dict: dict[int, Tensor] | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Generate new tokens from the model.

        Take a conditioning sequence of indices idx (LongTensor of shape (b,t))
        and complete the sequence max_new_tokens times, feeding the predictions back
        into the model each time. Most likely you'll want to make sure to be
        in model.eval() mode of operation for this.
        """
        max_tokens = 256
        batch_size = idx.size(0)
        ended = torch.zeros(batch_size, dtype=torch.bool, device=idx.device)

        for _ in range(idx.size(1), max_tokens):
            seq_size = idx.size(1)
            # if the sequence context is growing too long we must crop it at block_size
            if (
                self.extra_config.max_blocks > 0
                and seq_size > self.extra_config.max_blocks * self.extra_config.ncolumns
            ):
                idx_cond = idx[
                    :,
                    -(self.extra_config.max_blocks - 1) * self.extra_config.ncolumns
                    - (seq_size % self.extra_config.ncolumns) :,
                ]
            elif seq_size > self.config.block_size:
                # crop the idx to the last block_size tokens
                idx_cond = idx[:, -self.config.block_size :]
            else:
                idx_cond = idx

            if (
                self.extra_config.max_blocks > 0
                and seq_size % self.extra_config.ncolumns == 0
            ):
                idx_next = torch.ones(
                    (batch_size, 1),
                    dtype=torch.long,
                    device=idx.device,
                ) * (seq_size // self.extra_config.ncolumns + 1)
                idx = torch.cat((idx, idx_next), dim=1)
                continue

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # remove invalid tokens for the position
            position = seq_size % self.extra_config.ncolumns
            if column_mask_dict is not None:
                logits = torch.where(~column_mask_dict[position], float("-inf"), logits)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Update ended mask: mark rows where end token was just generated
            is_end = torch.isin(idx[:, -1], end_tensor)
            ended = ended | is_end
            # For rows that have ended, force idx_next to 0
            idx_next[ended] = 0

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            # Stop if model predicts end of sentence
            if torch.all(
                torch.isin(idx_next, end_tensor),
            ):
                break

        return idx
