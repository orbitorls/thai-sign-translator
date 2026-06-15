"""Sign-to-text Transformer for Thai SLT.

SignToTextTransformer is an encoder-decoder Transformer that maps a
padded landmark sequence (B, T_src, D) to a distribution over Thai
target tokens (B, T_tgt, vocab_size). Unlike LandmarkEncoder, it does
not mean-pool the encoder output; the full temporal memory is kept so
the autoregressive decoder can attend to every source frame.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tsl.models.encoder import _SinusoidalPositionalEncoding


__all__ = ["SignToTextTransformer"]


class SignToTextTransformer(nn.Module):
    """Encoder-decoder Transformer for Thai sign-to-text translation.

    Input: (B, T_src, D) landmark/keypoint sequence with lengths
    Output: (B, T_tgt, vocab_size) logits over Thai target tokens
    """

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_pos_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.src_pos_enc = _SinusoidalPositionalEncoding(d_model, max_len=max_pos_len)
        self.tgt_pos_enc = _SinusoidalPositionalEncoding(d_model, max_len=max_pos_len)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def _build_src_pad_mask(self, src: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        T = src.size(1)
        arange = torch.arange(T, device=src.device)
        lengths = src_lengths.to(src.device)
        return arange[None, :] >= lengths[:, None]

    def encode(self, src: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        """Returns (B, T_src, d_model) memory for the decoder."""
        pad_mask = self._build_src_pad_mask(src, src_lengths)
        h = self.input_proj(src)
        h = self.src_pos_enc(h)
        return self.encoder(h, src_key_padding_mask=pad_mask)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns logits (B, T_tgt, vocab_size).

        The caller is responsible for right-shifting ``tgt`` (e.g.
        ``tgt = [BOS, w1, w2, ...]``) and for providing labels shifted
        one step to the left when computing the loss.
        """
        memory = self.encode(src, src_lengths)
        memory_key_padding_mask = self._build_src_pad_mask(src, src_lengths)
        T_tgt = tgt.size(1)
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                T_tgt, device=tgt.device, dtype=memory.dtype
            )
        h = self.tgt_embed(tgt)
        h = self.tgt_pos_enc(h)
        h = self.decoder(
            tgt=h,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_proj(h)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 128,
    ) -> torch.Tensor:
        """Returns (B, <=max_len) token ids.

        Starts each sequence with ``[bos_id]`` and extends autoregressively.
        Stops when ``eos_id`` is produced for each sequence (independently)
        or when ``max_len`` is reached.
        """
        B = src.size(0)
        memory = self.encode(src, src_lengths)
        memory_key_padding_mask = self._build_src_pad_mask(src, src_lengths)
        decoded = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            T_tgt = decoded.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                T_tgt, device=decoded.device, dtype=memory.dtype
            )
            h = self.tgt_embed(decoded)
            h = self.tgt_pos_enc(h)
            h = self.decoder(
                tgt=h,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            logits = self.out_proj(h[:, -1, :])
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, eos_id), next_token
            )
            decoded = torch.cat([decoded, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_id)
            if finished.all():
                break
        return decoded

    @torch.no_grad()
    def beam_decode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        bos_id: int,
        eos_id: int,
        beam_size: int = 5,
        max_len: int = 128,
        length_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Beam-search decoding. Returns (B, <=max_len) token ids (best hypothesis)."""
        B = src.size(0)
        device = src.device
        memory = self.encode(src, src_lengths)
        memory_key_padding_mask = self._build_src_pad_mask(src, src_lengths)

        all_hyps: list[list[list[int]]] = []
        for b in range(B):
            mem = memory[b:b+1]
            mem_mask = memory_key_padding_mask[b:b+1] if memory_key_padding_mask is not None else None

            beams = [(torch.full((1, 1), bos_id, dtype=torch.long, device=device), 0.0)]
            completed: list[tuple[list[int], float]] = []

            for _ in range(max_len - 1):
                new_beams: list[tuple[torch.Tensor, float]] = []
                for seq, score in beams:
                    if seq[0, -1].item() == eos_id:
                        completed.append((seq[0].tolist(), score))
                        continue
                    T_tgt = seq.size(1)
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                        T_tgt, device=device, dtype=memory.dtype
                    )
                    h = self.tgt_embed(seq)
                    h = self.tgt_pos_enc(h)
                    h = self.decoder(
                        tgt=h,
                        memory=mem.expand(1, -1, -1) if mem.size(0) == 1 else mem,
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=mem_mask,
                    )
                    logits = self.out_proj(h[:, -1, :])
                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)
                    for k in range(beam_size):
                        new_seq = torch.cat([seq, topk_ids[:, k:k+1]], dim=1)
                        new_score = score + topk_log_probs[0, k].item()
                        new_beams.append((new_seq, new_score))

                if not new_beams:
                    break
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_size]

                if len(completed) >= beam_size:
                    break

            for seq, score in beams:
                if seq[0, -1].item() != eos_id:
                    completed.append((seq[0].tolist(), score))

            completed.sort(
                key=lambda x: x[1] / ((len(x[0]) - 1) ** length_penalty),
                reverse=True,
            )
            best = completed[0][0] if completed else [bos_id]
            all_hyps.append(best)

        max_hyp_len = max(len(h) for h in all_hyps)
        out = torch.full((B, max_hyp_len), eos_id, dtype=torch.long, device=device)
        for i, hyp in enumerate(all_hyps):
            out[i, :len(hyp)] = torch.tensor(hyp, dtype=torch.long, device=device)
        return out
