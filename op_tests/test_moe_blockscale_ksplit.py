# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Regression for the get_ksplit KBatch==1 bug (block-FP8 fused_moe gibberish).

The CK stage1 split-k kernel computes KBatch = K / (split * KPerBlock). When
that collapses to 1 the kernel still uses atomic-add but skips the output
memset, so results accumulate onto uninitialized memory (e.g. K=4096,
split=16 -> KBatch=1 -> bs=1 MoE gibberish). aiter/fused_moe.py get_ksplit now
requires KBatch >= 2; this test guards that.

Unlike test_moe_blockscale.py (which calls the asm kernel directly and never
touches get_ksplit), this exercises the high-level fused_moe() CK 2-stage
dispatch that get_ksplit actually feeds. Ground truth is a bf16 MoE built from
the *pre-quant* weights, so the only allowed error is fp8 block-quant noise
(~1-2%); a wrong split-k reduction blows the rel error well past the threshold.

Defaults reproduce Qwen3.5-397B-A17B-FP8 TP8 per-rank (inter=128). The real
E=512/topk=10 matters: a small E takes a different sorting/tiling path where
bs=1 looks broken for every config and no longer discriminates the bug.
"""

import torch
import torch.nn.functional as F
from aiter import dtypes, QuantType
from aiter.fused_moe import fused_moe, fused_topk
from aiter.ops.shuffle import shuffle_weight
from aiter import pertoken_quant
from aiter.utility import fp4_utils
import argparse

torch.set_default_device("cuda")

BLK = 128


def block_quant(w):
    """per-(128x128) fp8 quant of [E, d1, d2] -> (wq_fp8, scale, w_dequant_bf16)."""
    Ec, d1, d2 = w.shape
    b = w.view(Ec, d1 // BLK, BLK, d2 // BLK, BLK)
    b = b.permute(0, 1, 3, 2, 4).contiguous().view(Ec, -1, BLK * BLK)
    wq, s = pertoken_quant(b, quant_dtype=dtypes.fp8)
    wq5 = wq.view(Ec, d1 // BLK, d2 // BLK, BLK, BLK)
    wdq = wq5.float() * s.view(Ec, d1 // BLK, d2 // BLK, 1, 1)
    wdq = wdq.permute(0, 1, 3, 2, 4).contiguous().view(Ec, d1, d2).to(torch.bfloat16)
    wq = wq5.permute(0, 1, 3, 2, 4).contiguous().view(Ec, d1, d2)
    return wq, s.view(Ec, d1 // BLK, d2 // BLK), wdq


def ref_moe_bf16(h, w1dq, w2dq, tw, tid, hidden, inter, topk):
    """bf16 reference (no quant noise): silu(gate)*up then down, topk-weighted."""
    M = h.shape[0]
    out = torch.zeros(M, hidden, dtype=torch.float32)
    for m in range(M):
        acc = torch.zeros(hidden, dtype=torch.float32)
        for k in range(topk):
            e = tid[m, k].item()
            gu = h[m].float() @ w1dq[e].float().t()
            gate, up = gu[:inter], gu[inter:]
            act = F.silu(gate) * up
            acc += tw[m, k].item() * (act @ w2dq[e].float().t())
        out[m] = acc
    return out


def test_fmoe_ksplit(hidden, inter, E, topk, M, rtol=0.05):
    g = torch.Generator(device="cuda").manual_seed(inter * 100 + M)
    w1 = torch.randn(E, 2 * inter, hidden, dtype=torch.bfloat16, generator=g) * 0.1
    w2 = torch.randn(E, hidden, inter, dtype=torch.bfloat16, generator=g) * 0.1
    w1q, w1s, w1dq = block_quant(w1)
    w2q, w2s, w2dq = block_quant(w2)
    h = torch.randn(M, hidden, dtype=torch.bfloat16, generator=g) * 0.5
    score = torch.randn(M, E, dtype=torch.float32, generator=g)
    tw, tid = fused_topk(h, score, topk, True)

    ref = ref_moe_bf16(h, w1dq, w2dq, tw, tid, hidden, inter, topk)
    act = fused_moe(
        hidden_states=h,
        w1=shuffle_weight(w1q, layout=(16, 16)),
        w2=shuffle_weight(w2q, layout=(16, 16)),
        topk_weight=tw,
        topk_ids=tid,
        quant_type=QuantType.per_128x128,
        w1_scale=fp4_utils.e8m0_shuffle(w1s),
        w2_scale=fp4_utils.e8m0_shuffle(w2s),
    ).float()

    rel = (act[0] - ref[0]).norm() / ref[0].norm().clamp(min=1e-9)
    tag = "BROKEN" if rel.item() > rtol else "ok"
    msg = (
        f"hidden={hidden} inter={inter} E={E} topk={topk} M={M}: "
        f"row0 rel_err vs bf16-ref = {rel.item():.4f}  [{tag}] "
        f"(fp8 noise should be <{rtol}; large error means get_ksplit KBatch==1 bug)"
    )
    print(msg)
    assert rel.item() < rtol, msg


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="get_ksplit KBatch==1 regression for block-FP8 fused_moe",
)
parser.add_argument(
    "-dim",
    type=int,
    default=4096,
    help="""Hidden dimension. Default is 4096.
    e.g.: -dim 4096""",
)
parser.add_argument(
    "-idim",
    type=int,
    nargs="*",
    default=[256, 128],
    help="""Intermediate dimension(s); 128 == TP8 per-rank (the broken case).
    e.g.: -idim 256 128""",
)
parser.add_argument(
    "-e",
    "--expert",
    type=int,
    default=512,
    help="""Number of experts. Default is 512.
    e.g.: -e 512""",
)
parser.add_argument(
    "-k",
    "--topk",
    type=int,
    default=10,
    help="""Top-k value. Default is 10.
    e.g.: -k 10""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[1, 8],
    help="""M (token count); 1 == bs=1 decode (the broken case).
    e.g.: -m 1 8""",
)

args = parser.parse_args()

for inter in args.idim:
    for M in args.m:
        test_fmoe_ksplit(args.dim, inter, args.expert, args.topk, M)
