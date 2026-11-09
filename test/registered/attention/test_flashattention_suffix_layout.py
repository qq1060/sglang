"""One-query FA4 suffixes preserve outputs, LSEs, and CUDA Graph replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")


class TestSuffixLayout(CustomTestCase):
    def test_layout_and_fallbacks(self):
        backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
        backend.num_splits = 1
        metadata = FlashAttentionMetadata(
            page_table=torch.zeros(3, 4, dtype=torch.int32),
            cache_seqlens_int32=torch.ones(3, dtype=torch.int32),
            cu_seqlens_q=torch.arange(4, dtype=torch.int32),
            cu_seqlens_k=torch.arange(4, dtype=torch.int32),
            max_seq_len_q=1,
        )
        backend.forward_metadata_spec_decode_expand = metadata
        q = torch.randn(3, 2, 64)
        k = v = torch.randn(4, 1, 1, 64)
        output, lse, scratch = torch.randn_like(q), torch.randn(2, 3), object()
        layer = SimpleNamespace(scaling=0.125, logit_cap=0.0)
        for ver, kwargs in (
            (4, {}),
            (3, {}),
            (4, {"sinks": torch.zeros(2)}),
            (4, {"q_descale": torch.ones(3, 1)}),
            (4, {"score_mod": object(), "aux_tensors": []}),
            (4, {"max_seqlen_k": 4}),
        ):
            with self.subTest(ver=ver, kwargs=tuple(kwargs)):
                fixed = ver == 4 and not kwargs
                backend.flash_attn_with_kvcache = Mock(
                    return_value=(
                        output.unsqueeze(1) if fixed else output,
                        lse.T.unsqueeze(-1) if fixed else lse,
                        scratch,
                    )
                )
                actual, actual_lse, actual_scratch = backend._forward_suffix_attention(
                    q, k, v, layer, (-1, -1), kwargs, ver
                )
                torch.testing.assert_close(actual, output)
                torch.testing.assert_close(actual_lse, lse)
                self.assertIs(actual_scratch, scratch)
                args = backend.flash_attn_with_kvcache.call_args.kwargs
                self.assertEqual(args["q"].shape, (3, 1, 2, 64) if fixed else q.shape)
                self.assertEqual(args["q"].data_ptr(), q.data_ptr())
                self.assertIs(
                    args["cu_seqlens_q"], None if fixed else metadata.cu_seqlens_q
                )
                self.assertIs(
                    args["cu_seqlens_k_new"], None if fixed else metadata.cu_seqlens_k
                )
                for name, value in kwargs.items():
                    self.assertIs(args[name], value)


@unittest.skipIf(get_device_sm() < 90, "FA4 paged attention requires SM90+")
class TestSuffixLayoutCUDA(CustomTestCase):
    def test_reference_and_graph_replay(self):
        for dtype in (torch.float16, torch.bfloat16):
            for heads, kv_heads, dim, value_dim, length, page_size, softcap in (
                (4, 4, 64, 64, 17, 1, 0.0),
                (8, 2, 128, 128, 128, 1, 0.0),
                (8, 1, 128, 128, 257, 64, 0.0),
                (8, 2, 256, 256, 65, 1, 0.0),
                (8, 2, 128, 64, 33, 16, 0.0),
                (8, 2, 128, 128, 17, 1, 10.0),
            ):
                with self.subTest(
                    dtype=dtype,
                    heads=heads,
                    kv_heads=kv_heads,
                    dim=dim,
                    value_dim=value_dim,
                    length=length,
                    page_size=page_size,
                    softcap=softcap,
                ):
                    self._case(
                        dtype,
                        heads,
                        kv_heads,
                        dim,
                        value_dim,
                        length,
                        page_size,
                        softcap,
                    )

    def _case(self, dtype, heads, kv_heads, dim, value_dim, length, page_size, softcap):
        torch.manual_seed(42)
        batch = 9
        pages_per_seq = (length + page_size - 1) // page_size
        pages = (
            torch.randperm(batch * pages_per_seq, device="cuda").int().view(batch, -1)
        )
        k = torch.randn(
            batch * pages_per_seq, page_size, kv_heads, dim, device="cuda", dtype=dtype
        )
        v = torch.randn(*k.shape[:-1], value_dim, device="cuda", dtype=dtype)
        q = torch.randn(batch, heads, dim, device="cuda", dtype=dtype)
        lengths = torch.linspace(1, length, batch, device="cuda").int()
        metadata = FlashAttentionMetadata(
            page_table=pages,
            cache_seqlens_int32=lengths,
            cu_seqlens_q=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
            cu_seqlens_k=torch.cat((lengths.new_zeros(1), lengths.cumsum(0).int())),
            max_seq_len_q=1,
        )
        backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
        backend.num_splits = 1
        backend.flash_attn_with_kvcache = flash_attn_with_kvcache
        backend.forward_metadata_spec_decode_expand = metadata
        layer = SimpleNamespace(scaling=dim**-0.5, logit_cap=softcap)

        def run():
            return backend._forward_suffix_attention(q, k, v, layer, (-1, -1), {}, 4)

        def check(output, lse):
            positions = torch.arange(length, device="cuda")
            slots = pages[:, positions // page_size] * page_size + positions % page_size
            head_indices = torch.arange(heads, device="cuda") // (heads // kv_heads)
            keys = k.flatten(0, 1)[slots.long()][:, :, head_indices].float()
            values = v.flatten(0, 1)[slots.long()][:, :, head_indices].float()
            scores = torch.einsum("bhd,blhd->bhl", q.float(), keys) * layer.scaling
            if softcap:
                scores = softcap * torch.tanh(scores / softcap)
            scores.masked_fill_(
                positions[None, None, :] >= lengths[:, None, None], -float("inf")
            )
            expected = torch.einsum("bhl,blhd->bhd", scores.softmax(-1), values)
            tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
            torch.testing.assert_close(
                output.float(), expected, atol=tolerance, rtol=tolerance
            )
            torch.testing.assert_close(
                lse, scores.logsumexp(-1).T, atol=1e-4, rtol=1e-4
            )

        output, lse, *_ = run()
        check(output, lse)
        original, original_lse, *_ = flash_attn_with_kvcache(
            q=q,
            k_cache=k,
            v_cache=v,
            page_table=pages,
            cache_seqlens=lengths,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k_new=metadata.cu_seqlens_k,
            max_seqlen_q=1,
            softmax_scale=layer.scaling,
            causal=False,
            softcap=softcap,
            return_softmax_lse=True,
            num_splits=1,
            ver=4,
        )
        check(original, original_lse)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for _ in range(2):
            q.normal_()
            k.normal_()
            v.normal_()
            pages.copy_(pages.roll(1, 0))
            lengths.copy_(lengths.roll(1))
            metadata.cu_seqlens_k[1:].copy_(lengths.cumsum(0))
            graph.replay()
            check(*captured[:2])


if __name__ == "__main__":
    unittest.main()
