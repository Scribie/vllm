# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the configurable Gemma 4 audio ceiling.

No model weights, no GPU: the processor is a stand-in carrying only the
attributes the code under test reads.
"""

from types import SimpleNamespace

import pytest

SR = 16000
MS_PER_TOKEN = 40


def _gemma4():
    from vllm.model_executor.models import gemma4_mm

    return gemma4_mm


def _processor(seq=750, fft=None, per_token=640):
    ext = SimpleNamespace(sampling_rate=SR, audio_samples_per_token=per_token)
    if fft is not None:
        ext.fft_length = fft
    return SimpleNamespace(
        audio_seq_length=seq,
        audio_ms_per_token=MS_PER_TOKEN,
        feature_extractor=ext,
        audio_token_id=7,
    )


def test_unset_env_keeps_upstream_ceiling(monkeypatch):
    monkeypatch.delenv("VLLM_GEMMA4_MAX_AUDIO_SECONDS", raising=False)
    p = _processor()
    assert _gemma4()._apply_gemma4_audio_cap(p).audio_seq_length == 750


@pytest.mark.parametrize("seconds,tokens", [("30", 750), ("120", 3000), ("300", 7500)])
def test_env_sets_ceiling_at_25_tokens_per_second(monkeypatch, seconds, tokens):
    monkeypatch.setenv("VLLM_GEMMA4_MAX_AUDIO_SECONDS", seconds)
    p = _gemma4()._apply_gemma4_audio_cap(_processor())
    assert p.audio_seq_length == tokens


def test_apply_is_idempotent(monkeypatch):
    monkeypatch.setenv("VLLM_GEMMA4_MAX_AUDIO_SECONDS", "300")
    p = _processor()
    _gemma4()._apply_gemma4_audio_cap(p)
    _gemma4()._apply_gemma4_audio_cap(p)
    assert p.audio_seq_length == 7500


@pytest.mark.parametrize("bad", ["0", "-5", "nan", "inf", "abc"])
def test_bad_env_raises(monkeypatch, bad):
    monkeypatch.setenv("VLLM_GEMMA4_MAX_AUDIO_SECONDS", bad)
    with pytest.raises(ValueError):
        _gemma4()._apply_gemma4_audio_cap(_processor())


def test_300s_clip_counts_7500_tokens_with_cap_and_750_without():
    info = _gemma4().Gemma4ProcessingInfo
    assert info._compute_audio_num_tokens(300 * SR, SR, 7500) == 7500
    assert info._compute_audio_num_tokens(300 * SR, SR, 750) == 750


def test_dummy_audio_len_does_not_need_fft_length():
    # Unified extractor: no fft_length. Worst case = full ceiling.
    g = _gemma4()
    assert g._gemma4_dummy_audio_len(_processor(seq=7500)) == 7500 * 640
    # Tower extractor keeps using its own fft_length.
    assert g._gemma4_dummy_audio_len(_processor(seq=7500, fft=512)) == 512


def _repl(g, audio_seconds, processor):
    stub = SimpleNamespace(
        _compute_audio_num_tokens=g.Gemma4ProcessingInfo._compute_audio_num_tokens,
        get_hf_config=lambda: SimpleNamespace(
            boa_token_id=1, eoa_token_id=2, eoa_token_index=2
        ),
    )
    r = g.Gemma4ProcessingInfo.get_audio_repl(
        stub, audio_len=int(audio_seconds * SR), processor=processor
    )
    ids = r.full if hasattr(r, "full") else list(r)
    return sum(1 for t in ids if t == processor.audio_token_id)


def test_overlength_default_clips_like_upstream(monkeypatch):
    monkeypatch.delenv("VLLM_GEMMA4_AUDIO_OVERLENGTH", raising=False)
    assert _repl(_gemma4(), 330, _processor(seq=7500)) == 7500


def test_overlength_refuse_raises_but_in_range_passes(monkeypatch):
    monkeypatch.setenv("VLLM_GEMMA4_AUDIO_OVERLENGTH", "refuse")
    g = _gemma4()
    assert _repl(g, 300, _processor(seq=7500)) == 7500
    with pytest.raises(ValueError, match="over-length"):
        _repl(g, 330, _processor(seq=7500))


def test_overlength_typo_raises(monkeypatch):
    monkeypatch.setenv("VLLM_GEMMA4_AUDIO_OVERLENGTH", "refues")
    with pytest.raises(ValueError):
        _repl(_gemma4(), 30, _processor())


def test_unified_processor_getter_applies_the_cap():
    import inspect

    from vllm.model_executor.models import gemma4_unified

    src = inspect.getsource(gemma4_unified.Gemma4UnifiedProcessingInfo.get_hf_processor)
    assert "_apply_gemma4_audio_cap" in src
