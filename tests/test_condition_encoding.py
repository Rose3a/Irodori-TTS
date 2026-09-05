from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from irodori_tts.inference_runtime import InferenceRuntime, SamplingRequest
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.rf import sample_euler_rf_cfg


class _SamplingModel:
    device = torch.device("cpu")
    dtype = torch.float32
    cfg = SimpleNamespace(
        patched_latent_dim=2,
        use_speaker_condition_resolved=False,
        use_caption_condition=False,
    )

    def __init__(self, *, fail_on_encode: bool = True) -> None:
        self.encode_calls = 0
        self.fail_on_encode = fail_on_encode

    def encode_conditions(self, **kwargs):
        self.encode_calls += 1
        if self.fail_on_encode:
            raise AssertionError("pre-encoded conditions must be reused")
        return (
            torch.ones((1, 3, 4), dtype=torch.float32),
            kwargs["text_mask"],
            None,
            None,
            None,
            None,
        )

    def forward_with_encoded_conditions(self, *, x_t, **kwargs):
        return torch.zeros_like(x_t)


class _ConditionEncoder(nn.Module):
    def __init__(self, output_dim: int, *, fail_on_call: bool = False) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.fail_on_call = fail_on_call
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.fail_on_call:
            raise AssertionError("empty caption must not invoke its encoder")
        return torch.ones((*input_ids.shape, self.output_dim), dtype=torch.float32)


class _Tokenizer:
    def batch_encode(self, texts, max_length):
        return torch.ones((len(texts), 3), dtype=torch.long), torch.ones(
            (len(texts), 3), dtype=torch.bool
        )


class _DurationModel:
    def __init__(self, encoded_conditions) -> None:
        self.encoded_conditions = encoded_conditions
        self.encode_calls = 0

    def encode_conditions(self, **kwargs):
        self.encode_calls += 1
        return self.encoded_conditions

    def predict_duration_log_frames(self, **kwargs):
        return torch.log1p(torch.tensor([1.0]))


class ConditionEncodingTests(unittest.TestCase):
    def test_sample_reuses_preencoded_conditions(self) -> None:
        model = _SamplingModel()
        text_state = torch.ones((1, 3, 4), dtype=torch.float32)
        text_mask = torch.ones((1, 3), dtype=torch.bool)
        encoded_conditions = (text_state, text_mask, None, None, None, None)

        result = sample_euler_rf_cfg(
            model=model,
            text_input_ids=torch.ones((1, 3), dtype=torch.long),
            text_mask=text_mask,
            ref_latent=None,
            ref_mask=None,
            sequence_length=2,
            num_steps=1,
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            use_context_kv_cache=False,
            encoded_conditions=encoded_conditions,
            seed=1,
        )

        self.assertEqual(model.encode_calls, 0)
        self.assertEqual(result.shape, (1, 2, 2))

    def test_sample_encodes_when_preencoded_conditions_are_absent(self) -> None:
        model = _SamplingModel(fail_on_encode=False)
        text_mask = torch.ones((1, 3), dtype=torch.bool)

        sample_euler_rf_cfg(
            model=model,
            text_input_ids=torch.ones((1, 3), dtype=torch.long),
            text_mask=text_mask,
            ref_latent=None,
            ref_mask=None,
            sequence_length=2,
            num_steps=1,
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            use_context_kv_cache=False,
            seed=1,
        )

        self.assertEqual(model.encode_calls, 1)

    def test_empty_caption_skips_caption_encoder(self) -> None:
        caption_encoder = _ConditionEncoder(5, fail_on_call=True)
        model = SimpleNamespace(
            training=False,
            cfg=SimpleNamespace(
                use_speaker_condition_resolved=False,
                use_caption_condition=True,
                caption_dim_resolved=5,
            ),
            pretrained_text_backbone=None,
            text_encoder=_ConditionEncoder(3),
            text_norm=nn.Identity(),
            caption_encoder=caption_encoder,
            caption_norm=nn.Identity(),
        )

        encoded = TextToLatentRFDiT.encode_conditions(
            model,
            text_input_ids=torch.tensor([[1, 2, 3]]),
            text_mask=torch.tensor([[True, True, True]]),
            ref_latent=None,
            ref_mask=None,
            caption_input_ids=torch.zeros((1, 4), dtype=torch.long),
            caption_mask=torch.zeros((1, 4), dtype=torch.bool),
            skip_caption_encoding=True,
        )

        caption_state = encoded[4]
        self.assertEqual(caption_encoder.calls, 0)
        self.assertIsNotNone(caption_state)
        self.assertEqual(caption_state.shape, (1, 4, 5))
        self.assertEqual(torch.count_nonzero(caption_state).item(), 0)

    def test_nonempty_caption_uses_caption_encoder(self) -> None:
        caption_encoder = _ConditionEncoder(5)
        model = SimpleNamespace(
            training=False,
            cfg=SimpleNamespace(
                use_speaker_condition_resolved=False,
                use_caption_condition=True,
                caption_dim_resolved=5,
            ),
            pretrained_text_backbone=None,
            text_encoder=_ConditionEncoder(3),
            text_norm=nn.Identity(),
            caption_encoder=caption_encoder,
            caption_norm=nn.Identity(),
        )

        encoded = TextToLatentRFDiT.encode_conditions(
            model,
            text_input_ids=torch.tensor([[1, 2, 3]]),
            text_mask=torch.tensor([[True, True, True]]),
            ref_latent=None,
            ref_mask=None,
            caption_input_ids=torch.tensor([[1, 2, 0, 0]]),
            caption_mask=torch.tensor([[True, True, False, False]]),
        )

        self.assertEqual(caption_encoder.calls, 1)
        self.assertGreater(torch.count_nonzero(encoded[4]).item(), 0)

    def test_training_keeps_empty_caption_encoder_in_graph(self) -> None:
        caption_encoder = _ConditionEncoder(5)
        model = SimpleNamespace(
            training=True,
            cfg=SimpleNamespace(
                use_speaker_condition_resolved=False,
                use_caption_condition=True,
                caption_dim_resolved=5,
            ),
            pretrained_text_backbone=None,
            text_encoder=_ConditionEncoder(3),
            text_norm=nn.Identity(),
            caption_encoder=caption_encoder,
            caption_norm=nn.Identity(),
        )

        TextToLatentRFDiT.encode_conditions(
            model,
            text_input_ids=torch.tensor([[1, 2, 3]]),
            text_mask=torch.tensor([[True, True, True]]),
            ref_latent=None,
            ref_mask=None,
            caption_input_ids=torch.zeros((1, 4), dtype=torch.long),
            caption_mask=torch.zeros((1, 4), dtype=torch.bool),
            skip_caption_encoding=True,
        )

        self.assertEqual(caption_encoder.calls, 1)

    def test_runtime_passes_duration_conditions_to_sampler(self) -> None:
        text_state = torch.ones((1, 3, 4), dtype=torch.float32)
        text_mask = torch.ones((1, 3), dtype=torch.bool)
        encoded_conditions = (text_state, text_mask, None, None, None, None)
        model = _DurationModel(encoded_conditions)
        runtime = object.__new__(InferenceRuntime)
        runtime.key = SimpleNamespace(
            model_device="cpu",
            model_precision="fp32",
            codec_device="cpu",
            codec_precision="fp32",
        )
        runtime.watermarker = None
        runtime.model_device = torch.device("cpu")
        runtime.codec_device = torch.device("cpu")
        runtime.model_cfg = SimpleNamespace(
            use_caption_condition=False,
            use_speaker_condition_resolved=False,
            use_duration_predictor=True,
            latent_patch_size=1,
            latent_dim=2,
        )
        runtime.default_text_max_len = 3
        runtime.default_caption_max_len = 3
        runtime.tokenizer = _Tokenizer()
        runtime.caption_tokenizer = None
        runtime.model = model
        runtime.codec = SimpleNamespace(
            sample_rate=48000,
            model=SimpleNamespace(hop_length=480),
            decode_latent=lambda z: torch.zeros((z.shape[0], 1, 480)),
        )
        runtime.train_cfg = {}
        runtime._infer_lock = nullcontext()
        runtime._prepare_lora_for_request = lambda *args, **kwargs: nullcontext()
        runtime._load_speaker_embedding_condition = lambda **kwargs: (None, None)
        runtime._load_reference_latent = lambda **kwargs: (None, None)

        captured = {}

        def fake_sample(**kwargs):
            captured["encoded_conditions"] = kwargs.get("encoded_conditions")
            captured["skip_caption_encoding"] = kwargs.get("skip_caption_encoding")
            return torch.zeros((1, 1, 2), dtype=torch.float32)

        with patch("irodori_tts.inference_runtime.sample_euler_rf_cfg", side_effect=fake_sample):
            runtime.synthesize(
                SamplingRequest(
                    text="test",
                    no_ref=True,
                    min_seconds=0.01,
                    max_seconds=0.01,
                    trim_tail=False,
                    seed=1,
                )
            )

        self.assertIs(captured["encoded_conditions"], encoded_conditions)
        self.assertIs(captured["skip_caption_encoding"], True)
        self.assertEqual(model.encode_calls, 1)


if __name__ == "__main__":
    unittest.main()
