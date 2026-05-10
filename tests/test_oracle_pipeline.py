"""Lightweight regression tests for the Oracle training pipeline.

Covers the end-to-end smoke path:
    recorder JSONL → parser → collate → teacher train → student distill

All tests use tiny models (d_model=16, n_layers=1) and CPU only.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from models.transformer_policy_value import TransformerPolicyValueNet


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save_oracle_jsonl(path, records):
    """Write a list of dicts as JSONL."""
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def _make_oracle_step(pub_len=20, priv_len=100, has_private=True):
    """Return a single oracle step dict with valid structure."""
    pub_ids = list(range(1, pub_len + 1))
    pub_types = [1] * pub_len
    pub_bids = [0] * pub_len
    action_mask = [1.0] * 34 + [0.0] * 43  # first 34 actions legal
    if has_private:
        priv_ids = list(range(1, priv_len + 1))
        priv_types = [1] * priv_len
        priv_bids = [0] * priv_len
    else:
        priv_ids = []
        priv_types = []
        priv_bids = []
    return {
        'public_token_ids': pub_ids,
        'public_token_types': pub_types,
        'public_behavior_ids': pub_bids,
        'private_token_ids': priv_ids,
        'private_token_types': priv_types,
        'private_behavior_ids': priv_bids,
        'action_mask': action_mask,
        'chosen_action': 0,
        'reward': 1.0,
        'player_idx': 0,
        'game_seed': 0,
        'step': 1,
        'round_wind': 27,
        'round_number': 1,
        'honba': 0,
    }


def _save_teacher_checkpoint(path, max_len=128, d_model=16, n_layers=1, n_heads=2):
    """Save a teacher checkpoint, optionally without max_len metadata."""
    model = TransformerPolicyValueNet(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        n_concept=10, max_len=max_len)
    ckpt = {
        'model_state_dict': model.state_dict(),
        'metadata': {
            'model_arch': 'transformer',
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_concept': 10,
            'max_len': max_len,
        },
    }
    torch.save(ckpt, path)
    return path


def _save_teacher_checkpoint_no_meta(path, max_len=128, d_model=16, n_layers=1, n_heads=2):
    """Save a teacher checkpoint WITHOUT max_len in metadata."""
    model = TransformerPolicyValueNet(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        n_concept=10, max_len=max_len)
    ckpt = {
        'model_state_dict': model.state_dict(),
        'metadata': {
            'model_arch': 'transformer',
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_concept': 10,
            # max_len deliberately omitted
        },
    }
    torch.save(ckpt, path)
    return path


# ── Parser Tests ─────────────────────────────────────────────────────────────

class TestOracleTrajectoryParser:
    """OracleTrajectoryJSONLParser validation tests."""

    def test_parser_skips_empty_private(self):
        """Samples with empty private_token_ids must be skipped."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            # One valid, one with empty private
            f.write(json.dumps(_make_oracle_step(has_private=True)) + '\n')
            f.write(json.dumps(_make_oracle_step(has_private=False)) + '\n')
            f.write(json.dumps({'type': 'game_summary'}) + '\n')
            tmp = f.name

        try:
            samples = list(parser.parse_file(tmp))
            assert len(samples) == 1, f"Expected 1 sample, got {len(samples)}"
            assert samples[0][5].shape[0] > 0, "Private IDs should be non-empty"
        finally:
            os.unlink(tmp)

    def test_parser_skips_empty_public(self):
        """Samples with empty public_token_ids must be skipped."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()

        step = _make_oracle_step(has_private=True)
        step['public_token_ids'] = []

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name

        try:
            samples = list(parser.parse_file(tmp))
            assert len(samples) == 0
        finally:
            os.unlink(tmp)


# ── Recorder Token Production ────────────────────────────────────────────────

class TestRecorderTokenProduction:
    """Selfplay recorder must produce tokens regardless of agent architecture."""

    def test_heuristic_recorder_has_tokens(self):
        """Heuristic recorder (no model) must still produce public/private tokens."""
        from training.selfplay_recorder import record_games
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'test.jsonl')
            stats = record_games(output_path=out, num_games=1, agent=None,
                                 tokenizer=None, progress_every=0)
            assert stats['total_steps'] > 0

            from data.record_parser import OracleTrajectoryJSONLParser
            parser = OracleTrajectoryJSONLParser()
            samples = list(parser.parse_file(out))
            assert len(samples) > 0, "Heuristic recorder should produce token samples"
            # Check first sample has non-empty public and private
            assert samples[0][0].shape[0] > 0, "Public IDs should be non-empty"
            assert samples[0][5].shape[0] > 0, "Private IDs should be non-empty"


# ── Distillation Tests ───────────────────────────────────────────────────────

class TestDistillationGating:
    """Teacher forward must execute when either KL or value distillation is active."""

    def test_value_only_distillation_produces_value_mse(self):
        """distill_alpha=0, distill_value_alpha>0 must execute teacher forward."""
        model = TransformerPolicyValueNet(
            d_model=16, n_layers=1, n_heads=2, n_concept=10, max_len=64)
        model.train()

        B, S_pub, S_priv = 2, 15, 10
        token_ids = torch.randint(1, 50, (B, S_pub))
        token_types = torch.randint(0, 5, (B, S_pub))
        behavior_ids = torch.randint(0, 20, (B, S_pub))
        attn = torch.zeros(B, S_pub, dtype=torch.bool)
        action_mask = torch.ones(B, 77)
        labels = torch.zeros(B, dtype=torch.long)
        priv_ids = torch.randint(1, 50, (B, S_priv))
        priv_types = torch.randint(0, 5, (B, S_priv))
        priv_bids = torch.randint(0, 20, (B, S_priv))
        priv_attn = torch.zeros(B, S_priv, dtype=torch.bool)

        # Student forward
        out_s = model(token_ids, token_types, behavior_ids, attn, action_mask,
                      mode='student')

        # Teacher forward
        with torch.no_grad():
            out_t = model(token_ids, token_types, behavior_ids, attn, action_mask,
                          private_token_ids=priv_ids, private_token_types=priv_types,
                          private_behavior_ids=priv_bids, private_attention_mask=priv_attn,
                          mode='teacher')

        # Simulate distill_alpha=0, distill_value_alpha=1
        distill_alpha = 0.0
        distill_value_alpha = 1.0
        has_private = True

        distill_kl = torch.tensor(0.0)
        value_mse = torch.tensor(0.0)

        if has_private and (distill_alpha > 0 or distill_value_alpha > 0):
            if distill_alpha > 0:
                from training.distillation import masked_kl_loss
                distill_kl = masked_kl_loss(out_t['policy_logits'].detach(),
                                            out_s['policy_logits'], action_mask)
            if distill_value_alpha > 0:
                oracle_v = out_t.get('oracle_value', out_t['value']).detach()
                value_mse = torch.nn.MSELoss()(out_s['value'], oracle_v)

        assert distill_kl.item() == 0.0, "KL should be 0 when distill_alpha=0"
        assert value_mse.item() > 0, "value_mse should be > 0 when distill_value_alpha>0"


# ── Teacher Checkpoint max_len Inference ──────────────────────────────────────

class TestTeacherMaxLenInference:
    """Teacher max_len must be inferred from state_dict when metadata is missing."""

    def test_infer_max_len_from_pos_embedding(self):
        """When metadata lacks max_len, infer from backbone.pos_embedding shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, 'teacher_no_meta.pt')
            _save_teacher_checkpoint_no_meta(ckpt_path, max_len=128)

            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            meta = ckpt.get('metadata', {})
            sd = ckpt['model_state_dict']

            # This is the logic from sl_pretrain.py
            state_max_len = sd['backbone.pos_embedding'].shape[1] \
                if 'backbone.pos_embedding' in sd else None
            teacher_max_len = meta.get('max_len')
            if teacher_max_len is None:
                teacher_max_len = state_max_len if state_max_len else 256

            assert teacher_max_len == 128, f"Expected 128, got {teacher_max_len}"

            # Model should load without error
            model = TransformerPolicyValueNet(
                d_model=16, n_layers=1, n_heads=2, n_concept=10,
                max_len=teacher_max_len)
            model.load_state_dict(sd)

    def test_metadata_conflict_uses_state_dict(self):
        """When metadata max_len conflicts with state_dict, state_dict wins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, 'teacher_mismatch.pt')
            # Save model with max_len=128 but metadata says 64
            model = TransformerPolicyValueNet(
                d_model=16, n_layers=1, n_heads=2, n_concept=10, max_len=128)
            ckpt = {
                'model_state_dict': model.state_dict(),
                'metadata': {'model_arch': 'transformer', 'max_len': 64},
            }
            torch.save(ckpt, ckpt_path)

            loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            meta = loaded.get('metadata', {})
            sd = loaded['model_state_dict']
            state_max_len = sd['backbone.pos_embedding'].shape[1]

            # When conflict, use state_dict
            assert state_max_len == 128
            assert meta.get('max_len') == 64
            # Actual max_len should come from state_dict
            teacher_max_len = meta.get('max_len')
            if teacher_max_len != state_max_len:
                teacher_max_len = state_max_len
            assert teacher_max_len == 128


# ── Empty Private Token Rejection ────────────────────────────────────────────

class TestEmptyPrivateRejection:
    """Oracle training must reject samples with empty private tokens."""

    def test_parser_rejects_empty_private(self):
        """OracleTrajectoryJSONLParser must skip empty-private samples."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            # All samples have empty private
            for _ in range(5):
                f.write(json.dumps(_make_oracle_step(has_private=False)) + '\n')
            tmp = f.name

        try:
            samples = list(parser.parse_file(tmp))
            assert len(samples) == 0, \
                f"Expected 0 samples (all empty private), got {len(samples)}"
        finally:
            os.unlink(tmp)

    def test_collate_empty_private_low_level(self):
        """Low-level: collate can pad empty private to (B, 0) tensors.

        Note: The parser now rejects empty-private samples before they reach collate.
        This test documents collate's padding behavior for non-oracle datasets.
        """
        from data.dataset import collate_transformer_batch
        pub_ids = torch.arange(1, 11)
        pub_types = torch.ones(10, dtype=torch.long)
        pub_bids = torch.zeros(10, dtype=torch.long)
        action_mask = torch.ones(77)
        label = torch.tensor(0)
        priv_ids = torch.tensor([], dtype=torch.long)
        priv_types = torch.tensor([], dtype=torch.long)
        priv_bids = torch.tensor([], dtype=torch.long)
        reward = torch.tensor(1.0)

        batch = [(pub_ids, pub_types, pub_bids, action_mask, label,
                  priv_ids, priv_types, priv_bids, reward)]
        result = collate_transformer_batch(batch)

        assert 'private_token_ids' in result
        assert result['private_token_ids'].shape[1] == 0, \
            "Empty private should produce (B, 0) tensor"


# ── Parser Schema Validation ─────────────────────────────────────────────────

class TestParserSchemaValidation:
    """OracleTrajectoryJSONLParser must reject malformed samples."""

    def test_illegal_chosen_action_raises(self):
        """chosen_action not in action_mask must raise ValueError."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()

        step = _make_oracle_step(has_private=True)
        step['chosen_action'] = 10
        step['action_mask'][10] = 0.0  # make chosen action illegal

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name

        try:
            with pytest.raises(ValueError, match="is illegal"):
                list(parser.parse_file(tmp))
        finally:
            os.unlink(tmp)

    def test_wrong_action_mask_length_raises(self):
        """action_mask length != 77 must raise ValueError."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()

        step = _make_oracle_step(has_private=True)
        step['action_mask'] = [1.0, 0.0]  # wrong length

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name

        try:
            with pytest.raises(ValueError, match="action_mask length"):
                list(parser.parse_file(tmp))
        finally:
            os.unlink(tmp)

    def test_mismatched_public_lengths_raises(self):
        """public_token_ids/types/behavior_ids length mismatch must raise ValueError."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()

        step = _make_oracle_step(has_private=True)
        step['public_token_types'] = [1]  # length 1, but ids length is 20

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name

        try:
            with pytest.raises(ValueError, match="public field length mismatch"):
                list(parser.parse_file(tmp))
        finally:
            os.unlink(tmp)

    def test_mismatched_private_lengths_raises(self):
        """private_token_ids/types/behavior_ids length mismatch must raise ValueError."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()

        step = _make_oracle_step(has_private=True)
        step['private_token_types'] = [1]  # length 1, but ids length is 100

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name

        try:
            with pytest.raises(ValueError, match="private field length mismatch"):
                list(parser.parse_file(tmp))
        finally:
            os.unlink(tmp)

    def test_out_of_range_chosen_action_raises(self):
        """chosen_action >= 77 must raise ValueError."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()

        step = _make_oracle_step(has_private=True)
        step['chosen_action'] = 100

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name

        try:
            with pytest.raises(ValueError, match="out of range"):
                list(parser.parse_file(tmp))
        finally:
            os.unlink(tmp)

    def test_valid_sample_passes(self):
        """A well-formed sample must pass all validation."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()

        step = _make_oracle_step(has_private=True)
        step['chosen_action'] = 0  # legal (action_mask[0]=1.0)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name

        try:
            samples = list(parser.parse_file(tmp))
            assert len(samples) == 1
        finally:
            os.unlink(tmp)


# ── masked_kl_loss Validation ────────────────────────────────────────────────

class TestMaskedKLLossValidation:
    """masked_kl_loss must reject all-zero action masks."""

    def test_all_zero_mask_raises(self):
        from training.distillation import masked_kl_loss
        teacher = torch.randn(2, 77)
        student = torch.randn(2, 77)
        mask = torch.zeros(2, 77)
        with pytest.raises(ValueError, match="no legal actions"):
            masked_kl_loss(teacher, student, mask)

    def test_valid_mask_works(self):
        from training.distillation import masked_kl_loss
        teacher = torch.randn(2, 77)
        student = torch.randn(2, 77)
        mask = torch.zeros(2, 77)
        mask[0, 0] = 1.0
        mask[1, 0:3] = 1.0
        kl = masked_kl_loss(teacher, student, mask)
        assert torch.isfinite(kl)


# ── Advanced Parser Validation ────────────────────────────────────────────

class TestAdvancedParserValidation:
    """Parser must reject NaN reward, out-of-range IDs, non-binary action_mask."""

    def test_nan_reward_raises(self):
        """NaN reward must raise ValueError."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()
        step = _make_oracle_step(has_private=True)
        step['reward'] = float('nan')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="not finite"):
                list(parser.parse_file(tmp))
        finally:
            os.unlink(tmp)

    def test_inf_reward_raises(self):
        """Inf reward must raise ValueError."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()
        step = _make_oracle_step(has_private=True)
        step['reward'] = float('inf')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="not finite"):
                list(parser.parse_file(tmp))
        finally:
            os.unlink(tmp)

    def test_negative_token_id_raises(self):
        """Negative token ID must raise ValueError."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()
        step = _make_oracle_step(has_private=True)
        step['public_token_ids'][0] = -1
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="< 0"):
                list(parser.parse_file(tmp))
        finally:
            os.unlink(tmp)

    def test_negative_private_token_id_raises(self):
        """Negative private token ID must raise ValueError."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()
        step = _make_oracle_step(has_private=True)
        step['private_token_ids'][0] = -5
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="< 0"):
                list(parser.parse_file(tmp))
        finally:
            os.unlink(tmp)

    def test_out_of_range_behavior_id_raises(self):
        """Behavior ID >= 64 must raise ValueError."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()
        step = _make_oracle_step(has_private=True)
        step['public_behavior_ids'][0] = 100
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="out of range"):
                list(parser.parse_file(tmp))
        finally:
            os.unlink(tmp)

    def test_non_binary_action_mask_raises(self):
        """action_mask value of 0.5 must raise ValueError."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()
        step = _make_oracle_step(has_private=True)
        step['action_mask'][0] = 0.5
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="not binary"):
                list(parser.parse_file(tmp))
        finally:
            os.unlink(tmp)

    def test_nan_action_mask_raises(self):
        """NaN in action_mask must raise ValueError."""
        import math
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()
        step = _make_oracle_step(has_private=True)
        step['action_mask'][0] = float('nan')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="not finite"):
                list(parser.parse_file(tmp))
        finally:
            os.unlink(tmp)

    def test_valid_reward_finite(self):
        """Valid finite reward (positive) must pass."""
        from data.record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()
        step = _make_oracle_step(has_private=True)
        step['reward'] = 0.75
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False,
                                          encoding='utf-8') as f:
            f.write(json.dumps(step) + '\n')
            tmp = f.name
        try:
            samples = list(parser.parse_file(tmp))
            assert len(samples) == 1
            assert samples[0][8] == 0.75  # reward at index 8
        finally:
            os.unlink(tmp)
