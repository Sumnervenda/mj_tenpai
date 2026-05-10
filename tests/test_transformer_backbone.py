"""Tests for Transformer Backbone + Multi-Task Heads + TransformerPolicyValueNet."""
import pytest
import torch

from models.transformer_backbone import TransformerBlock, TransformerBackbone
from models.multi_task_heads import (
    MultiTaskHeads, ShantenHead, EfficiencyHead, DangerHead,
    ScoreHead, PolicyHead, ValueHead,
)
from models.transformer_policy_value import TransformerPolicyValueNet
from models.tokenizer import TokenVocab, TokenType


# =============================================================================
# Transformer Block
# =============================================================================

class TestTransformerBlock:
    def test_forward_shape(self):
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)
        x = torch.randn(2, 16, 256)
        out = block(x)
        assert out.shape == (2, 16, 256)

    def test_forward_with_mask(self):
        block = TransformerBlock(d_model=256, n_heads=8)
        x = torch.randn(2, 16, 256)
        mask = torch.zeros(2, 16, dtype=torch.bool)
        mask[:, 12:] = True  # last 4 positions are padding
        out = block(x, key_padding_mask=mask)
        assert out.shape == (2, 16, 256)

    def test_pre_ln_stability(self):
        """Pre-LN should handle unscaled inputs gracefully."""
        block = TransformerBlock(d_model=256, n_heads=8)
        x = torch.randn(2, 16, 256) * 100  # large unscaled input
        out = block(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_different_configs(self):
        for d_model, n_heads in [(128, 4), (256, 8), (512, 8)]:
            block = TransformerBlock(d_model=d_model, n_heads=n_heads)
            x = torch.randn(2, 8, d_model)
            out = block(x)
            assert out.shape == (2, 8, d_model)


# =============================================================================
# Transformer Backbone
# =============================================================================

class TestTransformerBackbone:
    def test_forward_shape(self):
        backbone = TransformerBackbone(
            layers=2, d_model=256, n_heads=8, d_ff=1024)
        x = torch.randn(2, 32, 256)
        out = backbone(x)
        assert out.shape == (2, 32, 256)

    def test_with_mask(self):
        backbone = TransformerBackbone(layers=2, d_model=256, n_heads=8)
        x = torch.randn(2, 32, 256)
        mask = torch.zeros(2, 32, dtype=torch.bool)
        mask[:, 30:] = True
        out = backbone(x, mask=mask)
        assert out.shape == (2, 32, 256)

    def test_positional_embedding_added(self):
        """Positional embedding should change output."""
        backbone = TransformerBackbone(layers=1, d_model=64, n_heads=4)
        x_same = torch.ones(1, 4, 64)  # all identical inputs

        # Without pos_embed, identical inputs would produce identical outputs
        # With different positions, outputs should differ
        out = backbone(x_same)
        # Different positions should have different outputs
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_different_lengths(self):
        backbone = TransformerBackbone(
            layers=2, d_model=128, n_heads=4)
        x_short = torch.randn(1, 8, 128)
        x_long = torch.randn(1, 32, 128)
        out_short = backbone(x_short)
        out_long = backbone(x_long)
        assert out_short.shape == (1, 8, 128)
        assert out_long.shape == (1, 32, 128)

    def test_count_parameters(self):
        backbone = TransformerBackbone(layers=2, d_model=128, n_heads=4)
        assert backbone.count_parameters() > 0


# =============================================================================
# Multi-Task Heads
# =============================================================================

class TestShantenHead:
    def test_forward_shape(self):
        head = ShantenHead(d_model=256)
        concept = torch.randn(2, 2, 256)
        out = head(concept)
        assert out.shape == (2, 7)

    def test_output_range(self):
        head = ShantenHead(d_model=64)
        concept = torch.randn(4, 2, 64)
        out = head(concept)
        assert out.shape == (4, 7)
        # Logits, no specific range constraint


class TestEfficiencyHead:
    def test_forward_shape(self):
        head = EfficiencyHead(d_model=256)
        concept = torch.randn(2, 2, 256)
        out = head(concept)
        assert out.shape == (2, 3)


class TestDangerHead:
    def test_forward_shape(self):
        head = DangerHead(d_model=256)
        concept = torch.randn(2, 2, 256)
        out = head(concept)
        assert out.shape == (2, 34)


class TestScoreHead:
    def test_forward_shape(self):
        head = ScoreHead(d_model=256)
        concept = torch.randn(2, 1, 256)
        out = head(concept)
        assert out.shape == (2, 1)


class TestPolicyHead:
    def test_forward_shape(self):
        head = PolicyHead(d_model=256, action_dim=77)
        concept = torch.randn(2, 1, 256)
        out = head(concept)
        assert out.shape == (2, 77)

    def test_action_mask(self):
        head = PolicyHead(d_model=64, action_dim=77)
        concept = torch.randn(2, 1, 64)
        mask = torch.zeros(2, 77)
        mask[:, 0] = 1.0  # only action 0 is legal
        out = head(concept, action_mask=mask)
        # Masked actions should have very negative logits
        assert out[:, 1:].max() < -1e9, "Masked actions not suppressed"


class TestValueHead:
    def test_forward_shape(self):
        head = ValueHead(d_model=256)
        concept = torch.randn(2, 2, 256)
        out = head(concept)
        assert out.shape == (2, 1)

    def test_tanh_range(self):
        head = ValueHead(d_model=64)
        concept = torch.randn(10, 2, 64) * 10  # large values
        out = head(concept)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


class TestMultiTaskHeads:
    def test_all_heads_output(self):
        heads = MultiTaskHeads(d_model=256)
        concept = torch.randn(2, 10, 256)
        outputs = heads(concept)

        assert 'policy_logits' in outputs
        assert 'value' in outputs
        assert 'shanten' in outputs
        assert 'efficiency' in outputs
        assert 'danger' in outputs
        assert 'score_value' in outputs

        assert outputs['policy_logits'].shape == (2, 77)
        assert outputs['value'].shape == (2, 1)
        assert outputs['shanten'].shape == (2, 7)
        assert outputs['efficiency'].shape == (2, 3)
        assert outputs['danger'].shape == (2, 34)
        assert outputs['score_value'].shape == (2, 1)

    def test_action_mask_passed_through(self):
        heads = MultiTaskHeads(d_model=64)
        concept = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 77)
        mask[:, :10] = 1.0
        outputs = heads(concept, action_mask=mask)

        # Masked actions should be suppressed
        assert outputs['policy_logits'][:, 10:].max() < -1e9

    def test_count_parameters(self):
        heads = MultiTaskHeads(d_model=128)
        assert heads.count_parameters() > 0


# =============================================================================
# TransformerPolicyValueNet (Integration)
# =============================================================================

class TestTransformerPolicyValueNet:
    @pytest.fixture
    def model(self):
        return TransformerPolicyValueNet(
            vocab_size=128,
            num_token_types=6,
            d_model=128,  # small for testing
            n_concept=10,
            n_layers=2,
            n_heads=4,
            d_ff=512,
        )

    @pytest.fixture
    def sample_batch(self):
        B, S = 2, 32
        return {
            'token_ids': torch.randint(1, 50, (B, S)),
            'token_types': torch.randint(0, 6, (B, S)),
            'behavior_ids': torch.randint(0, 20, (B, S)),
            'attention_mask': torch.zeros(B, S, dtype=torch.bool),
            'action_mask': torch.ones(B, 77),
        }

    def test_forward(self, model, sample_batch):
        outputs = model(**sample_batch)

        assert outputs['policy_logits'].shape == (2, 77)
        assert outputs['value'].shape == (2, 1)
        assert outputs['shanten'].shape == (2, 7)
        assert outputs['efficiency'].shape == (2, 3)
        assert outputs['danger'].shape == (2, 34)
        assert outputs['score_value'].shape == (2, 1)

    def test_forward_no_behavior_ids(self, model, sample_batch):
        del sample_batch['behavior_ids']
        outputs = model(**sample_batch)
        assert outputs['policy_logits'].shape == (2, 77)

    def test_forward_single_sample(self, model):
        S = 32
        token_ids = torch.randint(1, 50, (S,))
        token_types = torch.randint(0, 6, (S,))
        action_mask = torch.ones(77)

        outputs = model(
            token_ids.unsqueeze(0),
            token_types.unsqueeze(0),
            attention_mask=torch.zeros(1, S, dtype=torch.bool),
            action_mask=action_mask.unsqueeze(0),
        )
        assert outputs['policy_logits'].shape == (1, 77)

    def test_get_action_deterministic(self, model):
        S = 32
        token_ids = torch.randint(1, 50, (S,))
        token_types = torch.randint(0, 6, (S,))
        action_mask = torch.ones(77)

        action_idx, log_prob = model.get_action(
            token_ids=token_ids,
            token_types=token_types,
            action_mask=action_mask,
            deterministic=True,
        )
        assert isinstance(action_idx, int)
        assert 0 <= action_idx < 77
        assert isinstance(log_prob, torch.Tensor)

    def test_get_action_sampling(self, model, sample_batch):
        # Single sample
        action_idx, log_prob = model.get_action(
            token_ids=sample_batch['token_ids'][0],
            token_types=sample_batch['token_types'][0],
            behavior_ids=sample_batch['behavior_ids'][0],
            attention_mask=sample_batch['attention_mask'][0],
            action_mask=sample_batch['action_mask'][0],
            deterministic=False,
        )
        assert isinstance(action_idx, int)
        assert 0 <= action_idx < 77

    def test_evaluate_actions(self, model, sample_batch):
        action_indices = torch.randint(0, 77, (2,))
        log_probs, values, entropy = model.evaluate_actions(
            **sample_batch, action_indices=action_indices)
        assert log_probs.shape == (2,)
        assert values.shape == (2, 1)
        assert entropy.shape == (2,)

    def test_action_mask_effect(self, model, sample_batch):
        """Masked actions should have very low probability."""
        mask = torch.zeros(1, 77)
        mask[:, 0] = 1.0  # only action 0 is legal

        outputs = model(
            token_ids=sample_batch['token_ids'][:1],
            token_types=sample_batch['token_types'][:1],
            behavior_ids=sample_batch['behavior_ids'][:1],
            attention_mask=sample_batch['attention_mask'][:1],
            action_mask=mask,
        )
        # Action 0 should have finite logits
        assert torch.isfinite(outputs['policy_logits'][0, 0])
        # Other actions should be -inf
        assert outputs['policy_logits'][0, 1:].max() < -1e9

    def test_diversity_loss(self, model):
        loss = model.compute_diversity_loss()
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_count_parameters(self, model):
        count = model.count_parameters()
        assert count > 0
        print(f"TransformerPolicyValueNet parameters: {count:,}")

    def test_variable_length(self, model):
        """Handle different sequence lengths in a batch via attention mask."""
        token_ids = torch.randint(1, 50, (2, 16))
        token_types = torch.randint(0, 6, (2, 16))
        mask = torch.zeros(2, 16, dtype=torch.bool)

        outputs = model(
            token_ids=token_ids,
            token_types=token_types,
            attention_mask=mask,
        )
        assert outputs['policy_logits'].shape == (2, 77)


# =============================================================================
# Gradient Flow
# =============================================================================

class TestGradientFlow:
    def test_backprop_full_network(self):
        model = TransformerPolicyValueNet(
            vocab_size=64, d_model=64, n_concept=10,
            n_layers=2, n_heads=4, d_ff=256)
        B, S = 2, 16
        token_ids = torch.randint(1, 50, (B, S))
        token_types = torch.randint(0, 6, (B, S))
        behavior_ids = torch.randint(0, 20, (B, S))
        action_mask = torch.ones(B, 77)

        # Student forward
        outputs = model(token_ids, token_types, behavior_ids,
                       action_mask=action_mask)
        loss = sum(v.sum() for v in outputs.values())

        # Teacher forward to exercise private_concept_tokens
        priv_ids = torch.randint(1, 50, (B, 8))
        priv_types = torch.randint(0, 6, (B, 8))
        t_out = model(token_ids, token_types, behavior_ids,
                      action_mask=action_mask,
                      private_token_ids=priv_ids,
                      private_token_types=priv_types,
                      mode="teacher")
        loss = loss + sum(v.sum() for v in t_out.values() if v is not None)

        loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert param.grad.abs().sum() > 0, f"{name} gradient is zero"

    def test_diversity_loss_backprop(self):
        model = TransformerPolicyValueNet(
            vocab_size=64, d_model=64, n_concept=10,
            n_layers=1, n_heads=4, d_ff=256)
        loss = model.compute_diversity_loss()
        loss.backward()
        assert model.concept_tokens.grad is not None
        assert model.private_concept_tokens.grad is not None
# 中文注释：验证 Transformer Backbone、多任务预测头和 TransformerPolicyValueNet 的前向传播与梯度流。

