"""
Neural network architecture for the flow model.

Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/flow_model.py
"""

import torch
from torch import nn

from ribocouple.models.node_embedder import NodeEmbedder
from ribocouple.models import torsion_net
from ribocouple.data import utils as du
from ribocouple.models.ipa_pytorch import InvariantPointAttention, StructureModuleTransition, BackboneUpdate, EdgeTransition, Linear
from ribocouple.models.edge_embedder import EdgeEmbedder
from ribocouple.models.sequence_head import PairAwareTransformerSeqHead

try:
    from flash_ipa.utils import check_config_ipa
    from flash_ipa.edge_embedder import EdgeEmbedder as FlashEdgeEmbedder
    from flash_ipa.ipa import (
        InvariantPointAttention as FlashInvariantPointAttention,
        StructureModuleTransition as FlashStructureModuleTransition,
        BackboneUpdate as FlashBackboneUpdate,
        EdgeTransition as FlashEdgeTransition,
    )
    from flash_ipa.linear import Linear as FlashLinear
    HAS_FLASH_IPA = True
except ImportError:
    HAS_FLASH_IPA = False

    def check_config_ipa(*args, **kwargs):
        return None

    FlashEdgeEmbedder = None
    FlashInvariantPointAttention = None
    FlashStructureModuleTransition = None
    FlashBackboneUpdate = None
    FlashEdgeTransition = None
    FlashLinear = None


class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self._topology_conf = getattr(model_conf, 'topology', None)
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.node_embedder = NodeEmbedder(model_conf.node_features)

        self.use_flashipa = model_conf.use_flashipa
        if self.use_flashipa:
        # Check variables are consistent for experiment.
            check_config_ipa(model_conf.ipa, model_conf.mode)
            self.mode = model_conf.mode
            self.edge_embedder = FlashEdgeEmbedder(model_conf.edge_features)
        else:
            self.edge_embedder = EdgeEmbedder(model_conf.edge_features)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            if self.use_flashipa:
                self.trunk[f'ipa_{b}'] = FlashInvariantPointAttention(self._ipa_conf)
            else:
                self.trunk[f'ipa_{b}'] = InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            if self.use_flashipa:
                self.trunk[f'post_tfmr_{b}'] = FlashLinear(
                    tfmr_in, self._ipa_conf.c_s, init="final")
                self.trunk[f'node_transition_{b}'] = FlashStructureModuleTransition(
                    c=self._ipa_conf.c_s)
                self.trunk[f'bb_update_{b}'] = FlashBackboneUpdate(
                    self._ipa_conf.c_s, use_rot_updates=True)
            else:
                self.trunk[f'post_tfmr_{b}'] = Linear(
                    tfmr_in, self._ipa_conf.c_s, init="final")
                self.trunk[f'node_transition_{b}'] = StructureModuleTransition(
                    c=self._ipa_conf.c_s)
                self.trunk[f'bb_update_{b}'] = BackboneUpdate(
                    self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                if self.use_flashipa:
                    self.trunk[f'edge_transition_{b}'] = FlashEdgeTransition(
                        mode="2d" if self.mode == "orig_2d_bias" else "1d",
                        node_embed_size=self._ipa_conf.c_s,
                        edge_embed_in=edge_in,
                        edge_embed_out=self._model_conf.edge_embed_size,
                        z_factor_rank=self._ipa_conf.z_factor_rank,
                    )
                else:
                    self.trunk[f'edge_transition_{b}'] = EdgeTransition(
                        node_embed_size=self._ipa_conf.c_s,
                        edge_embed_in=edge_in,
                        edge_embed_out=self._model_conf.edge_embed_size,
                    )

        # hparams taken from OpenFold's config.py
        self.angle_pred_net = torsion_net.TorsionAngleHead(c_in=self._ipa_conf.c_s, c_hidden=128, no_blocks=2, no_angles=8, epsilon=1e-12)
        self.pair_pred_net = nn.Sequential(
            nn.Linear(self._model_conf.edge_embed_size, self._model_conf.edge_embed_size),
            nn.ReLU(),
            nn.Linear(self._model_conf.edge_embed_size, 1),
        )

        # Sequence head configuration
        self._seq_head_conf = getattr(model_conf, 'sequence_head', None)
        self._use_sequence_head = False
        self._seq_head_num_classes = 4
        self._seq_head_hidden_dim = 256
        self._seq_head_use_pair_context = True
        self._seq_head_use_node_embed = True
        self._seq_head_use_transformer = False
        self._seq_head_transformer_num_layers = 4
        self._seq_head_transformer_num_heads = 8
        self._seq_head_transformer_hidden_dim = 256
        self._seq_head_transformer_dropout = 0.1
        self._seq_head_transformer_use_pair_bias = True
        self._use_pair_type_head = False
        self._pair_type_num_classes = 4
        self._seq_head_transformer_use_pair_type_bias = False
        if self._seq_head_conf is not None:
            self._use_sequence_head = bool(getattr(self._seq_head_conf, 'enabled', False))
            self._seq_head_num_classes = int(getattr(self._seq_head_conf, 'num_classes', 4))
            self._seq_head_hidden_dim = int(getattr(self._seq_head_conf, 'hidden_dim', 256))
            self._seq_head_use_pair_context = bool(
                getattr(self._seq_head_conf, 'use_pair_context', True)
            )
            self._seq_head_use_node_embed = bool(
                getattr(self._seq_head_conf, 'use_node_embed', True)
            )
            self._seq_head_use_transformer = bool(
                getattr(self._seq_head_conf, 'use_transformer_head', False)
            )
            self._seq_head_transformer_num_layers = int(
                getattr(self._seq_head_conf, 'transformer_num_layers', 4)
            )
            self._seq_head_transformer_num_heads = int(
                getattr(self._seq_head_conf, 'transformer_num_heads', 8)
            )
            self._seq_head_transformer_hidden_dim = int(
                getattr(self._seq_head_conf, 'transformer_hidden_dim', 256)
            )
            self._seq_head_transformer_dropout = float(
                getattr(self._seq_head_conf, 'transformer_dropout', 0.1)
            )
            self._seq_head_transformer_use_pair_bias = bool(
                getattr(self._seq_head_conf, 'transformer_use_pair_bias', True)
            )
            self._use_pair_type_head = bool(
                getattr(self._seq_head_conf, 'use_pair_type_head', False)
            )
            self._pair_type_num_classes = int(
                getattr(self._seq_head_conf, 'pair_type_num_classes', 4)
            )
            self._seq_head_transformer_use_pair_type_bias = bool(
                getattr(self._seq_head_conf, 'transformer_use_pair_type_bias', False)
            )

        if self._use_pair_type_head:
            self.pair_type_pred_net = nn.Sequential(
                nn.Linear(self._model_conf.edge_embed_size, self._model_conf.edge_embed_size),
                nn.ReLU(),
                nn.Linear(self._model_conf.edge_embed_size, self._pair_type_num_classes),
            )
        else:
            self.pair_type_pred_net = None

        if self._use_sequence_head:
            if self._seq_head_use_transformer and not self.use_flashipa:
                self.seq_pred_net = PairAwareTransformerSeqHead(
                    node_dim=self._ipa_conf.c_s,
                    num_classes=self._seq_head_num_classes,
                    num_layers=self._seq_head_transformer_num_layers,
                    num_heads=self._seq_head_transformer_num_heads,
                    hidden_dim=self._seq_head_transformer_hidden_dim,
                    dropout=self._seq_head_transformer_dropout,
                    use_pair_bias=self._seq_head_transformer_use_pair_bias,
                    pair_type_num_classes=self._pair_type_num_classes,
                    use_pair_type_bias=self._seq_head_transformer_use_pair_type_bias,
                )
            else:
                seq_in_dim = 0
                if self._seq_head_use_node_embed:
                    seq_in_dim += self._ipa_conf.c_s
                if self._seq_head_use_pair_context:
                    seq_in_dim += self._ipa_conf.c_s
                if seq_in_dim == 0:
                    raise ValueError(
                        "sequence_head must use at least one of node_embed or pair_context"
                    )
                self.seq_pred_net = nn.Sequential(
                    nn.Linear(seq_in_dim, self._seq_head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self._seq_head_hidden_dim, self._seq_head_num_classes),
                )
        else:
            self.seq_pred_net = None

        self._use_topology_feedback = False
        self._topology_feedback_detach = True
        self._topology_feedback_start_block = max(self._ipa_conf.num_blocks - 2, 0)
        self._topology_feedback_scale = 0.1
        if self._topology_conf is not None:
            self._use_topology_feedback = bool(getattr(self._topology_conf, 'use_feedback', False))
            self._topology_feedback_detach = bool(getattr(self._topology_conf, 'detach_pair_prob', True))
            self._topology_feedback_start_block = int(
                getattr(self._topology_conf, 'feedback_start_block', max(self._ipa_conf.num_blocks - 2, 0))
            )
            self._topology_feedback_scale = float(getattr(self._topology_conf, 'feedback_scale', 0.1))

        self.topology_feedback_proj = nn.Sequential(
            nn.Linear(1, self._model_conf.edge_embed_size),
            nn.GELU(),
        )
        self._topology_feedback_logged_once = False

        # Pair recycle configuration
        self._pair_recycle_conf = getattr(model_conf, 'pair_recycle', None)
        self._use_pair_recycle = False
        self._pair_recycle_detach = True
        self._pair_recycle_scale = 1.0
        self._pair_recycle_pass1_pair_bce = True
        if self._pair_recycle_conf is not None:
            self._use_pair_recycle = bool(getattr(self._pair_recycle_conf, 'use_recycle', False))
            self._pair_recycle_detach = bool(getattr(self._pair_recycle_conf, 'detach_pair_prob', True))
            self._pair_recycle_scale = float(getattr(self._pair_recycle_conf, 'recycle_scale', 1.0))
            self._pair_recycle_pass1_pair_bce = bool(
                getattr(self._pair_recycle_conf, 'pass1_pair_bce', True)
            )

        self.pair_recycle_proj = nn.Sequential(
            nn.Linear(1, self._model_conf.edge_embed_size),
            nn.GELU(),
        )
        self._pair_recycle_logged_once = False

    def _build_initial_embeddings(self, input_feats):
        """Build initial node/edge embeddings and rigids from input features."""
        node_mask = input_feats['res_mask']
        continuous_t = input_feats['t']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']

        init_node_embed = self.node_embedder(continuous_t, node_mask)
        init_node_embed = init_node_embed * node_mask[..., None]

        if self.use_flashipa:
            edge_embed, z_factor_1, z_factor_2, edge_mask = self.edge_embedder(
                init_node_embed, trans_t, trans_sc, node_mask)
        else:
            edge_mask = node_mask[:, None] * node_mask[:, :, None]
            edge_embed = self.edge_embedder(init_node_embed, trans_t, trans_sc, edge_mask)
            z_factor_1, z_factor_2 = None, None

        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)

        return {
            'init_node_embed': init_node_embed,
            'edge_embed': edge_embed,
            'edge_mask': edge_mask,
            'z_factor_1': z_factor_1,
            'z_factor_2': z_factor_2,
            'curr_rigids': curr_rigids,
            'node_mask': node_mask,
        }

    def _inject_pair_recycle(self, init_state, pair_prob):
        """Add recycled pair probability to initial edge embeddings."""
        edge_embed = init_state['edge_embed']
        edge_mask = init_state['edge_mask']
        z_factor_1 = init_state.get('z_factor_1')
        z_factor_2 = init_state.get('z_factor_2')

        recycle_bias = self.pair_recycle_proj(pair_prob.unsqueeze(-1))
        edge_embed = edge_embed + self._pair_recycle_scale * recycle_bias
        edge_embed = edge_embed * edge_mask[..., None]

        # For flashIPA, we do not have a direct 2D edge_embed to recycle into.
        # For flashIPA, keep z_factors unchanged because it has no direct 2D edge embeddings.
        return {
            'edge_embed': edge_embed,
            'edge_mask': edge_mask,
            'z_factor_1': z_factor_1,
            'z_factor_2': z_factor_2,
        }

    def _forward_trunk(self, init_state):
        """Run the IPA trunk once and return outputs."""
        node_embed = init_state['init_node_embed'] * init_state['node_mask'][..., None]
        edge_embed = init_state['edge_embed']
        edge_mask = init_state['edge_mask']
        z_factor_1 = init_state.get('z_factor_1')
        z_factor_2 = init_state.get('z_factor_2')
        curr_rigids = init_state['curr_rigids']
        node_mask = init_state['node_mask']
        init_node_embed = init_state['init_node_embed']

        topology_feedback_stats = None
        for b in range(self._ipa_conf.num_blocks):
            if self.use_flashipa:
                ipa_embed = self.trunk[f'ipa_{b}'](
                    node_embed,
                    edge_embed,
                    z_factor_1,
                    z_factor_2,
                    curr_rigids,
                    node_mask,
                )
            else:
                ipa_embed = self.trunk[f'ipa_{b}'](
                    node_embed,
                    edge_embed,
                    curr_rigids,
                    node_mask,
                )

            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(node_mask < 0.5))
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                if (
                    self._use_topology_feedback and
                    b >= self._topology_feedback_start_block and
                    not self.use_flashipa
                ):
                    latest_pair_logits = self.pair_pred_net(edge_embed).squeeze(-1)
                    pair_prob = torch.sigmoid(latest_pair_logits)
                    pair_prob_stats = pair_prob.detach()
                    topology_feedback_stats = {
                        'enabled': 1.0,
                        'block': float(b),
                        'pair_prob_mean': pair_prob_stats.mean(),
                        'pair_prob_std': pair_prob_stats.std(),
                        'pair_prob_min': pair_prob_stats.min(),
                        'pair_prob_max': pair_prob_stats.max(),
                    }
                    if self._topology_feedback_detach:
                        pair_prob = pair_prob.detach()
                    topo_bias = self.topology_feedback_proj(pair_prob.unsqueeze(-1))
                    edge_embed = edge_embed + self._topology_feedback_scale * topo_bias
                    edge_embed = edge_embed * edge_mask[..., None]

                if not self.use_flashipa or self.mode == "orig_2d_bias":
                    edge_embed = self.trunk[f'edge_transition_{b}'](node_embed, edge_embed)
                    edge_embed *= edge_mask[..., None]
                elif self.mode == "flash_1d_bias" or self.mode == "flash_2d_factorize_bias":
                    z_factor_1, z_factor_2 = self.trunk[f"edge_transition_{b}"](
                        node_embed, None, z_factor_1, z_factor_2)
                    z_factor_1 *= node_mask[:, :, None, None]
                    z_factor_2 *= node_mask[:, :, None, None]
                else:
                    continue

        _, pred_torsions = self.angle_pred_net(node_embed, init_node_embed)
        pair_logits = self.pair_pred_net(edge_embed).squeeze(-1)
        pair_type_logits = None
        if self._use_pair_type_head and self.pair_type_pred_net is not None:
            pair_type_logits = self.pair_type_pred_net(edge_embed)

        # Topology-aware sequence head
        seq_logits = None
        if self._use_sequence_head and not self.use_flashipa:
            node_mask = init_state['node_mask']
            if self._seq_head_use_transformer:
                seq_logits = self.seq_pred_net(
                    node_embed,
                    pair_logits,
                    node_mask,
                    pair_type_logits=pair_type_logits,
                )
            else:
                seq_input_parts = []
                if self._seq_head_use_node_embed:
                    seq_input_parts.append(node_embed)
                if self._seq_head_use_pair_context:
                    pair_prob = torch.sigmoid(pair_logits)
                    # Aggregate pair information into each node: [B, L, c_s]
                    pair_context = torch.matmul(pair_prob, node_embed)
                    seq_input_parts.append(pair_context)
                seq_input = torch.cat(seq_input_parts, dim=-1)
                seq_logits = self.seq_pred_net(seq_input)

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

        return {
            'pred_torsions': pred_torsions,
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            'pair_logits': pair_logits,
            'pair_type_logits': pair_type_logits,
            'seq_logits': seq_logits,
            'topology_feedback_stats': topology_feedback_stats,
        }

    def forward(self, input_feats):
        init_state = self._build_initial_embeddings(input_feats)

        pair_recycle_stats = None
        pair_logits_pass1 = None

        if self._use_pair_recycle and not self.use_flashipa:
            # Pass 1: predict pair probabilities.
            # We need gradients iff pair head is supervised from pass1 OR if we do
            # not detach the recycle signal (rare). Otherwise run no_grad to save
            # memory.
            pass1_grad_enabled = (
                self._pair_recycle_pass1_pair_bce or not self._pair_recycle_detach
            )
            with torch.set_grad_enabled(pass1_grad_enabled):
                pass1_output = self._forward_trunk(init_state)
            pair_logits_pass1 = pass1_output['pair_logits']
            # Free the structure outputs from pass 1; only keep pair_logits.
            del pass1_output

            pair_prob = torch.sigmoid(pair_logits_pass1)
            # Respect the configured gradient path from pass 2 to pass 1.
            pair_prob_for_cond = pair_prob.detach() if self._pair_recycle_detach else pair_prob

            pair_recycle_stats = {
                'enabled': 1.0,
                'pair_prob_mean': float(pair_prob.mean()),
                'pair_prob_std': float(pair_prob.std()),
                'pair_prob_min': float(pair_prob.min()),
                'pair_prob_max': float(pair_prob.max()),
            }

            # Re-initialize edge embeddings with recycled pair information
            recycled = self._inject_pair_recycle(init_state, pair_prob_for_cond)
            init_state['edge_embed'] = recycled['edge_embed']
            init_state['z_factor_1'] = recycled.get('z_factor_1')
            init_state['z_factor_2'] = recycled.get('z_factor_2')

            if not self._pair_recycle_logged_once:
                print(f"[Pair recycle] enabled, detach={self._pair_recycle_detach}, "
                      f"scale={self._pair_recycle_scale}")
                self._pair_recycle_logged_once = True

        # Pass 2 (or single pass if pair recycle disabled / flashIPA): final output
        output = self._forward_trunk(init_state)
        output['pair_logits_pass1'] = pair_logits_pass1
        output['pair_recycle_stats'] = pair_recycle_stats
        return output
