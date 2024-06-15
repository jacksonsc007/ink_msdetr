# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
# from config import *


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, look_forward_twice=False, mixed_selection=False, use_ms_detr=False, use_aux_ffn=True):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.look_forward_twice = look_forward_twice
        self.mixed_selection = mixed_selection
        self.use_ms_detr = use_ms_detr

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, use_ms_detr=use_ms_detr, use_aux_ffn=use_aux_ffn)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec, look_forward_twice=look_forward_twice, use_ms_detr=use_ms_detr)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

        self.num_detection_stages = len( self.encoder.layers )
        assert self.num_detection_stages == len( self.decoder.layers )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio


    def forward(self, srcs, masks, pos_embeds, query_embed=None, **kwargs):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        hs_o2o = []
        hs_o2m = [] 
        inter_references = []
        memory = src_flatten
        enc_pos = lvl_pos_embed_flatten
        enc_padding_mask = mask_flatten
        enc_reference_points = self.encoder.get_reference_points(spatial_shapes, valid_ratios, device=src_flatten.device)

        '''Initialize query for decoder'''
        bs, _, c = memory.shape
        assert not self.two_stage
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, enc_padding_mask, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))

            if not self.mixed_selection:
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            else:
                # tgt: content embedding, query_embed here is the learnable content embedding
                tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)
                # query_embed: position embedding, transformed from the topk proposals
                query_embed, _ = torch.split(pos_trans_out, c, dim=2)

        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points
        init_dec_tgt = tgt
        dec_query_pos = query_embed
        init_dec_reference_points = reference_points

        '''backbone feature -> id-0 decoder layer'''
        dec_query_o2o, dec_query_o2m, dec_ref, dec_new_ref, dec_sampling_locations, dec_attention_weights= \
            self.decoder.cascade_stage_forward(0, init_dec_tgt, init_dec_reference_points, memory,
                spatial_shapes, level_start_index, valid_ratios, dec_query_pos, enc_padding_mask, **kwargs)

        hs_o2o.append(dec_query_o2o)
        hs_o2m.append(dec_query_o2m)
        inter_references.append(dec_new_ref if self.decoder.look_forward_twice else dec_ref)
        cross_attn_map_list = []

        valid_tokens_nums_all_imgs = (~mask_flatten).int().sum(dim=1)
        valid_enc_token_num =  (valid_tokens_nums_all_imgs * 0.3 ).int() + 1
        batch_token_num = max(valid_enc_token_num)
        bs, num_dec_q, nheads, nlvls, npoints, _ = dec_sampling_locations.shape
        topk_obj_num = int(num_dec_q * 0.2)
        for enc_start_idx, enc_end_idx, dec_start_idx, dec_end_idx in ( (0, 1, 1, 2), 
                                                                        (1, 3, 2, 3),
                                                                        (3, 6, 3, 6)):
            outputs_class = self.decoder.class_embed[dec_start_idx-1](dec_query_o2o).max(-1)[0]
            obj_q_idx = outputs_class.topk(topk_obj_num, dim=1)[1]
            # sampling_locations: (1, 300, 8, 4, 4, 2)
            dec_sampling_locations = torch.gather(dec_sampling_locations, dim=1, 
                                                  index=obj_q_idx.reshape(bs, topk_obj_num, 1, 1, 1, 1).expand(bs, topk_obj_num, nheads, nlvls, npoints, 2))
            dec_attention_weights = torch.gather(dec_attention_weights, dim=1,
                                                 index=obj_q_idx.reshape(bs, topk_obj_num, 1, 1, 1).expand(bs, topk_obj_num, nheads, nlvls, npoints))

            # ==== select tokens =====
            dec_sampling_locations = dec_sampling_locations[:, None]
            dec_attention_weights = dec_attention_weights[:, None]
            # (bs, 1, num_head, num_all_lvl_tokens) -> (bs, num_all_lvl_tokens)
            cross_attn_map = attn_map_to_flat_grid(spatial_shapes, level_start_index, dec_sampling_locations, dec_attention_weights).sum(dim=(1,2))
            assert cross_attn_map.size() == mask_flatten.size()
            cross_attn_map = cross_attn_map.masked_fill(mask_flatten, cross_attn_map.min()-1)
            topk_enc_token_indice = cross_attn_map.topk(batch_token_num, dim=1)[1] # (bs, batch_token_num)

            memory = self.encoder(enc_start_idx, enc_end_idx, enc_reference_points, memory, spatial_shapes, level_start_index, valid_ratios, 
                                  lvl_pos_embed_flatten, mask_flatten, topk_enc_token_indice, valid_enc_token_num)

            hs_o2o_, hs_o2m_, inter_references_, dec_sampling_locations_list, dec_attention_weights_list = self.decoder(dec_start_idx, dec_end_idx, dec_query_o2o, dec_ref, memory,
                                                spatial_shapes, level_start_index, valid_ratios, dec_query_pos, mask_flatten, **kwargs)
            dec_sampling_locations = dec_sampling_locations_list[-1]
            dec_attention_weights = dec_attention_weights_list[-1]
            dec_query_o2o = hs_o2o_[-1]
            dec_ref = inter_references_[-1].detach()

            hs_o2o = hs_o2o + hs_o2o_
            hs_o2m = hs_o2m + hs_o2m_
            inter_references = inter_references + inter_references_

        inter_references = torch.stack(inter_references)
        hs_o2m = torch.stack(hs_o2m)
        hs_o2o = torch.stack(hs_o2o)
        inter_references_out = inter_references
        # ===================== End cascade detection stage =====================
        if self.two_stage:
            return hs_o2o, hs_o2m, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, output_proposals.sigmoid(),
        return hs_o2o, hs_o2m, init_reference_out, inter_references_out, None, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, query, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None, topk_enc_token_indice=None, valid_enc_token_num=None):
        # self attention
        src2, _, _ = self.self_attn(self.with_pos_embed(query, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        if topk_enc_token_indice is not None:
            outputs=[]
            for img_id, (num, idx) in enumerate(zip(valid_enc_token_num, topk_enc_token_indice)):
                valid_idx = idx[:num]
                # src[0]: (ori_num_token, 256)
                # idx: (valid_num,) -> (valid_num, 256)
                # sparse_memory[0]: (max_token_num, 256)
                # src[0][index[i,j], j] = sparse_memory[0][i,j]
                outputs.append(src[img_id].scatter(dim=0, index=valid_idx.unsqueeze(1).repeat(1, src.size(-1)), src=src2[img_id][:num]))
            src2 = torch.stack(outputs)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, start_layer_idx, end_layer_idx, reference_points, src, spatial_shapes, level_start_index, valid_ratios, 
                pos, padding_mask, topk_enc_token_indice, valid_enc_token_num):

        output = src
        num_lvl = reference_points.size(2)
        d_model = src.size(2)
        # get sparse tokens
        sparse_enc_query_pos = pos.gather(dim=1, index=topk_enc_token_indice.unsqueeze(dim=2).repeat(1, 1, d_model))
        sparse_enc_ref = reference_points.gather(dim=1, index=topk_enc_token_indice.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, num_lvl, 2)) # (x, y) for ref points

        for layer_idx in range(start_layer_idx, end_layer_idx):
            sparse_enc_query = output.gather(dim=1, index=topk_enc_token_indice.unsqueeze(dim=2).repeat(1, 1, d_model))
            output = self.layers[layer_idx]( output, sparse_enc_query, sparse_enc_query_pos, sparse_enc_ref, spatial_shapes,
                                             level_start_index, padding_mask, topk_enc_token_indice, valid_enc_token_num)

        return output

    def cascade_stage_forward(self, stage_idx, src, query, spatial_shapes, level_start_index, reference_points, pos, padding_mask):
        # when use sparse tokens, src is not equivalent to query
        layer = self.layers[stage_idx]
        output = layer(src, query, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output



class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, use_ms_detr=False, use_aux_ffn=True):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.use_ms_detr = use_ms_detr
        self.use_aux_ffn = use_aux_ffn
        
        # auxiliary ffn
        if self.use_ms_detr and self.use_aux_ffn:
            self.linear3 = nn.Linear(d_model, d_ffn)
            self.dropout5 = nn.Dropout(dropout)
            self.linear4 = nn.Linear(d_ffn, d_model)
            self.dropout6 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_aux_ffn(self, tgt):
        tgt2 = self.linear4(self.dropout5(self.activation(self.linear3(tgt))))
        tgt = tgt + self.dropout6(tgt2)
        tgt = self.norm4(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        assert self.use_ms_detr
        if self.use_ms_detr:
            # cross attention
            tgt2, dec_sampling_locations, dec_attention_weights = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                                   reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            if self.use_aux_ffn:
                # auxiliary ffn
                tgt_o2m = self.forward_aux_ffn(tgt)
            else:
                tgt_o2m = tgt

            # self attention
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

            # ffn
            tgt_o2o = self.forward_ffn(tgt)

        else:
            # self attention
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

            # cross attention
            tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), 
                                   reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            # ffn
            tgt_o2o = tgt_o2m = self.forward_ffn(tgt)
        
        return tgt_o2o, tgt_o2m, dec_sampling_locations, dec_attention_weights


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, look_forward_twice=False, use_ms_detr=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.look_forward_twice = look_forward_twice
        self.use_ms_detr = use_ms_detr

    def forward(self, start_layer_idx, end_layer_idx, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, **kwargs):
        output = tgt

        intermediate = []
        intermediate_o2m = []
        intermediate_reference_points = []
        sampling_locations = []
        attn_weights = []
        for lid in range(start_layer_idx, end_layer_idx):
            layer = self.layers[lid]
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output, output_o2m, sampling_location, attn_weight = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, **kwargs)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_o2m.append(output_o2m)
                intermediate_reference_points.append(
                    new_reference_points
                    if self.look_forward_twice
                    else reference_points
                )
                sampling_locations.append(sampling_location)
                attn_weights.append(attn_weight)
        assert self.return_intermediate
        if self.return_intermediate:
            return intermediate, intermediate_o2m, intermediate_reference_points, sampling_locations, attn_weights

        return output, output_o2m, reference_points

    def cascade_stage_forward(
        self, stage_idx, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios, query_pos, src_padding_mask, **kwargs
    ):
        layer = self.layers[stage_idx]
        if reference_points.shape[-1] == 4:
            reference_points_input = reference_points[:, :, None] \
                                        * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
        output, output_o2m, dec_sampling_locations, dec_attention_weights = layer(tgt, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, **kwargs)

        # hack implementation for iterative bounding box refinement
        if self.bbox_embed is not None:
            tmp = self.bbox_embed[stage_idx](output)
            if reference_points.shape[-1] == 4:
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
            else:
                assert reference_points.shape[-1] == 2
                new_reference_points = tmp
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points.detach()

        assert self.return_intermediate, "cascade detection requires return_intermediate."
        if self.look_forward_twice:
            return output, output_o2m, reference_points, new_reference_points, dec_sampling_locations, dec_attention_weights
        else:
            return output, output_o2m, reference_points, None


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        mixed_selection=args.mixed_selection,
        look_forward_twice=args.look_forward_twice,
        use_ms_detr=args.use_ms_detr,
        use_aux_ffn=args.use_aux_ffn,
    )

def attn_map_to_flat_grid(spatial_shapes, level_start_index, sampling_locations, attention_weights):
    # sampling_locations: [N, n_layers, Len_q, n_heads, n_levels, n_points, 2]
    # attention_weights: [N, n_layers, Len_q, n_heads, n_levels, n_points]
    N, n_layers, _, n_heads, *_ = sampling_locations.shape
    sampling_locations = sampling_locations.permute(0, 1, 3, 2, 5, 4, 6).flatten(0, 2).flatten(1, 2)
    # [N * n_layers * n_heads, Len_q * n_points, n_levels, 2]
    attention_weights = attention_weights.permute(0, 1, 3, 2, 5, 4).flatten(0, 2).flatten(1, 2)
    # [N * n_layers * n_heads, Len_q * n_points, n_levels]

    rev_spatial_shapes = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1) # hw -> wh (xy)
    col_row_float = sampling_locations * rev_spatial_shapes
    # get 4 corner integeral positions around the floating-type sampling locations. 
    col_row_ll = col_row_float.floor().to(torch.int64)
    zero = torch.zeros(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)
    one = torch.ones(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)
    col_row_lh = col_row_ll + torch.stack([zero, one], dim=-1)
    col_row_hl = col_row_ll + torch.stack([one, zero], dim=-1)
    col_row_hh = col_row_ll + 1
    # compute magin for bilinear interpolation
    margin_ll = (col_row_float - col_row_ll).prod(dim=-1)
    margin_lh = -(col_row_float - col_row_lh).prod(dim=-1)
    margin_hl = -(col_row_float - col_row_hl).prod(dim=-1)
    margin_hh = (col_row_float - col_row_hh).prod(dim=-1)

    flat_grid_shape = (attention_weights.shape[0], int(torch.sum(spatial_shapes[..., 0] * spatial_shapes[..., 1])))
    flat_grid = torch.zeros(flat_grid_shape, dtype=torch.float32, device=attention_weights.device)

    zipped = [(col_row_ll, margin_hh), (col_row_lh, margin_hl), (col_row_hl, margin_lh), (col_row_hh, margin_ll)]
    for col_row, margin in zipped:
        valid_mask = torch.logical_and(
            torch.logical_and(col_row[..., 0] >= 0, col_row[..., 0] < rev_spatial_shapes[..., 0]),
            torch.logical_and(col_row[..., 1] >= 0, col_row[..., 1] < rev_spatial_shapes[..., 1]),
        )
        idx = col_row[..., 1] * spatial_shapes[..., 1] + col_row[..., 0] + level_start_index
        idx = (idx * valid_mask).flatten(1, 2)
        weights = (attention_weights * valid_mask * margin).flatten(1)
        flat_grid.scatter_add_(1, idx, weights)

    return flat_grid.reshape(N, n_layers, n_heads, -1)