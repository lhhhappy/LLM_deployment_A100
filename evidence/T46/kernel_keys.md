# Static kernel inventory, exact 150 candidate

Every row also specializes on pointer dtypes, constexpr values and runtime scalar alignment/equality unless excluded. No row here certifies service execution.

| Source/function | Autotune key | JIT constexpr | do_not_specialize |
|---|---|---|---|
| kernels/ops/attention/fla/chunk_delta_h.py:53 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | ['H', 'K', 'V', 'BT', 'USE_GK', 'NT_BUCKET'] | H, Hg, K, V, BT, BV, USE_G, USE_GK, USE_INITIAL_STATE, INPLACE_UPDATE, SAVE_NEW_VALUE, IS_VARLEN, NT_BUCKET, USE_EXP2 | ['T'] |
| kernels/ops/attention/fla/chunk_delta_h_snapshot.py:53 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64_snapshot` | ['H', 'K', 'V', 'BT', 'USE_GK', 'NT_BUCKET'] | H, Hg, K, V, BT, BV, USE_G, USE_GK, USE_INITIAL_STATE, INPLACE_UPDATE, SAVE_NEW_VALUE, IS_VARLEN, NT_BUCKET, USE_EXP2, EXPORT_SNAPSHOTS | ['T'] |
| kernels/ops/attention/fla/chunk_fwd.py:40 `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | ['H', 'Hg', 'K', 'BC'] | H, Hg, K, BT, BC, BK, USE_G, IS_VARLEN | ['T'] |
| kernels/ops/attention/fla/chunk_intra.py:47 `chunk_kda_fwd_kernel_inter_solve_fused` | ['H', 'K', 'BC', 'V', 'FUSE_RECOMPUTE', 'FUSE_DIAGONAL'] | H, K, V, BT, BC, BK, BV, IS_VARLEN, USE_SAFE_GATE, FUSE_RECOMPUTE, FUSE_DIAGONAL | ['T'] |
| kernels/ops/attention/fla/chunk_intra.py:799 `chunk_kda_fwd_kernel_intra_sub_chunk` | ['BT', 'BC'] | H, K, BT, BC, BK, IS_VARLEN, USE_GATHER | ['T'] |
| kernels/ops/attention/fla/chunk_intra_token_parallel.py:28 `chunk_kda_fwd_kernel_intra_token_parallel` | ['K', 'H'] | H, K, BT, BC, BK, BH, IS_VARLEN | ['T', 'N'] |
| kernels/ops/attention/fla/chunk_o.py:30 `chunk_fwd_kernel_o` | none | H, Hg, K, V, BT, BK, BV, USE_G, IS_VARLEN | ['T'] |
| kernels/ops/attention/fla/cumsum.py:22 `chunk_local_cumsum_scalar_kernel` | none | B, H, BT, REVERSE, HAS_SCALE, IS_VARLEN, HEAD_FIRST | ['T'] |
| kernels/ops/attention/fla/cumsum.py:80 `chunk_local_cumsum_vector_kernel` | ['B', 'H', 'S', 'BT', 'IS_VARLEN', 'REVERSE', 'HAS_SCALE'] | B, H, S, BT, BS, REVERSE, HAS_SCALE, IS_VARLEN, HEAD_FIRST | ['T'] |
| kernels/ops/attention/fla/fused_gdn_gating.py:11 `fused_gdn_gating_kernel` | none | NUM_HEADS, beta, threshold, BLK_HEADS |  |
| kernels/ops/attention/fla/fused_kda_conv_recurrent_verify.py:41 `fused_kda_conv_gating_verify_kernel` | none | T, W, H, HV, K, V, BK, BV, HAS_BIAS, USE_QK_L2NORM_IN_KERNEL, USE_LOWER_BOUND, SAVE_INTERMEDIATE_WINDOW, CACHE_INTERMEDIATE_STATES, USE_GDC |  |
| kernels/ops/attention/fla/fused_norm_gate.py:27 `layer_norm_gated_fwd_kernel` | none | D, BT, BD, ACTIVATION, IS_RMS_NORM, STORE_RESIDUAL_OUT, HAS_RESIDUAL, HAS_WEIGHT, HAS_BIAS, USE_GDC |  |
| kernels/ops/attention/fla/fused_norm_gate.py:116 `layer_norm_gated_fwd_kernel1` | none | D, BD, ACTIVATION, IS_RMS_NORM, STORE_RESIDUAL_OUT, HAS_RESIDUAL, HAS_WEIGHT, HAS_BIAS |  |
| kernels/ops/attention/fla/fused_recurrent.py:16 `fused_recurrent_gated_delta_rule_fwd_kernel` | none | B, H, HV, K, V, BK, BV, USE_INITIAL_STATE, STORE_FINAL_STATE, IS_BETA_HEADWISE, USE_QK_L2NORM_IN_KERNEL, IS_VARLEN, IS_KDA | ['T'] |
| kernels/ops/attention/fla/fused_recurrent.py:186 `fused_recurrent_gated_delta_rule_packed_decode_kernel` | none | stride_mixed_qkv_tok, stride_a_tok, stride_b_tok, stride_init_state_token, stride_final_state_token, stride_indices_seq, H, HV, K, V, BK, BV, SOFTPLUS_THRESHOLD, USE_QK_L2NORM_IN_KERNEL |  |
| kernels/ops/attention/fla/fused_recurrent.py:406 `fused_recurrent_kda_packed_decode_kernel` | none | stride_mixed_qkv_tok, stride_a_tok, stride_b_tok, stride_init_state_token, stride_final_state_token, stride_indices_seq, H, HV, K, V, BK, BV, SOFTPLUS_THRESHOLD, USE_QK_L2NORM_IN_KERNEL, USE_LOWER_BOUND |  |
| kernels/ops/attention/fla/fused_recurrent.py:870 `fused_recurrent_gated_delta_rule_update_fwd_kernel` | none | stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token, NP2_T, B, H, HV, K, V, BK, BV, USE_INITIAL_STATE, IS_BETA_HEADWISE, USE_QK_L2NORM_IN_KERNEL, IS_VARLEN, DISABLE_STATE_UPDATE, DISABLE_OUTPUT_CALCULATION, CACHE_INTERMEDIATE_STATES, HAS_EAGLE_TREE_CUSTOM_ATTN_MASK | ['T'] |
| kernels/ops/attention/fla/fused_recurrent_linear_replayssm.py:62 `fused_recurrent_linear_replayssm_decode_kernel` | none | stride_mixed_qkv_tok, stride_a_tok, stride_b_tok, stride_init_state_token, stride_final_state_token, stride_indices_seq, H, HV, K, V, BK, BV, BC, NK, BKT, MAX_CACHE_LEN, SOFTPLUS_THRESHOLD, USE_QK_L2NORM_IN_KERNEL, HAS_FORCE_FLUSH, IS_KDA |  |
| kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py:11 `fused_sigmoid_gating_delta_rule_update_kernel` | none | stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token, NP2_T, B, H, HV, K, V, BK, BV, USE_INITIAL_STATE, USE_QK_L2NORM_IN_KERNEL, IS_VARLEN, IS_KDA, USE_LOWER_BOUND, DISABLE_STATE_UPDATE, CACHE_INTERMEDIATE_STATES, HAS_EAGLE_TREE_CUSTOM_ATTN_MASK, stride_rawv_slot, stride_rawk_slot, stride_g_slot, stride_beta_slot, MAX_CACHE_LEN, CACHE_RING, SPLIT_N_HV_GRID, USE_GDC | ['T'] |
| kernels/ops/attention/fla/gdn_replayssm_spec_decode.py:65 `gdn_replayssm_spec_circular_kernel` | none | stride_q_t, stride_k_t, stride_v_t, stride_a_t, stride_b_t, stride_o_t, stride_state_slot, stride_d_slot, stride_k_slot, stride_g_slot, stride_rawv_slot, stride_rawk_slot, stride_beta_slot, stride_qsl, stride_indices, H, HV, K, V, BK, BV, BS, BC, NK, BKT, MAX_CACHE_LEN, SOFTPLUS_THRESHOLD, USE_QK_L2NORM_IN_KERNEL, IS_FLUSH, NULL_BLOCK_ID, DOT_PRECISION |  |
| kernels/ops/attention/fla/gdn_replayssm_spec_decode.py:440 `gdn_replayssm_exact_fold_kernel` | none | stride_state_slot, stride_rawv_slot, stride_rawk_slot, stride_g_slot, stride_beta_slot, stride_indices, H, HV, K, V, BK, BV, MAX_CACHE_LEN, USE_QK_L2NORM_IN_KERNEL, NULL_BLOCK_ID |  |
| kernels/ops/attention/fla/gdn_replayssm_spec_decode.py:552 `_advance_gdn_spec_cursors_kernel` | none | stride_sbi, stride_na, MAX_CACHE_LEN, MAX_SPEC_LEN, CACHE_BUF_LEN, BLOCK, NULL_BLOCK_ID |  |
| kernels/ops/attention/fla/gdn_replayssm_spec_decode.py:609 `_reset_gdn_replayssm_spec_cursors_kernel` | none | stride_sbi, stride_reset, INIT_FLUSH, BLOCK, NULL_BLOCK_ID |  |
| kernels/ops/attention/fla/gdn_replayssm_spec_fold.py:19 `gdn_replayssm_exact_fold_kernel` | none | stride_state_slot, stride_rawv_slot, stride_rawk_slot, stride_g_slot, stride_beta_slot, stride_state_layer, stride_rawv_layer, stride_rawk_layer, stride_g_layer, stride_beta_layer, stride_indices, stride_accept, stride_track, stride_steps, H, HV, K, V, BK, BV, MAX_CACHE_LEN, USE_QK_L2NORM_IN_KERNEL, NULL_BLOCK_ID, HAS_TRACK |  |
| kernels/ops/attention/fla/kda.py:225 `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` | ['BC', 'IS_VARLEN'] | H, K, BT, BC, BK, NC, IS_VARLEN | ['T'] |
| kernels/ops/attention/fla/kda.py:334 `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra` | ['BK', 'BT', 'IS_VARLEN'] | H, K, BT, BC, BK, IS_VARLEN | ['T'] |
| kernels/ops/attention/fla/kda.py:521 `_recompute_w_u_fwd_kernel` | ['H', 'K', 'V', 'BT', 'IS_VARLEN'] | H, K, V, BT, BK, BV, STORE_KG, IS_VARLEN, DOT_PRECISION | ['T'] |
| kernels/ops/attention/fla/kda.py:762 `chunk_gla_fwd_kernel_o` | ['BT', 'IS_VARLEN'] | H, K, V, BT, BK, BV, IS_VARLEN | ['T'] |
| kernels/ops/attention/fla/kda.py:913 `softplus_fwd` | none |  |  |
| kernels/ops/attention/fla/kda.py:935 `kda_gate_chunk_cumsum_vector_kernel` | ['H', 'S', 'BT', 'IS_VARLEN'] | H, S, BT, BS, HAS_BIAS, HAS_SCALE, IS_VARLEN, USE_LOWER_BOUND | ['T'] |
| kernels/ops/attention/fla/kda_replayssm_spec_decode.py:40 `kda_replayssm_exact_fold_kernel` | none | stride_state_slot, stride_rawv_slot, stride_rawk_slot, stride_gk_slot, stride_beta_slot, stride_state_layer, stride_rawv_layer, stride_rawk_layer, stride_gk_layer, stride_beta_layer, stride_indices, stride_accept, stride_track, stride_steps, H, HV, K, V, BK, BV, MAX_CACHE_LEN, USE_QK_L2NORM_IN_KERNEL, NULL_BLOCK_ID, HAS_TRACK |  |
| kernels/ops/attention/fla/kda_snapshot.py:11 `_store_conv` | none | raw_s0, raw_s1, state_s0, state_s1, state_s2, WIDTH, HISTORY, BLOCK |  |
| kernels/ops/attention/fla/l2norm.py:24 `l2norm_fwd_kernel1` | none | BD |  |
| kernels/ops/attention/fla/l2norm.py:55 `l2norm_fwd_kernel` | none | D, BT, BD | ['T'] |
| kernels/ops/attention/fla/layernorm_gated.py:75 `_layer_norm_fwd_1pass_kernel` | none | N, BLOCK_N, ROWS_PER_BLOCK, HAS_BIAS, HAS_Z, NORM_BEFORE_GATE, IS_RMS_NORM, ACTIVATION, USE_GDC |  |
| kernels/ops/attention/fla/op.py:26 `safe_exp` | none |  |  |
| kernels/ops/attention/fla/op.py:33 `gather` | none |  |  |
| kernels/ops/attention/fla/op.py:59 `make_tensor_descriptor` | none |  |  |
| kernels/ops/attention/fla/wy_fast.py:23 `recompute_w_u_fwd_kernel` | none | H, Hg, K, V, BT, BK, BV, IS_VARLEN | ['T'] |
| kernels/ops/mamba/causal_conv1d_triton.py:19 `_causal_conv1d_fwd_kernel` | none | dim, num_cache_lines, stride_x_seq, stride_x_dim, stride_x_token, stride_w_dim, stride_w_width, stride_istate_seq, stride_istate_dim, stride_istate_token, stride_o_seq, stride_o_dim, stride_o_token, pad_slot_id, HAS_BIAS, KERNEL_WIDTH, SILU_ACTIVATION, HAS_INITIAL_STATES, HAS_CACHE, IS_CONTINUOUS_BATCHING, USE_PAD_SLOT, NP2_STATELEN, BLOCK_M, BLOCK_N |  |
| kernels/ops/mamba/causal_conv1d_triton.py:581 `_causal_conv1d_update_kernel` | none | dim, seqlen, state_len, num_cache_lines, stride_x_seq, stride_x_dim, stride_x_token, stride_w_dim, stride_w_width, stride_conv_state_seq, stride_conv_state_dim, stride_conv_state_tok, stride_state_indices, stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win, stride_intermediate_state_indices, stride_retrieve_next_token_seq, stride_retrieve_next_token_token, stride_retrieve_next_sibling_seq, stride_retrieve_next_sibling_token, stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token, stride_o_seq, stride_o_dim, stride_o_token, pad_slot_id, HAS_BIAS, KERNEL_WIDTH, SILU_ACTIVATION, IS_CONTINUOUS_BATCHING, IS_SPEC_DECODING, NP2_STATELEN, NP2_SEQLEN, USE_PAD_SLOT, BLOCK_N, SAVE_INTERMEDIATE, HAS_EAGLE_TREE_CUSTOM_ATTN_MASK, USE_GDC |  |
| kernels/ops/mamba/mamba_state_indices_triton.py:23 `_fused_replay_state_indices_kernel` | none | BS_UPPER |  |
| kernels/ops/mamba/mamba_state_scatter_triton.py:34 `track_mamba_state_if_needed_kernel` | none | conv_state_numel_per_row, ssm_state_numel_per_row, BLOCK_SIZE, check_freed_slots |  |
| kernels/ops/mamba/mamba_state_scatter_triton.py:148 `_fused_mamba_state_scatter_with_mask_kernel` | none | elem_per_entry, BLOCK_SIZE |  |
| kernels/ops/mamba/mamba_state_scatter_triton.py:322 `_fused_conv_window_scatter_with_mask_kernel` | none | elem_per_entry, KM1, BLOCK_SIZE |  |
| kernels/ops/mamba/mamba_state_scatter_triton.py:478 `_fused_conv_window_scatter_multi_kernel` | none | NUM_TYPES, META_COLS, BLOCK_SIZE |  |
| kernels/ops/mamba/mamba_state_scatter_triton.py:736 `_fused_commit_track_indices_kernel` | none | HAS_TRACK |  |
| kernels/ops/mamba/mamba_state_scatter_triton.py:795 `track_mamba_states_all_layers_kernel` | none | conv_state_numel_per_row, ssm_state_numel_per_row, BLOCK_SIZE, check_freed_slots |  |
| kernels/ops/mamba/triton_ops/mamba_ssm.py:36 `convert_rs_fp16x2` | none |  |  |
| kernels/ops/mamba/triton_ops/mamba_ssm.py:85 `_selective_scan_update_kernel` | none | DT_SOFTPLUS, TIE_HDIM, BLOCK_SIZE_M, HAS_DT_BIAS, HAS_D, HAS_Z, HAS_STATE_BATCH_INDICES, DISABLE_STATE_UPDATE, CACHE_INTERMEDIATE_STATES, HAS_EAGLE_TREE_CUSTOM_ATTN_MASK, HAS_INTERMEDIATE_STATE_INDICES, BLOCK_SIZE_DSTATE, USE_RS_ROUNDING, PHILOX_ROUNDS, USE_GDC | ['T'] |
| kernels/ops/mamba/triton_ops/mamba_ssm.py:23 `softplus` | none |  |  |
| kernels/ops/mamba/triton_ops/mamba_ssm.py:30 `softplus` | none |  |  |
| kernels/ops/mamba/triton_ops/ssd_bmm.py:19 `_bmm_chunk_fwd_kernel` | none | IS_CAUSAL, dot_dtype, HAS_SEQ_IDX, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K |  |
| kernels/ops/mamba/triton_ops/ssd_chunk_scan.py:20 `_chunk_scan_fwd_kernel` | none | IS_CAUSAL, HAS_D, D_HAS_HDIM, HAS_Z, HAS_SEQ_IDX, BLOCK_SIZE_DSTATE, IS_TRITON_22, HAS_INITSTATES, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K |  |
| kernels/ops/mamba/triton_ops/ssd_chunk_state.py:21 `_chunk_cumsum_fwd_kernel` | none | DT_SOFTPLUS, HAS_DT_BIAS, BLOCK_SIZE_CHUNK, BLOCK_SIZE_H |  |
| kernels/ops/mamba/triton_ops/ssd_chunk_state.py:113 `_chunk_state_fwd_kernel` | none | HAS_SEQ_IDX, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K |  |
| kernels/ops/mamba/triton_ops/ssd_chunk_state.py:263 `_chunk_state_varlen_kernel` | none | HAS_INITSTATES, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K |  |
| kernels/ops/mamba/triton_ops/ssd_state_passing.py:17 `_state_passing_fwd_kernel` | none | HAS_INITSTATES, HAS_SEQ_IDX, IS_CONT_BATCHED, BLOCK_SIZE |  |
| srt/layers/attention/dsa/sm80_indexer_kernels.py:12 `_e4m3_to_bf16` | none |  |  |
| srt/layers/attention/dsa/sm80_indexer_kernels.py:24 `_paged` | none | N, H, D, P, S, PAGE, QB, QN, QH, QD, WB, WH, CB, CN, TB, TP, KB, HH, DD |  |
| srt/layers/attention/dsa/sm80_indexer_kernels.py:60 `_ragged` | none | NQ, NK, H, D, QQ, QH, QD, KK, KD, SS, WQ, WH, KSS, KES, CLEAN, BQ, BK, HH, DD |  |
| srt/layers/attention/dsa/sm80_indexer_kernels.py:155 `_unpack_prefill` | none | R, H, D, XR, XH, XD, BLOCK |  |
| srt/layers/attention/dsa/sm80_indexer_kernels.py:165 `_prefill` | none | NQ, NK, H, QQ, QH, QD, KK, KD, SS, WQ, WH, KSS, KES, CLEAN, BQ, BK, HH, GROUP, LOOP |  |
