"""
Server-side aggregation for FedSpaLLM (Bai et al., NAACL 2025; arXiv:2410.14852).

Implements:
  - Eq. (5): l0-norm aggregation — per parameter index j,
        W_g[j] = sum_i ( M_i[j] * W_i[j] ) / sum_i M_i[j],
    with M_i[j] in {0,1} (1 = retained, 0 = pruned). If sum_i M_i[j] == 0, W_g[j] = 0.
  - Eq. (6)-(7) & Sec. 4.2: adaptive mask expansion to reach target unstructured sparsity s:
        C_j = sum_i I(M_i[j]=0),  k = ceil(s * d) - sum_j I(C_j = N),
    then prune (set to zero) k additional positions among those not pruned by all clients,
    choosing indices with largest C_j first.

Mask expansion is only applied for unstructured sparsity (caller should disable when using N:M).

Reference: https://arxiv.org/abs/2410.14852
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch


def _l0_norm_aggregate_tensor(
    weights: Sequence[torch.Tensor], masks: Sequence[torch.Tensor]
) -> torch.Tensor:
    """
    FedSpaLLM Eq.(5) l0-norm aggregation.

    For each parameter index j:

        W_g[j] =
            sum_i ( M_i[j] * W_i[j] )
            -------------------------
              sum_i M_i[j]

    where:

        M_i[j] = 1:
            parameter is retained by client i

        M_i[j] = 0:
            parameter is pruned by client i


    If all clients prune the same parameter:

        sum_i M_i[j] == 0

    then:

        W_g[j] = 0


    This keeps the original function interface unchanged.
    """

    if len(weights) != len(masks):
        raise ValueError("weights and masks must have same length")

    if len(weights) == 0:
        raise ValueError("empty client list")


    # reference tensor
    w0 = weights[0]

    device = w0.device
    dtype = w0.dtype


    # numerator:
    #
    # Σ(M_i[j] * W_i[j])
    #
    numerator = torch.zeros_like(
        w0,
        dtype=torch.float32,
        device=device
    )


    # denominator:
    #
    # ΣM_i[j]
    #
    denominator = torch.zeros_like(
        w0,
        dtype=torch.float32,
        device=device
    )


    for w, m in zip(weights, masks):

        weight = w.to(
            device=device,
            dtype=torch.float32
        )

        mask = m.to(
            device=device,
            dtype=torch.float32
        )


        # accumulate retained weights
        numerator += weight * mask


        # accumulate number of retained clients
        denominator += mask



    # initialize output
    out = torch.zeros_like(
        numerator,
        dtype=torch.float32,
        device=device
    )


    # positions retained by at least one client
    valid = denominator > 0


    # FedSpaLLM Eq.(5)
    out[valid] = (
        numerator[valid]
        /
        denominator[valid]
    )


    # if denominator == 0:
    #
    # all clients prune this position
    #
    # keep zero


    return out.to(dtype=dtype)


def _adaptive_mask_expansion_tensor(
    W_agg: torch.Tensor,
    masks: Sequence[torch.Tensor],
    target_sparsity: float,
) -> torch.Tensor:
    """
    Sec. 4.2 adaptive mask expansion after l0 aggregation.
    masks: per client, 1 = retained, 0 = pruned.
    target_sparsity: fraction of entries that should be zero (same meaning as --sparsity in scripts).
    """
    if not (0.0 <= target_sparsity <= 1.0):
        raise ValueError("target_sparsity must be in [0, 1]")
    N = len(masks)
    if N == 0:
        return W_agg
    device = W_agg.device
    dtype = W_agg.dtype
    flat_w = W_agg.to(dtype=torch.float32, device=device).flatten()
    d = flat_w.numel()
    if d == 0:
        return W_agg

    # C_j = number of clients that pruned position j (Eq. 6)
    C = torch.zeros(d, device=device, dtype=torch.float32)
    for m in masks:
        mf = m.to(device=device, dtype=torch.float32).flatten()
        C += 1.0 - mf

    desired_zeros = int(math.ceil(float(d) * float(target_sparsity)))
    all_pruned_count = int((C >= float(N)).sum().item())
    k = desired_zeros - all_pruned_count
    if k <= 0:
        return W_agg

    candidate = C < float(N)
    if not bool(candidate.any()):
        return W_agg

    idx = torch.nonzero(candidate, as_tuple=False).squeeze(1)
    scores = C[idx]
    order = torch.argsort(scores, descending=True)
    take = min(k, idx.numel())
    chosen = idx[order[:take]]
    flat_out = flat_w.clone()
    flat_out[chosen] = 0.0
    return flat_out.view_as(W_agg).to(dtype=dtype)


def fedspallm_aggregate_state_dicts(
    reference: Dict[str, torch.Tensor],
    client_state_dicts: List[Dict[str, torch.Tensor]],
    client_masks: List[Dict[str, torch.Tensor]],
    target_sparsity: float,
    apply_mask_expansion: bool,
) -> Dict[str, torch.Tensor]:
    """
    Full-model aggregation: same keys as `reference`.
    Non-floating tensors are copied from the first client (unchanged across FedSpaLLM in this codebase).
    """
    if len(client_state_dicts) != len(client_masks):
        raise ValueError("client_state_dicts and client_masks length mismatch")
    if len(client_state_dicts) == 0:
        raise ValueError("no clients to aggregate")

    out: Dict[str, torch.Tensor] = {}
    for name, ref_t in reference.items():
        if name not in client_state_dicts[0]:
            raise KeyError(f"missing key {name} in client 0 state_dict")
        tensors = [sd[name] for sd in client_state_dicts]
        masks = [md[name] for md in client_masks]

        if not ref_t.is_floating_point():
            out[name] = tensors[0].clone()
            continue

        W_agg = _l0_norm_aggregate_tensor(tensors, masks)
        if apply_mask_expansion and target_sparsity > 0.0:
            W_agg = _adaptive_mask_expansion_tensor(W_agg, masks, target_sparsity)
        out[name] = W_agg.to(device=ref_t.device, dtype=ref_t.dtype)

    return out


def build_client_masks_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """M_i[j]=1 if weight retained (non-zero), 0 if pruned. Same convention as paper Sec. 4.1."""
    masks: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if v.is_floating_point():
            masks[k] = (v != 0).to(dtype=v.dtype)
        else:
            masks[k] = torch.ones_like(v, dtype=torch.float32)
    return masks


def _build_full_client_masks_from_layer_report(
    reference_state_dict: Dict[str, torch.Tensor],
    layer_ids: Sequence[int],
    sparse_param_masks: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Expand a sparse pruning report into the full mask format expected by
    `fedspallm_aggregate_state_dicts`.

    Client report format:
      - `layer_ids`: which layer indices the client actually pruned
      - `sparse_param_masks`: {param_name: mask_tensor} only for selected layers

    Expansion rule:
      - for params in selected layers: use provided mask if present, else all-ones mask
      - for params not in selected layers: all-ones mask
    """
    layer_prefixes = [f"model.decoder.layers.{i}." for i in layer_ids]
    def belongs_to_selected_layer(param_name: str) -> bool:
        return any(param_name.startswith(prefix) for prefix in layer_prefixes)

    full_masks: Dict[str, torch.Tensor] = {}
    for name, ref_t in reference_state_dict.items():
        if ref_t.is_floating_point():
            full_masks[name] = torch.ones_like(ref_t, dtype=ref_t.dtype)
            if belongs_to_selected_layer(name) and name in sparse_param_masks:
                full_masks[name] = sparse_param_masks[name].to(dtype=ref_t.dtype)
        else:
            # Non-floating tensors are ignored by the L0 aggregation path,
            # but we keep masks for compatibility with the original implementation.
            full_masks[name] = torch.ones_like(ref_t, dtype=torch.float32)
    return full_masks


def fedspallm_aggregate_state_dicts_from_layer_reports(
    reference: Dict[str, torch.Tensor],
    client_state_dicts: List[Dict[str, torch.Tensor]],
    client_layer_reports: List[Dict[str, object]],
    target_sparsity: float,
    apply_mask_expansion: bool,
) -> Dict[str, torch.Tensor]:
    """
    Modified: For each parameter, only aggregate using clients that were assigned to prune its layer.
    """
    if len(client_state_dicts) != len(client_layer_reports):
        raise ValueError("client_state_dicts and client_layer_reports length mismatch")
    if len(client_state_dicts) == 0:
        raise ValueError("no clients to aggregate")

    client_masks: List[Dict[str, torch.Tensor]] = []
    for report in client_layer_reports:
        layer_ids = report.get("layer_ids", [])
        sparse_masks = report.get("masks", {})
        if not isinstance(layer_ids, Sequence):
            raise TypeError("report['layer_ids'] must be a sequence of ints")
        if not isinstance(sparse_masks, dict):
            raise TypeError("report['masks'] must be a dict {param_name: mask_tensor}")
        client_masks.append(
            _build_full_client_masks_from_layer_report(
                reference_state_dict=reference,
                layer_ids=layer_ids,
                sparse_param_masks=sparse_masks,
            )
        )

    out: Dict[str, torch.Tensor] = {}
    for name, ref_t in reference.items():
        if name not in client_state_dicts[0]:
            raise KeyError(f"missing key {name} in client 0 state_dict")

        # Find which clients are responsible for this parameter's layer
        responsible_clients = []
        for client_id, report in enumerate(client_layer_reports):
            layer_ids = report.get("layer_ids", [])
            # Check if name belongs to any of the client's selected layers
            for lid in layer_ids:
                if name.startswith(f"model.decoder.layers.{lid}."):
                    responsible_clients.append(client_id)
                    break

        if not responsible_clients:
            # If no client pruned this parameter's layer, keep the reference value unchanged.
            out[name] = ref_t.clone()
            continue

        # Aggregate only using responsible clients for this parameter's layer.
        tensors = [client_state_dicts[client_id][name] for client_id in responsible_clients]
        masks = [client_masks[client_id][name] for client_id in responsible_clients]

        if not ref_t.is_floating_point():
            out[name] = tensors[0].clone()
            continue

        W_agg = _l0_norm_aggregate_tensor(tensors, masks)
        if apply_mask_expansion and target_sparsity > 0.0:
            W_agg = _adaptive_mask_expansion_tensor(W_agg, masks, target_sparsity)
        out[name] = W_agg.to(device=ref_t.device, dtype=ref_t.dtype)

    return out
