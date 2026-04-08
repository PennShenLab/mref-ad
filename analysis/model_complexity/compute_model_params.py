#!/usr/bin/env python3
"""Compute parameter counts for models used in this repo.

Usage examples:
  # Logistic Regression
  python scripts/compute_model_params.py --model logreg --d_in 128 --n_classes 3 --out results/model_param_counts_logreg

  # SVM
  python scripts/compute_model_params.py --model svm --d_in 128 --n_classes 3 --out results/model_param_counts_svm

  # XGBoost
  python scripts/compute_model_params.py --model xgb --model-path results/models/xgb_all/xgb_concat_all/fold0/fold0.joblib --out results/model_param_counts_xgb_fold0

  # Random Forest
  python scripts/compute_model_params.py --model rf --model-path results/models/rf_all/rf_concat_all/fold0/fold0.joblib --out results/model_param_counts_rf_fold0

  # MLP seed 7
  python scripts/compute_model_params.py --model mlp --d_in 128 --hidden 507 --n_classes 3 

  # FT-Transformer (requires rtdl_revisiting_models installed)
  python scripts/compute_model_params.py --model ftt --d_num 128 --n_classes 3 --out results/model_param_counts_ftt

  # MoE: provide a training meta.pkl produced by train_moe (contains 'groups') seed 1234
  python scripts/compute_model_params.py --model moe --meta results/missingness/pet_missing/p1p00/folds/fold0/fold0.meta.pkl --hidden_exp 84 --hidden_gate 183 --n_classes 3

  # seed 7
  python scripts/compute_model_params.py --model moe --meta results/missingness/pet_missing/p1p00/folds/fold0/fold0.meta.pkl --hidden_exp 145 --hidden_gate 91 --n_classes 3

  # optionally add --topk 2 to compute when running with top-2 experts


The script prints total and trainable parameter counts and a breakdown per-module.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import OrderedDict
from typing import List

import torch

# ensure repository root is on sys.path so imports like `baselines.*` work when
# running the script directly from `scripts/` or the repo root.
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def breakdown_params(model: torch.nn.Module):
    out = OrderedDict()
    for name, module in model.named_children():
        t, tr = count_params(module)
        out[name] = (t, tr)
    return out


def count_active_params_moe(model, topk=None):
    """
    Calculate active parameters for MoE models.
    
    For MoE with topk experts:
    - Gate always active (all gate params)
    - Only topk experts are active per sample
    
    Returns:
        active_params: Number of parameters used per forward pass
        total_params: Total model parameters
        breakdown: Dictionary with detailed breakdown
    """
    total, trainable = count_params(model)
    
    # Get gate and expert parameters
    gate_params = 0
    expert_params_each = 0
    num_experts = 0
    
    if hasattr(model, 'gate'):
        gate_params, _ = count_params(model.gate)
    
    if hasattr(model, 'experts'):
        num_experts = len(model.experts)
        if num_experts > 0:
            expert_params_each, _ = count_params(model.experts[0])
    
    # Determine number of active experts
    if topk is None and hasattr(model, 'topk'):
        topk = model.topk
    
    if topk is None or topk >= num_experts:
        # All experts active (soft routing)
        active_experts = num_experts
    else:
        # Only topk experts active
        active_experts = topk
    
    active_params = gate_params + (active_experts * expert_params_each)
    
    breakdown = {
        "total_params": total,
        "gate_params": gate_params,
        "expert_params_each": expert_params_each,
        "num_experts": num_experts,
        "active_experts": active_experts,
        "active_params": active_params,
        "param_efficiency": f"{(active_params / total * 100):.2f}%" if total > 0 else "N/A"
    }
    
    return active_params, total, breakdown


def count_flops_linear(in_features, out_features, bias=True):
    """Count FLOPs for a linear layer: 2*in*out (MAC) + bias"""
    flops = 2 * in_features * out_features
    if bias:
        flops += out_features
    return flops


def count_flops_mlp(model):
    """Count FLOPs for an MLP expert (3-layer network)."""
    flops = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            in_f = module.in_features
            out_f = module.out_features
            has_bias = module.bias is not None
            flops += count_flops_linear(in_f, out_f, has_bias)
    return flops


def count_flops_moe(model, batch_size=1, topk=None):
    """
    Calculate FLOPs for MoE forward pass.
    
    Args:
        model: MoE model
        batch_size: Batch size for computation
        topk: Number of active experts (if None, uses model.topk or all)
    
    Returns:
        total_flops: Total FLOPs per batch
        active_flops: FLOPs for active path only
        breakdown: Dictionary with detailed breakdown
    """
    # Gate FLOPs
    gate_flops = 0
    if hasattr(model, 'gate'):
        gate = model.gate
        # Projection layers: M x (d_i -> 32)
        if hasattr(gate, 'proj'):
            for proj_layer in gate.proj:
                gate_flops += count_flops_mlp(proj_layer)
        # FC layers: (32*M + M) -> hidden -> M
        if hasattr(gate, 'fc'):
            gate_flops += count_flops_mlp(gate.fc)
    
    gate_flops *= batch_size
    
    # Expert FLOPs
    expert_flops_each = 0
    num_experts = 0
    
    if hasattr(model, 'experts'):
        num_experts = len(model.experts)
        if num_experts > 0:
            expert_flops_each = count_flops_mlp(model.experts[0])
    
    # Determine active experts
    if topk is None and hasattr(model, 'topk'):
        topk = model.topk
    
    if topk is None or topk >= num_experts:
        active_experts = num_experts
    else:
        active_experts = topk
    
    # Active expert FLOPs (only topk experts compute)
    active_expert_flops = active_experts * expert_flops_each * batch_size
    
    # Total active FLOPs
    active_flops = gate_flops + active_expert_flops
    
    # Total FLOPs if all experts were used
    total_expert_flops = num_experts * expert_flops_each * batch_size
    total_flops = gate_flops + total_expert_flops
    
    breakdown = {
        "gate_flops": gate_flops,
        "expert_flops_each": expert_flops_each,
        "num_experts": num_experts,
        "active_experts": active_experts,
        "active_expert_flops": active_expert_flops,
        "active_flops_total": active_flops,
        "total_flops_all_experts": total_flops,
        "flop_efficiency": f"{(active_flops / total_flops * 100):.2f}%" if total_flops > 0 else "N/A"
    }
    
    return active_flops, total_flops, breakdown


def count_flops_attention(d_model, n_heads, seq_len):
    """
    Count FLOPs for multi-head self-attention.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        seq_len: Sequence length
    
    Returns:
        Total FLOPs for attention computation
    """
    d_k = d_model // n_heads
    
    # Q, K, V projections: 3 * (seq_len * d_model * d_model)
    qkv_proj = 3 * seq_len * d_model * d_model * 2
    
    # Attention scores: Q @ K^T for each head
    # (seq_len * d_k * seq_len) * n_heads
    attn_scores = n_heads * seq_len * d_k * seq_len * 2
    
    # Softmax (approximate as 3 operations per element)
    softmax = n_heads * seq_len * seq_len * 3
    
    # Attention @ V: (seq_len * seq_len * d_k) * n_heads
    attn_output = n_heads * seq_len * seq_len * d_k * 2
    
    # Output projection: seq_len * d_model * d_model
    output_proj = seq_len * d_model * d_model * 2
    
    return qkv_proj + attn_scores + softmax + attn_output + output_proj


def count_flops_feedforward(d_model, d_ff, seq_len):
    """
    Count FLOPs for feedforward network in transformer.
    
    Args:
        d_model: Model dimension
        d_ff: Feedforward dimension
        seq_len: Sequence length
    
    Returns:
        Total FLOPs for feedforward computation
    """
    # First layer: seq_len * d_model * d_ff
    layer1 = seq_len * d_model * d_ff * 2
    
    # Activation (GELU/ReLU - approximate as 1 op per element)
    activation = seq_len * d_ff
    
    # Second layer: seq_len * d_ff * d_model
    layer2 = seq_len * d_ff * d_model * 2
    
    return layer1 + activation + layer2


def count_flops_layernorm(d_model, seq_len):
    """Count FLOPs for layer normalization."""
    # Mean: d_model ops, Variance: d_model ops, Normalize: d_model ops
    # Scale and shift: 2 * d_model ops
    return seq_len * (d_model * 5)


def count_flops_ftt(model, batch_size=1):
    """
    Calculate FLOPs for FT-Transformer forward pass.
    
    Args:
        model: FT-Transformer model
        batch_size: Batch size for computation
    
    Returns:
        total_flops: Total FLOPs per batch
        breakdown: Dictionary with detailed breakdown
    """
    flops = 0
    breakdown = {}
    
    # Try to extract model config
    d_token = getattr(model, 'd_token', 192)  # default from rtdl
    n_blocks = getattr(model, 'n_blocks', 3)
    attention_n_heads = getattr(model, 'attention_n_heads', 8)
    ffn_d_hidden = getattr(model, 'ffn_d_hidden', None)
    n_cont_features = getattr(model, 'n_cont_features', 0)
    d_out = getattr(model, 'd_out', 3)
    
    if ffn_d_hidden is None:
        ffn_d_hidden = d_token * 4  # typical default
    
    # Sequence length = n_features + 1 CLS token
    seq_len = n_cont_features + 1
    
    # Feature tokenization: Linear projections for each feature
    tokenization_flops = n_cont_features * d_token * 2
    breakdown['tokenization'] = tokenization_flops
    
    # Transformer blocks
    block_flops = 0
    for _ in range(n_blocks):
        # Layer norm before attention
        ln1 = count_flops_layernorm(d_token, seq_len)
        
        # Multi-head self-attention
        attn = count_flops_attention(d_token, attention_n_heads, seq_len)
        
        # Layer norm before feedforward
        ln2 = count_flops_layernorm(d_token, seq_len)
        
        # Feedforward network
        ffn = count_flops_feedforward(d_token, ffn_d_hidden, seq_len)
        
        block_flops += ln1 + attn + ln2 + ffn
    
    breakdown['transformer_blocks'] = block_flops
    breakdown['n_blocks'] = n_blocks
    breakdown['per_block'] = block_flops // n_blocks if n_blocks > 0 else 0
    
    # Output head: typically just use CLS token
    # Final layer norm + linear projection
    output_flops = count_flops_layernorm(d_token, 1) + d_token * d_out * 2
    breakdown['output_head'] = output_flops
    
    # Total FLOPs
    total_flops = (tokenization_flops + block_flops + output_flops) * batch_size
    breakdown['total_flops'] = total_flops
    breakdown['batch_size'] = batch_size
    
    return total_flops, breakdown


def build_mlp(args):
    from baselines.mlp import MLP

    model = MLP(d_in=args.d_in, hidden=args.hidden, drop=args.drop, n_classes=args.n_classes)
    return model


def build_ftt(args):
    try:
        from baselines.ftt import FTTransformer  # type: ignore
    except Exception:
        # baselines.ftt wraps rtdl import; try importing from the package directly
        try:
            from rtdl_revisiting_models import FTTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            print("FTTransformer not available: install rtdl_revisiting_models to build FTT model", file=sys.stderr)
            raise

    # use default kwargs if available
    try:
        kwargs = FTTransformer.get_default_kwargs()
    except Exception:
        kwargs = {}

    model = FTTransformer(n_cont_features=args.d_num, cat_cardinalities=None, d_out=args.n_classes, **kwargs)
    return model


def build_moe(args):
    # Try to obtain groups/dims from meta.pkl
    if args.meta is None:
        raise ValueError("--meta is required for moe to read saved groups mapping")
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    groups = meta.get("groups", None)
    if groups is None:
        raise ValueError(f"meta file {args.meta} does not contain 'groups' mapping")
    dims = [len(v) for v in groups.values()]

    # import MoE class
    try:
        # prefer the canonical train_moe implementation
        from scripts.train_moe import MoE
    except Exception:
        try:
            from scripts.train_moe_experts import MoE
        except Exception:
            raise

    model = MoE(dims, hidden_exp=args.hidden_exp, hidden_gate=args.hidden_gate, n_classes=args.n_classes, drop=args.drop, gate_type=args.gate_type, gumbel_hard=args.gumbel_hard, gate_noise=args.gate_noise, topk=args.topk)
    return model


def _flex_modality_groups(groups, selected_letters: str):
    selected_letters = selected_letters.upper()
    mapping = {"A": "amy", "M": "mri", "D": "demographic"}
    want = {mapping[c] for c in selected_letters if c in mapping}
    out = {k: [] for k in ["amy", "mri", "demographic"] if k in want}
    for expert, cols in groups.items():
        name = expert.lower()
        if name.startswith("amy_") and "amy" in out:
            out["amy"].extend(cols)
        elif name.startswith("mri_") and "mri" in out:
            out["mri"].extend(cols)
        elif name == "demographic" and "demographic" in out:
            out["demographic"].extend(cols)
    out = {k: list(dict.fromkeys(v)) for k, v in out.items() if len(v) > 0}
    if len(out) == 0:
        raise ValueError("No modality columns found for requested --modality")
    return out


def _load_flex_best_params(args):
    if not args.best_params_json:
        return
    with open(args.best_params_json, "r") as f:
        payload = json.load(f)
    p = payload.get("best_params", {})
    if not p:
        raise ValueError(f"No 'best_params' found in {args.best_params_json}")
    for k in [
        "hidden_dim",
        "top_k",
        "num_patches",
        "num_experts",
        "num_layers_fus",
        "num_layers_pred",
        "num_heads",
        "dropout",
        "num_routers",
    ]:
        if k in p:
            setattr(args, k, p[k])


class FlexMoEComplexityModel(torch.nn.Module):
    def __init__(self, encoder_dict, fusion_model, missing_embeds):
        super().__init__()
        self.encoder_dict = encoder_dict
        self.fusion_model = fusion_model
        self.missing_embeds = missing_embeds


def build_flex_moe(args):
    _load_flex_best_params(args)
    if args.experts_config is None:
        raise ValueError("--experts_config is required for flex_moe")

    # train_flex_moe expects "utils" to resolve to scripts/utils.py
    script_dir = os.path.join(_REPO_ROOT, "scripts")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import utils  # type: ignore

    df, groups, _ = utils.load_experts_from_yaml(args.experts_config)
    mod_cols = _flex_modality_groups(groups, args.modality)
    mod_keys = list(mod_cols.keys())
    num_modalities = len(mod_keys)

    flex_root = os.path.abspath(args.flex_moe_root)
    if flex_root not in sys.path:
        sys.path.insert(0, flex_root)
    from models import FlexMoE, PatchEmbeddings  # type: ignore

    encoder_dict = torch.nn.ModuleDict({
        m: PatchEmbeddings(len([c for c in mod_cols[m] if c in df.columns]), int(args.num_patches), int(args.hidden_dim))
        for m in mod_keys
    })
    full_modality_index = (2 ** num_modalities) - 2
    fusion_model = FlexMoE(
        num_modalities=num_modalities,
        full_modality_index=full_modality_index,
        num_patches=int(args.num_patches),
        hidden_dim=int(args.hidden_dim),
        output_dim=args.n_classes,
        num_layers=int(args.num_layers_fus),
        num_layers_pred=int(args.num_layers_pred),
        num_experts=int(args.num_experts),
        num_routers=int(args.num_routers),
        top_k=int(args.top_k),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
    )
    missing_embeds = torch.nn.Parameter(
        torch.randn((2 ** num_modalities) - 1, num_modalities, int(args.num_patches), int(args.hidden_dim))
    )
    return FlexMoEComplexityModel(encoder_dict, fusion_model, missing_embeds)


def count_active_params_flex_moe(model, topk):
    total_params, _ = count_params(model)
    sparse_expert_total = 0
    sparse_expert_active = 0
    sparse_layers = 0
    experts_per_sparse_layer = 0
    expert_params_each = 0

    if not hasattr(model, "fusion_model"):
        return total_params, {
            "total_params": total_params,
            "active_params": total_params,
            "sparse_layers": 0,
            "num_experts": 0,
            "expert_params_each": 0,
            "active_experts_per_sparse_layer": 0,
            "param_efficiency": "100.00%",
        }

    for layer in getattr(model.fusion_model, "network", []):
        if hasattr(layer, "mlp_sparse") and layer.mlp_sparse and hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            sparse_layers += 1
            expert_total, _ = count_params(layer.mlp.experts)
            sparse_expert_total += expert_total
            if experts_per_sparse_layer == 0:
                experts_per_sparse_layer = int(getattr(layer.mlp, "num_expert", 0))
            if experts_per_sparse_layer > 0:
                expert_params_each = expert_total // experts_per_sparse_layer
                active_experts = min(int(topk), experts_per_sparse_layer)
                sparse_expert_active += active_experts * expert_params_each
            else:
                sparse_expert_active += expert_total

    active_params = total_params - sparse_expert_total + sparse_expert_active
    breakdown = {
        "total_params": int(total_params),
        "active_params": int(active_params),
        "sparse_layers": int(sparse_layers),
        "num_experts": int(experts_per_sparse_layer),
        "expert_params_each": int(expert_params_each),
        "active_experts_per_sparse_layer": int(min(int(topk), experts_per_sparse_layer) if experts_per_sparse_layer > 0 else 0),
        "param_efficiency": f"{(active_params / total_params * 100):.2f}%" if total_params > 0 else "N/A",
    }
    return int(active_params), breakdown


# -----------------------------
# Classical / tree-based counting helpers
# -----------------------------
def _load_joblib(path: str):
    try:
        import joblib
    except Exception:
        raise RuntimeError("joblib is required to load sklearn models; pip install joblib")
    return joblib.load(path)


def _unwrap_model(obj):
    """If the loaded object is a wrapper (SklearnFitResult) or has a .model
    attribute, return the inner model; otherwise return the object itself.
    """
    if obj is None:
        return None
    # common wrapper in this repo: SklearnFitResult with .model attribute
    if hasattr(obj, "model"):
        return getattr(obj, "model")
    return obj


def count_logreg(args):
    # If a saved model is provided, load and count coef_/intercept_. Otherwise fall back to shape-based count
    if args.model_path:
        m = _load_joblib(args.model_path)
        m = _unwrap_model(m)
        coef = getattr(m, "coef_", None)
        intercept = getattr(m, "intercept_", None)
        total = int((coef.size if coef is not None else 0) + (intercept.size if intercept is not None else 0))
        breakdown = {"coef": {"total": int(coef.size) if coef is not None else 0, "trainable": int(coef.size) if coef is not None else 0}}
        if intercept is not None:
            breakdown["intercept"] = {"total": int(intercept.size), "trainable": int(intercept.size)}
        return total, total, breakdown
    else:
        # require d_in and n_classes
        d = args.d_in
        K = args.n_classes
        total = K * d + K
        breakdown = {"coef": {"total": K * d, "trainable": K * d}, "intercept": {"total": K, "trainable": K}}
        return total, total, breakdown


def count_svm(args):
    if args.model_path:
        m = _load_joblib(args.model_path)
        m = _unwrap_model(m)
        sv = getattr(m, "support_vectors_", None)
        dual = getattr(m, "dual_coef_", None)
        intercept = getattr(m, "intercept_", None)
        if sv is not None:
            total = int((sv.size if sv is not None else 0) + (dual.size if dual is not None else 0) + (intercept.size if intercept is not None else 0))
            breakdown = {
                "support_vectors": {"total": int(sv.size), "trainable": int(sv.size)},
                "dual_coef": {"total": int(dual.size) if dual is not None else 0, "trainable": int(dual.size) if dual is not None else 0},
            }
            if intercept is not None:
                breakdown["intercept"] = {"total": int(intercept.size), "trainable": int(intercept.size)}
            return total, total, breakdown
        else:
            coef = getattr(m, "coef_", None)
            intercept = getattr(m, "intercept_", None)
            total = int((coef.size if coef is not None else 0) + (intercept.size if intercept is not None else 0))
            breakdown = {"coef": {"total": int(coef.size) if coef is not None else 0, "trainable": int(coef.size) if coef is not None else 0}}
            if intercept is not None:
                breakdown["intercept"] = {"total": int(intercept.size), "trainable": int(intercept.size)}
            return total, total, breakdown
    else:
        # require d_in and n_classes: assume linear SVM
        d = args.d_in
        K = args.n_classes
        total = K * d + K
        breakdown = {"coef": {"total": K * d, "trainable": K * d}, "intercept": {"total": K, "trainable": K}}
        return total, total, breakdown


def count_rf(args):
    if not args.model_path:
        raise ValueError("--model-path is required for RandomForest counting (provide a saved joblib RandomForestClassifier)")
    m = _load_joblib(args.model_path)
    m = _unwrap_model(m)
    # sklearn RandomForest stores trees in estimators_
    total_nodes = sum(getattr(est.tree_, "node_count", 0) for est in getattr(m, "estimators_", []))
    # approximate stored elements per tree: children_left, children_right, feature, threshold arrays + value
    approx_elements = 0
    for est in getattr(m, "estimators_", []):
        node_count = getattr(est.tree_, "node_count", 0)
        value_arr = getattr(est.tree_, "value", None)
        value_size = int(value_arr.size) if (value_arr is not None and hasattr(value_arr, "size")) else 0
        approx_elements += node_count * 4 + value_size
    total = int(approx_elements)

    # compute leaf-based learned-parameter count: count leaves across all trees
    n_leaves = 0
    output_dim = 1
    for est in getattr(m, "estimators_", []):
        tree = getattr(est, "tree_", None)
        if tree is None:
            continue
        children_left = getattr(tree, "children_left", None)
        if children_left is not None:
            n_leaves += int((children_left == -1).sum())
        # try to infer output dim from tree_.value shape
        value = getattr(tree, "value", None)
        if value is not None and hasattr(value, "shape"):
            try:
                output_dim = int(value.shape[-1])
            except Exception:
                output_dim = output_dim

    learned_leaf_weights = int(n_leaves * output_dim)

    breakdown = {
        "n_trees": {"total": len(m.estimators_), "trainable": 0},
        "total_nodes": {"total": int(total_nodes), "trainable": 0},
        "approx_elements": {"total": int(approx_elements), "trainable": 0},
        "n_leaves": {"total": int(n_leaves), "trainable": 0},
        "learned_leaf_weights": {"total": int(learned_leaf_weights), "trainable": 0},
    }
    return total, 0, breakdown


def count_xgb(args):
    if not args.model_path:
        raise ValueError("--model-path is required for XGBoost counting (provide a saved xgb model)")
    try:
        import xgboost as xgb
    except Exception:
        raise RuntimeError("xgboost is required to load XGBoost models; pip install xgboost")
    # try Booster load
    try:
        bst = xgb.Booster()
        bst.load_model(args.model_path)
        # prefer structured dataframe if available
        try:
            df = bst.trees_to_dataframe()
            n_trees = int(df['Tree'].nunique())
            n_leaves = int((df['Feature'] == 'Leaf').sum()) if 'Feature' in df.columns else int((df['Node'] == 'Leaf').sum())
        except Exception:
            dumps = bst.get_dump()
            n_trees = len(dumps)
            # approximate nodes by counting lines
            total_nodes = sum(s.count('\n') + 1 for s in dumps)
            # try to estimate leaves by counting lines containing 'leaf'
            import re
            n_leaves = 0
            for s in dumps:
                for line in s.splitlines():
                    if re.search(r"leaf", line, flags=re.IGNORECASE):
                        n_leaves += 1

        # try to infer output dim: prefer sklearn wrapper n_classes_ if available (not for raw Booster)
        output_dim = 1
        # total nodes approximation
        total_nodes = None
        try:
            dumps = bst.get_dump()
            total_nodes = sum(s.count('\n') + 1 for s in dumps)
        except Exception:
            total_nodes = None

        learned_leaf_weights = int(n_leaves * output_dim)
        total = int(total_nodes) if total_nodes is not None else int(n_leaves)
        breakdown = {
            "n_trees": {"total": n_trees, "trainable": 0},
            "approx_nodes": {"total": int(total_nodes) if total_nodes is not None else int(n_leaves), "trainable": 0},
            "n_leaves": {"total": int(n_leaves), "trainable": 0},
            "learned_leaf_weights": {"total": int(learned_leaf_weights), "trainable": 0},
        }
        return total, 0, breakdown
    except Exception:
        # maybe sklearn API object saved
        m = _load_joblib(args.model_path)
        m = _unwrap_model(m)
        try:
            bst = m.get_booster()
            df = bst.trees_to_dataframe()
            total_nodes = len(df)
            n_trees = int(df['Tree'].nunique())
            n_leaves = int((df['Feature'] == 'Leaf').sum()) if 'Feature' in df.columns else 0
            # try to infer output dim from sklearn wrapper
            output_dim = getattr(m, 'n_classes_', 1)
            learned_leaf_weights = int(n_leaves * output_dim)
            breakdown = {
                "n_trees": {"total": n_trees, "trainable": 0},
                "approx_nodes": {"total": int(total_nodes), "trainable": 0},
                "n_leaves": {"total": int(n_leaves), "trainable": 0},
                "learned_leaf_weights": {"total": int(learned_leaf_weights), "trainable": 0},
            }
            total = int(total_nodes)
            return total, 0, breakdown
        except Exception:
            raise RuntimeError("Failed to inspect xgboost model; ensure you provided a Booster-compatible model file or sklearn API XGBClassifier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "ftt", "moe", "flex_moe", "logreg", "svm", "rf", "xgb"], required=True)
    # mlp args
    ap.add_argument("--d_in", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--n_classes", type=int, default=3)
    # ftt args
    ap.add_argument("--d_num", type=int, default=128)
    ap.add_argument("--out", type=str, default="results/model_param_counts", help="Output file prefix (without extension)")
    # model file for scikit-learn / xgboost artifacts
    ap.add_argument("--model-path", type=str, default=None, help="Path to a saved model file (joblib for sklearn, xgb model for xgboost)")
    # moe args
    ap.add_argument("--meta", type=str, default=None, help="Path to .meta.pkl produced by train_moe (contains 'groups')")
    ap.add_argument("--hidden_exp", type=int, default=128)
    ap.add_argument("--hidden_gate", type=int, default=128)
    ap.add_argument("--gate_type", type=str, default="softmax")
    ap.add_argument("--gumbel_hard", action="store_true")
    ap.add_argument("--gate_noise", type=float, default=0.02)
    ap.add_argument("--topk", type=int, default=None)
    # flex-moe args
    ap.add_argument("--best_params_json", type=str, default=None, help="Path to Optuna best-trial JSON with 'best_params' for Flex-MoE")
    ap.add_argument("--experts_config", type=str, default=None, help="YAML mapping expert_name -> CSV path (required for flex_moe)")
    ap.add_argument("--flex_moe_root", type=str, default="third_party/flex-moe")
    ap.add_argument("--modality", type=str, default="AMD")
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--num_patches", type=int, default=16)
    ap.add_argument("--num_experts", type=int, default=8)
    ap.add_argument("--num_routers", type=int, default=1)
    ap.add_argument("--num_layers_fus", type=int, default=1)
    ap.add_argument("--num_layers_pred", type=int, default=1)
    ap.add_argument("--num_heads", type=int, default=4)

    args = ap.parse_args()

    # handle deep models that return torch.nn.Module
    if args.model in ("mlp", "ftt", "moe", "flex_moe"):
        if args.model == "mlp":
            model = build_mlp(args)
        elif args.model == "ftt":
            model = build_ftt(args)
        elif args.model == "flex_moe":
            model = build_flex_moe(args)
        else:
            model = build_moe(args)

        total, trainable = count_params(model)
        breakdown = breakdown_params(model)
        
        # For MoE: calculate active parameters and FLOPs
        if args.model == "moe":
            active_params, total_params, active_breakdown = count_active_params_moe(model, topk=args.topk)
            active_flops, total_flops, flops_breakdown = count_flops_moe(model, batch_size=1, topk=args.topk)
            
            results = {
                "model": args.model,
                "total_params": int(total),
                "trainable_params": int(trainable),
                "active_params": int(active_params),
                "active_params_breakdown": active_breakdown,
                "active_flops": int(active_flops),
                "total_flops": int(total_flops),
                "flops_breakdown": flops_breakdown,
                "args": vars(args),
            }
        elif args.model == "flex_moe":
            active_params, active_breakdown = count_active_params_flex_moe(model, topk=args.top_k)
            results = {
                "model": args.model,
                "total_params": int(total),
                "trainable_params": int(trainable),
                "active_params": int(active_params),
                "active_params_breakdown": active_breakdown,
                "args": vars(args),
            }
        elif args.model == "ftt":
            # Calculate FLOPs for FT-Transformer
            total_flops, flops_breakdown = count_flops_ftt(model, batch_size=1)
            
            results = {
                "model": args.model,
                "total_params": int(total),
                "trainable_params": int(trainable),
                "total_flops": int(total_flops),
                "flops_breakdown": flops_breakdown,
                "args": vars(args),
            }
        elif args.model == "mlp":
            # Calculate FLOPs for MLP
            total_flops = count_flops_mlp(model)
            
            results = {
                "model": args.model,
                "total_params": int(total),
                "trainable_params": int(trainable),
                "total_flops": int(total_flops),
                "args": vars(args),
            }
        else:
            results = {
                "model": args.model,
                "total_params": int(total),
                "trainable_params": int(trainable),
                "args": vars(args),
            }

    # classical / tree-based models: compute counts from saved artifacts or shapes
    elif args.model == "logreg":
        total, trainable, breakdown = count_logreg(args)
        results = {
            "model": args.model,
            "total_params": int(total),
            "trainable_params": int(trainable),
            "args": vars(args),
        }
    elif args.model == "svm":
        total, trainable, breakdown = count_svm(args)
        results = {
            "model": args.model,
            "total_params": int(total),
            "trainable_params": int(trainable),
            "args": vars(args),
        }
    elif args.model == "rf":
        total, trainable, breakdown = count_rf(args)
        results = {
            "model": args.model,
            "total_params": int(total),
            "trainable_params": int(trainable),
            "args": vars(args),
        }
    elif args.model == "xgb":
        total, trainable, breakdown = count_xgb(args)
        results = {
            "model": args.model,
            "total_params": int(total),
            "trainable_params": int(trainable),
            "args": vars(args),
        }
    else:
        raise SystemExit("unknown model")
    # Build results: use a single consistent metric for total_params.
    # For tree ensembles we prefer the learned-parameter equivalent (leaf-based).
    
    # If the breakdown supplies a leaf-based learned parameter count, use that
    # as the canonical total_params (this provides an apples-to-apples metric
    # comparable to NN parameter counts).
    if isinstance(breakdown, dict) and "learned_leaf_weights" in breakdown:
        try:
            learned = int(breakdown["learned_leaf_weights"]["total"])
            results["total_params"] = learned
            results["trainable_params"] = 0
        except Exception:
            pass

    print(f"Model: {args.model}")
    print(f"Total params: {results['total_params']:,}")
    print(f"Trainable params: {results['trainable_params']:,}")
    
    # Print FLOPs for models that support it
    if "total_flops" in results and args.model not in ["moe"]:
        print(f"Total FLOPs: {results['total_flops']:,}")
        
        # Print detailed breakdown if available
        if "flops_breakdown" in results:
            print(f"\nFLOPs Breakdown:")
            for k, v in results['flops_breakdown'].items():
                if isinstance(v, (int, float)):
                    print(f"  {k:30s}: {v:,}" if isinstance(v, int) else f"  {k:30s}: {v}")
                else:
                    print(f"  {k:30s}: {v}")
    
    # Print active params and FLOPs for MoE
    if args.model in ("moe", "flex_moe") and "active_params" in results:
        print(f"Active params: {results['active_params']:,}")
        if "active_flops" in results:
            print(f"Active FLOPs: {results['active_flops']:,}")
        print(f"\nActive Parameters Breakdown:")
        for k, v in results['active_params_breakdown'].items():
            if isinstance(v, (int, float)):
                print(f"  {k:30s}: {v:,}" if isinstance(v, int) else f"  {k:30s}: {v}")
            else:
                print(f"  {k:30s}: {v}")
        if "flops_breakdown" in results:
            print(f"\nFLOPs Breakdown:")
            for k, v in results['flops_breakdown'].items():
                if isinstance(v, (int, float)):
                    print(f"  {k:30s}: {v:,}" if isinstance(v, int) else f"  {k:30s}: {v}")
                else:
                    print(f"  {k:30s}: {v}")
    
    print("\nModule breakdown (top-level children):")
    # normalize breakdown entries: accept either (total, trainable) tuples or
    # {'total':..., 'trainable':...} dicts (the code handles both wrappers and raw estimators)
    normalized = {}
    for k, v in breakdown.items():
        # prepare key string
        k_str = str(k)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            t, tr = v
        elif isinstance(v, dict) and "total" in v and "trainable" in v:
            t, tr = v["total"], v["trainable"]
        else:
            # try numeric coercion (fallback)
            try:
                t = int(v)
            except Exception:
                t = 0
            tr = 0

        # safe-int conversion for printing
        try:
            t_int = int(t)
        except Exception:
            t_int = 0
        try:
            tr_int = int(tr)
        except Exception:
            tr_int = 0

        print(f"  {k_str:30s}: total={t_int:,} | trainable={tr_int:,}")
        normalized[k_str] = {"total": int(t_int), "trainable": int(tr_int)}

    # attach breakdown to results as simple dicts
    results["breakdown"] = normalized

    # save results automatically
    out_prefix = getattr(args, "out", "results/model_param_counts")
    json_path = out_prefix + ".json"
    csv_path = out_prefix + ".csv"

    try:
        # ensure results dir exists
        from pathlib import Path
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        # write simple CSV summary (one row per model run)
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "total_params", "trainable_params"])
            writer.writeheader()
            writer.writerow({"model": results["model"], "total_params": int(results["total_params"]), "trainable_params": int(results["trainable_params"])})

        print(f"\nSaved JSON results to: {json_path}")
        print(f"Saved CSV summary to: {csv_path}")
    except Exception as e:  # pragma: no cover
        print(f"Warning: failed to save outputs: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
