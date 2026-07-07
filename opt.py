import time
import copy
import os
from datetime import datetime
from typing import Dict, List, Sequence, Set

import torch
import torch.nn as nn

from quant import *
from sparsegpt import *
from modelutils import *
from fedspallm_agg import fedspallm_aggregate_state_dicts_from_layer_reports

DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False 


def maybe_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model):
    return sum((p != 0).sum().item() for p in model.parameters())


def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

# 
def normalize_opt_model_name(model_name: str) -> str:
    aliases = {
        'OPT-125m': 'facebook/opt-125m',
        'opt-125m': 'facebook/opt-125m',
        'OPT1.3b': 'facebook/opt-1.3b',
        'opt1.3b': 'facebook/opt-1.3b',
        'facebook/opt-125m': 'facebook/opt-125m',
        'facebook/opt-1.3b': 'facebook/opt-1.3b',
    }
    return aliases[model_name]


# 0-100等差数列模拟客户端资源能力评分
def allocate_client_resource_scores(num_clients: int) -> List[float]:
    """Assign arithmetic-sequence resource scores in [0, 100]."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if num_clients == 1:
        return [100.0]
    step = 100.0 / float(num_clients - 1)
    return [step * i for i in range(num_clients)]

# 根据客户端资源能力评分分配剪枝层数
def allocate_client_layer_counts(resource_scores: Sequence[float], total_layers: int) -> List[int]:
    """
    Compute per-client pruning layer count k proportional to r, with
    sum(k) > total_layers to ensure each layer can be covered.
    """
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if len(resource_scores) == 0:
        raise ValueError("resource_scores cannot be empty")

    # 要不要加一个冗余系数
    target_total = total_layers*2

    score_sum = float(sum(resource_scores))
    if score_sum <= 0:
        raise ValueError("resource_scores mistake.")
    else:
        raw = [target_total * (float(r) / score_sum) for r in resource_scores]
        ks = [int(x)+1 for x in raw] #进一取整
    return ks

def sample_client_pruning_layers(
    client_layer_counts: Sequence[int], total_layers: int, base_seed: int
) -> List[Set[int]]:
    """Sample k unique layer ids for each client from available model layers."""
    all_ids = torch.arange(total_layers)
    selected: List[Set[int]] = []
    for client_id, k in enumerate(client_layer_counts):
        if k <= 0:
            selected.append(set())
            continue
        k = min(k, total_layers)
        g = torch.Generator()
        g.manual_seed(int(base_seed + 1009 * (client_id + 1)))
        perm = all_ids[torch.randperm(total_layers, generator=g)]
        chosen = set(int(x.item()) for x in perm[:k])
        selected.append(chosen)
    return selected





def build_sparse_client_layer_masks(
    sd: Dict[str, torch.Tensor], selected_layer_ids: Set[int]
) -> Dict[str, torch.Tensor]:
    """
    Client-to-server sparse pruning report:
      - only include parameter masks for the selected (pruned) layer ids;
      - omit non-selected layers' masks.
    """
    if not selected_layer_ids:
        return {}
    layer_prefixes = [f"model.decoder.layers.{i}." for i in selected_layer_ids]
    masks: Dict[str, torch.Tensor] = {}
    for name, v in sd.items():
        if not v.is_floating_point():
            continue
        if any(name.startswith(prefix) for prefix in layer_prefixes):
            masks[name] = (v != 0).to(dtype=v.dtype)
    return masks

# 对模型进行逐层量化和剪枝
@torch.no_grad()
def opt_sequential(model, dataloader, dev, selected_layer_ids: Set[int] | None = None):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    maybe_empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        should_prune_layer = selected_layer_ids is None or i in selected_layer_ids

        subset = find_layers(layer)
        
        gpts = {}
        for name in subset:
            if not should_prune_layer:
                continue
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
              continue
            gpts[name] = SparseGPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        if should_prune_layer:
            for name in gpts:
                print(i, name)
                print('Pruning ...')
                sparsity = args.sparsity
                gpts[name].fasterprune(
                    sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
                )
                gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        maybe_empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    maybe_empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        maybe_empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
         wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    # 获取全局时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, choices=[
            'OPT-125m', 'OPT1.3b', 'facebook/opt-125m', 'facebook/opt-1.3b'
        ],
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=32,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0.8,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
       '--invert', action='store_true', 
       help='Invert subset.'
    )
    parser.add_argument(
       '--save', type=str, default='',
       help='Path to saved model.'
    )
    parser.add_argument(
       '--log_wandb', action='store_true',
       help='Whether to log to wandb.'
    )
    parser.add_argument(
        '--num_clients', type=int, default=4, choices=[4, 8, 16],
        help='Number of federated clients.'
    )
    parser.add_argument(
        '--epoch', type=int, default=1,
        help='T: number of federated communication rounds.'
    )


    args = parser.parse_args()
    args.model = normalize_opt_model_name(args.model)

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    global_model = get_opt(args.model)
    global_model.eval()
    total_params = count_parameters(global_model)
    nonzero_params = count_nonzero_parameters(global_model)
    sparsity = 1 - (nonzero_params / total_params)
    print('===== Initial model stats =====')
    print(f"Initial model: total params {total_params}, nonzero params {nonzero_params}, sparsity {sparsity:.4f}")
    total_layers = len(global_model.model.decoder.layers)
    client_resource_scores = allocate_client_resource_scores(args.num_clients)
    client_layer_counts = allocate_client_layer_counts(client_resource_scores, total_layers)
    print('===== Client capability assignment =====')
    for client_id, (r_i, k_i) in enumerate(zip(client_resource_scores, client_layer_counts)):
        print(f'  Client {client_id + 1}: r={r_i:.2f}, k={k_i}')
    print(f'  Total model layers={total_layers}, sum(k)={sum(client_layer_counts)}')

    # apply_mask_expansion=True
    apply_mask_expansion = False


    for t in range(args.epoch):
        print(f'===== Communication round {t + 1}/{args.epoch} =====')
        ref_sd = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()} # 保存当前全局模型
        client_states = []
        client_layer_reports = []
        client_selected_layers = sample_client_pruning_layers(
            client_layer_counts=client_layer_counts,
            total_layers=total_layers,
            base_seed=args.seed + t * 7919,
        )

        for client_id in range(args.num_clients):
            print(f'  Client {client_id + 1}/{args.num_clients}')
            model = copy.deepcopy(global_model)
            model.eval()
            selected_layers = client_selected_layers[client_id]
            print(f'    assigned layer ids: {sorted(selected_layers)}')

            
            # 生成客户端数据集 
            seed_c = args.seed + client_id + t * (args.num_clients + 1)
            dataloader, _ = get_loaders(
                args.dataset, nsamples=args.nsamples, seed=seed_c, model=args.model, seqlen=model.seqlen
            )

            if args.sparsity:
                tick = time.time()
                opt_sequential(model, dataloader, DEV, selected_layer_ids=selected_layers)
                for n, p in model.named_parameters():
                    print(n, torch.mean((p == 0).float()))
                    if 'fc2' in n:
                        break
                print(time.time() - tick)

            sd_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            client_states.append(sd_cpu)
            client_layer_reports.append(
                {
                    "layer_ids": sorted(selected_layers),
                    "masks": build_sparse_client_layer_masks(sd_cpu, selected_layers),
                }
            )

            del model
            maybe_empty_cache()

        agg_sd = fedspallm_aggregate_state_dicts_from_layer_reports(
            ref_sd,
            client_states,
            client_layer_reports,
            target_sparsity=args.sparsity,
            apply_mask_expansion=apply_mask_expansion,
        )
        dev = next(global_model.parameters()).device
        global_model.load_state_dict(
            {k: v.to(dev) for k, v in agg_sd.items()},
            strict=True,
        )

   
    print('===== Final evaluation (global model) =====')
    final_total_params = count_parameters(global_model)
    final_nonzero_params = count_nonzero_parameters(global_model)
    final_sparsity = 1 - (final_nonzero_params / final_total_params)
    print('===== Final model stats =====')
    print(f"Final model: total params {final_total_params}, nonzero params {final_nonzero_params}, sparsity {final_sparsity:.4f}")
    for dataset in ['wikitext2', 'ptb', 'c4']:
        _, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=global_model.seqlen
        )
        print(dataset)
        opt_eval(global_model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        save_root = os.path.join(args.save, timestamp) if args.save else ''
        if save_root:
            os.makedirs(save_root, exist_ok=True)
            global_model.save_pretrained(save_root)
