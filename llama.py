import time
import copy
import os
from datetime import datetime

import torch
import torch.nn as nn

from sparsegpt import *
from modelutils import *
from quant import *
from fedspallm_agg import build_client_masks_from_state_dict, fedspallm_aggregate_state_dicts

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


def normalize_llama_model_name(model_name: str) -> str:
    aliases = {
        'Llama-2-7b': 'meta-llama/Llama-2-7b-hf',
        'llama-2-7b': 'meta-llama/Llama-2-7b-hf',
        'meta-llama/Llama-2-7b-hf': 'meta-llama/Llama-2-7b-hf',
    }
    return aliases[model_name]


@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
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
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Pruning ...")
                sparsity = args.sparsity
                gpts[name].fasterprune(
                    sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                )
                gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][
                    int(W.numel() * args.sparsity)
                ]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        choices=["Llama-2-7b", "meta-llama/Llama-2-7b-hf"],
        help="LlaMA model to load",
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--sparsity", type=float, default=0, help="Target sparsity")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--gmp", action="store_true", help="Whether to run the GMP baseline."
    )
    parser.add_argument(
        "--wbits", type=int, default=16, help="Whether to quantize as well."
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--prune_only",
        type=str,
        default="",
        help="Prune only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument("--save", type=str, default="", help="Path to saved model.")
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=4,
        choices=[4, 8, 16],
        help="Number of federated clients.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="T: number of federated communication rounds (FedSpaLLM Algorithm 1).",
    )
    parser.add_argument(
        "--local_rounds",
        type=int,
        default=1,
        help="Local pruning passes per client per communication round (repeat SparseGPT pruning).",
    )

    args = parser.parse_args()
    args.model = normalize_llama_model_name(args.model)

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    global_model = get_llama(args.model)
    global_model.eval()

    apply_mask_expansion = (
        (args.sparsity or args.prunen)
        and not args.gmp
        and args.prunen == 0
        and args.sparsity > 0
    )

    for t in range(args.epoch):
        print(f"===== Communication round {t + 1}/{args.epoch} =====")
        ref_sd = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        client_states = []
        client_masks = []

        for client_id in range(args.num_clients):
            print(f"  Client {client_id + 1}/{args.num_clients}")
            model = copy.deepcopy(global_model)
            model.eval()

            seed_c = args.seed + client_id + t * (args.num_clients + 1)
            dataloader, _ = get_loaders(
                args.dataset,
                nsamples=args.nsamples,
                seed=seed_c,
                model=args.model,
                seqlen=model.seqlen,
            )

            if (args.sparsity or args.prunen) and not args.gmp:
                for _ in range(args.local_rounds):
                    tick = time.time()
                    llama_sequential(model, dataloader, DEV)
                    for n, p in model.named_parameters():
                        print(n, torch.mean((p == 0).float()))
                        if "down_proj" in n:
                            break
                    print(time.time() - tick)

            sd_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            client_states.append(sd_cpu)
            client_masks.append(build_client_masks_from_state_dict(sd_cpu))

            del model
            torch.cuda.empty_cache()

        agg_sd = fedspallm_aggregate_state_dicts(
            ref_sd,
            client_states,
            client_masks,
            target_sparsity=args.sparsity,
            apply_mask_expansion=apply_mask_expansion,
        )
        dev = next(global_model.parameters()).device
        global_model.load_state_dict(
            {k: v.to(dev) for k, v in agg_sd.items()},
            strict=True,
        )

    print("===== Final evaluation (global model) =====")
    for dataset in ["wikitext2", "ptb", "c4"]:
        _, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=global_model.seqlen
        )
        print("Dataset:", dataset)
        llama_eval(global_model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        save_root = os.path.join(args.save, timestamp)
        os.makedirs(save_root, exist_ok=True)
        global_model.save_pretrained(save_root)
