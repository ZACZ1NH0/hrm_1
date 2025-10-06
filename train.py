import os
import math
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from hrm_core import HRMCoreConfig, HRMForQA
from datasets.hotpotqa import HotpotQADataset, collate_train, collate_eval


# --------- Utils: metrics ---------
import re, string

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = {}
    for t in pred_tokens:
        if t in truth_tokens:
            common[t] = min(pred_tokens.count(t), truth_tokens.count(t))
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# --------- Repro ---------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------- Train/Eval ---------

# def best_span(start_logits: torch.Tensor, end_logits: torch.Tensor, attn_mask: torch.Tensor):
#     """Greedy best span per item: pick best start, then best end >= start among unmasked tokens."""
#     B, S = start_logits.shape
#     starts = start_logits.masked_fill(attn_mask == 0, -1e9).argmax(dim=1)  # [B]
#     ends = []
#     for b in range(B):
#         s = starts[b].item()
#         end = end_logits[b].masked_fill((attn_mask[b] == 0) | (torch.arange(S, device=end_logits.device) < s), -1e9).argmax().item()
#         ends.append(end)
#     ends = torch.tensor(ends, device=start_logits.device)
#     return starts, ends
def best_span(start_logits, end_logits, attn_mask, max_answer_len=30):
    B, S = start_logits.shape
    start_scores = start_logits.masked_fill(attn_mask == 0, -1e9)
    end_scores   = end_logits.masked_fill(attn_mask == 0, -1e9)
    best_s = torch.zeros(B, dtype=torch.long, device=start_logits.device)
    best_e = torch.zeros(B, dtype=torch.long, device=start_logits.device)
    for b in range(B):
        s_scores = start_scores[b]
        e_scores = end_scores[b]
        # prefix max để tìm nhanh e cho mỗi s trong cửa sổ dài tối đa
        best = -1e9
        bs = be = 0
        for s in range(S):
            e_max = min(S-1, s + max_answer_len)
            # lấy e trong [s, e_max] có điểm cao nhất
            e_slice = e_scores[s:e_max+1]
            e_rel = torch.argmax(e_slice).item()
            e = s + e_rel
            val = (s_scores[s] + e_scores[e]).item()
            if val > best:
                best = val; bs = s; be = e
        best_s[b] = bs; best_e[b] = be
    return best_s, best_e

def best_span_and_score(start_logits, end_logits, attn_mask, context_mask, max_answer_len=30):
    # valid = vừa không pad, vừa là token thuộc context
    valid = (attn_mask == 1) & (context_mask == 1)
    start_scores = start_logits.masked_fill(~valid, -1e9)
    end_scores   = end_logits.masked_fill(~valid, -1e9)

    B, S = start_scores.shape
    best_s = torch.zeros(B, dtype=torch.long, device=start_scores.device)
    best_e = torch.zeros(B, dtype=torch.long, device=start_scores.device)
    best_val = torch.full((B,), -1e9, device=start_scores.device)

    for b in range(B):
        cur = -1e9; bs = be = 0
        for s in range(S):
            if start_scores[b, s] <= -1e8:  # invalid
                continue
            e_max = min(S - 1, s + max_answer_len)
            e_slice = end_scores[b, s:e_max + 1]
            if e_slice.numel() == 0: 
                continue
            e_rel = int(torch.argmax(e_slice))
            e = s + e_rel
            val = (start_scores[b, s] + end_scores[b, e]).item()
            if val > cur:
                cur = val; bs = s; be = e
        best_s[b], best_e[b], best_val[b] = bs, be, cur
    return best_s, best_e, best_val

# def evaluate(model, tokenizer, loader, device):
#     model.eval()
#     losses = []
#     ems, f1s = [], []
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="eval", leave=False):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             start_positions = None
#             end_positions = None
#             if 'start_positions' in batch:
#                 start_positions = batch['start_positions'].to(device)
#                 end_positions = batch['end_positions'].to(device)

#             out = model(input_ids=input_ids, attention_mask=attention_mask,
#                         start_positions=start_positions, end_positions=end_positions)
#             if 'loss' in out:
#                 losses.append(out['loss'].item())

#             # Decode predictions
#             s_idx, e_idx = best_span(out['start_logits'], out['end_logits'], attention_mask)
#             for i in range(input_ids.size(0)):
#                 pred_ids = input_ids[i, s_idx[i]:e_idx[i] + 1].detach().cpu().tolist()
#                 pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
#                 gold_text = batch['answer_text'][i]
#                 ems.append(exact_match_score(pred_text, gold_text))
#                 f1s.append(f1_score(pred_text, gold_text))

#     avg_loss = float(np.mean(losses)) if losses else 0.0
#     avg_em = float(np.mean(ems)) if ems else 0.0
#     avg_f1 = float(np.mean(f1s)) if f1s else 0.0
#     return {"loss": avg_loss, "EM": avg_em, "F1": avg_f1}
def evaluate(model, tokenizer, loader, device, encoder=None, amp=False, max_answer_len=30):
    import contextlib, numpy as np
    model.eval()
    losses = []
    id2best = {}  # id -> {'pred': str, 'gold': str, 'score': float}

    autocast_ctx = (torch.autocast("cuda", enabled=amp and torch.cuda.is_available())
                    if torch.cuda.is_available() else contextlib.nullcontext())

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            context_mask = batch['context_mask'].to(device)

            with autocast_ctx:
                if encoder is not None:
                    enc_out = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
                    enc_out = enc_out.to(model.ln_in.weight.dtype)
                    out = model(attention_mask=attention_mask, inputs_embeds=enc_out)
                else:
                    out = model(input_ids=input_ids, attention_mask=attention_mask)

            if 'start_positions' in batch:  # optional dev loss
                start_positions = batch['start_positions'].to(device)
                end_positions   = batch['end_positions'].to(device)
                with autocast_ctx:
                    if encoder is not None:
                        out_loss = model(attention_mask=attention_mask, inputs_embeds=enc_out,
                                         start_positions=start_positions, end_positions=end_positions)
                    else:
                        out_loss = model(input_ids=input_ids, attention_mask=attention_mask,
                                         start_positions=start_positions, end_positions=end_positions)
                losses.append(out_loss['loss'].item())

            s_idx, e_idx, span_val = best_span_and_score(
                out['start_logits'], out['end_logits'], attention_mask, context_mask, max_answer_len
            )

            for i in range(input_ids.size(0)):
                ex_id = batch['id'][i]
                pred_ids = input_ids[i, s_idx[i]: e_idx[i] + 1].detach().cpu().tolist()
                pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                score = float(span_val[i].item())
                gold_text = batch['answer_text'][i]
                rec = id2best.get(ex_id)
                if (rec is None) or (score > rec['score']):
                    id2best[ex_id] = {'pred': pred_text, 'gold': gold_text, 'score': score}

    ems = [exact_match_score(v['pred'], v['gold']) for v in id2best.values()]
    f1s = [f1_score(v['pred'], v['gold']) for v in id2best.values()]
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "EM": float(np.mean(ems)) if ems else 0.0,
        "F1": float(np.mean(f1s)) if f1s else 0.0,
    }


# --------- Main ---------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True, help='./hotpot_train_v1.1.json')
    parser.add_argument('--dev_path', type=str, required=True, help='./hotpot_dev_distractor_v1.json')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='checkpoints')
    parser.add_argument('--encoder_name', type=str, default='', help='HF encoder name, e.g., bert-base-uncased')
    parser.add_argument('--freeze_encoder', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.encoder_name or 'bert-base-uncased', use_fast=True)
    
    encoder = None
    if args.encoder_name:
        encoder = AutoModel.from_pretrained(args.encoder_name).to(device)
        if args.freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad = False
    # Data
    train_ds = HotpotQADataset(args.train_path, tokenizer, max_length=args.max_length, doc_stride=128, is_train=True)
    dev_ds = HotpotQADataset(args.dev_path, tokenizer, max_length=args.max_length, doc_stride=128, is_train=True)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_train)
    dev_ld = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_eval)

    # Model
    cfg = HRMCoreConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=args.max_length,
                        hidden_size=768, num_heads=12, ff_mult=4, H_layers=2, L_layers=2,
                        H_cycles=2, L_cycles=1)
    model = HRMForQA(cfg).to(device)

    # Optimizer (standard AdamW)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_train),
                    desc=f"train epoch {epoch}")
        scaler = torch.cuda.amp.GradScaler()
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            if encoder is not None:
                with torch.no_grad() if args.freeze_encoder else torch.enable_grad():
                    enc_out = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
                out = model(attention_mask=attention_mask,
                    inputs_embeds=enc_out,
                    start_positions=start_positions,
                    end_positions=end_positions)
            else:
                out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions)
            
            loss = out['loss']

            opt.zero_grad()
            scaler.scale(loss).backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # opt.step()
            scaler.step(opt)
            scaler.update()

            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Eval
        metrics = evaluate(model, tokenizer, dev_ld, device)
        print(f"Epoch {epoch}: dev loss={metrics['loss']:.4f} EM={metrics['EM']*100:.2f} F1={metrics['F1']*100:.2f}")
        # Save best
        if metrics['F1'] > best_f1:
            best_f1 = metrics['F1']
            path = os.path.join(args.out_dir, 'best.pt')
            torch.save({'model_state_dict': model.state_dict(), 'cfg': cfg.__dict__, 'tokenizer': args.tokenizer}, path)
            print(f"Saved new best to {path}")


if __name__ == '__main__':
    main()
