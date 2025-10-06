# import json
# from typing import List, Dict, Any
# from dataclasses import dataclass
# import torch
# from torch.utils.data import Dataset


# @dataclass
# class QAExample:
#     id: str
#     question: str
#     context: str
#     answer_text: str
#     answer_start: int


# def load_jsonl(path: str) -> List[Dict[str, Any]]:
#     with open(path, 'r', encoding='utf-8') as f:
#         return [json.loads(line) for line in f if line.strip()]


# class HotpotQADataset(Dataset):
#     """Minimal JSONL dataset for extractive QA.
#     Expected JSONL fields per line:
#       {"id": str, "question": str, "context": str, "answer_text": str or "answer": str, "answer_start": int}
#     Note: context should already contain the gold answer span; long contexts should be pre-trimmed.
#     """
#     def __init__(self, path: str, tokenizer, max_length: int = 512):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         raw = load_jsonl(path)
#         self.data = []
#         for ex in raw:
#             answer_text = ex.get('answer_text') or ex.get('answer')
#             answer_start = ex.get('answer_start')
#             if (answer_text is None) or (answer_start is None):
#                 continue
#             self.data.append({
#                 'id': str(ex.get('id', '')),
#                 'question': ex['question'],
#                 'context': ex['context'],
#                 'answer_text': answer_text,
#                 'answer_start': int(answer_start),
#             })

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         ex = self.data[idx]
#         enc = self.tokenizer(
#             ex['question'],
#             ex['context'],
#             return_tensors='pt',
#             return_offsets_mapping=True,
#             padding='max_length',
#             truncation=True,
#             max_length=self.max_length,
#         )
#         input_ids = enc['input_ids'][0]
#         attention_mask = enc['attention_mask'][0]
#         offset_mapping = enc['offset_mapping'][0]  # (S,2)
#         seq_ids = enc.sequence_ids(0)  # None/0(question)/1(context)

#         start_char = ex['answer_start']
#         end_char = start_char + len(ex['answer_text'])
#         S = input_ids.size(0)
#         start_pos = end_pos = S  # default ignored index (== S)

#         # Primary mapping: token whose span covers the boundary char
#         for i, (seq_id, (st, ed)) in enumerate(zip(seq_ids, offset_mapping.tolist())):
#             if seq_id != 1:  # restrict to context tokens
#                 continue
#             if st <= start_char < ed and start_pos == S:
#                 start_pos = i
#             if st < end_char <= ed and end_pos == S:
#                 end_pos = i
#             if start_pos != S and end_pos != S:
#                 break

#         # Fallback: choose tightest span fully inside the answer chars
#         if start_pos == S or end_pos == S:
#             s = e = None
#             for i, (seq_id, (st, ed)) in enumerate(zip(seq_ids, offset_mapping.tolist())):
#                 if seq_id != 1:
#                     continue
#                 if st >= start_char and ed <= end_char:
#                     if s is None:
#                         s = i
#                     e = i
#             if s is not None and e is not None:
#                 start_pos, end_pos = s, e

#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'start_positions': torch.tensor(start_pos, dtype=torch.long),
#             'end_positions': torch.tensor(end_pos, dtype=torch.long),
#             'answer_text': ex['answer_text'],
#             'id': ex['id'],
#         }


# def collate_train(batch):
#     return {
#         'input_ids': torch.stack([b['input_ids'] for b in batch]),
#         'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
#         'start_positions': torch.stack([b['start_positions'] for b in batch]),
#         'end_positions': torch.stack([b['end_positions'] for b in batch]),
#     }


# def collate_eval(batch):
#     out = collate_train(batch)
#     out['answer_text'] = [b['answer_text'] for b in batch]
#     out['id'] = [b['id'] for b in batch]
#     return out

# datasets/hotpotqa.py (doc_stride ready)
import json
from typing import List, Dict, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

@dataclass
class QAExample:
    id: str
    question: str
    context: str
    answer_text: str
    answer_start: int

def load_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: yield json.loads(line)

class HotpotQADataset(Dataset):
    """
    Sinh nhiều feature/row cho mỗi mẫu bằng overflow + doc_stride.
    """
    def __init__(self, path: str, tokenizer, max_length: int = 512, doc_stride: int = 128, is_train: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.is_train = is_train
        self.features = []  # mỗi item là 1 feature đã gắn nhãn (start/end) hoặc chỉ lưu id/answer_text cho eval

        raw = list(load_jsonl(path))
        for ex in raw:
            qa = {
                "id": str(ex.get("id", "")),
                "question": ex["question"],
                "context": ex["context"],
                "answer_text": ex.get("answer_text") or ex.get("answer"),
                "answer_start": int(ex.get("answer_start")),
            }
            # tokenize với overflow theo "only_second" (chỉ cắt context)
            enc = self.tokenizer(
                qa["question"],
                qa["context"],
                return_offsets_mapping=True,
                padding="max_length",
                truncation="only_second",
                max_length=self.max_length,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            offset_mapping = enc["offset_mapping"]
            overflow_to_sample = enc["overflow_to_sample_mapping"]

            for i in range(input_ids.size(0)):
                # mapping về sample gốc
                sample_idx = overflow_to_sample[i].item()
                # offsets và sequence ids cho feature i
                offsets = offset_mapping[i]
                seq_ids = enc.sequence_ids(i)  # None/0(question)/1(context)
                context_mask = torch.tensor([1 if sid == 1 else 0 for sid in seq_ids], dtype=torch.long)
                start_char = qa["answer_start"]
                end_char = start_char + len(qa["answer_text"])

                S = input_ids.size(1)
                start_pos = end_pos = S  # ignored by default

                # chỉ xét token thuộc context (seq_id==1)
                for ti, (sid, (st, ed)) in enumerate(zip(seq_ids, offsets.tolist())):
                    if sid != 1: 
                        continue
                    if st <= start_char < ed and start_pos == S:
                        start_pos = ti
                    if st < end_char <= ed and end_pos == S:
                        end_pos = ti
                    if start_pos != S and end_pos != S:
                        break

                if start_pos == S or end_pos == S:
                    # fallback: chọn span nằm trọn trong khoảng
                    s = e = None
                    for ti, (sid, (st, ed)) in enumerate(zip(seq_ids, offsets.tolist())):
                        if sid != 1: continue
                        if st >= start_char and ed <= end_char:
                            if s is None: s = ti
                            e = ti
                    if s is not None and e is not None:
                        start_pos, end_pos = s, e

                feat = {
                    "input_ids": input_ids[i],
                    "attention_mask": attention_mask[i],
                    "id": qa["id"],
                    "answer_text": qa["answer_text"],
                    "context_mask": context_mask,
                }
                if self.is_train:
                    feat["start_positions"] = torch.tensor(start_pos, dtype=torch.long)
                    feat["end_positions"] = torch.tensor(end_pos, dtype=torch.long)
                self.features.append(feat)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

def collate_train(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "context_mask": torch.stack([b["context_mask"] for b in batch]),  # <<
        "start_positions": torch.stack([b["start_positions"] for b in batch]),
        "end_positions": torch.stack([b["end_positions"] for b in batch]),
    }

def collate_eval(batch):
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "context_mask": torch.stack([b["context_mask"] for b in batch]),
        "id": [b["id"] for b in batch],
        "answer_text": [b["answer_text"] for b in batch],
    }
    if "start_positions" in batch[0]:
        out["start_positions"] = torch.stack([b["start_positions"] for b in batch])
        out["end_positions"] = torch.stack([b["end_positions"] for b in batch])
    return out
