"""Reversible PII redaction for Arabic (Gulf/Saudi) and English text, on CPU.

    pip install onnxruntime tokenizers numpy
    python pii_redact.py "كلم محمد عبدالله الغامدي على 0551234567"

The ONNX model finds names and home addresses; regex rules own the fixed
formats (phones, e-mails, national ID / iqama, IBAN, plates), which they get
right and the model only guesses at. Every value becomes a placeholder such
as [PERSON_1]; `restore()` puts the real values back into a reply.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = Path(__file__).parent / "model"
TAGS = json.loads((HERE / "config.json").read_text())["id2label"]
TAGS = [TAGS[str(i)] for i in range(len(TAGS))]
TYPES = sorted({t[2:] for t in TAGS if t != "O"})
MODEL_TYPES = {"PERSON", "ADDRESS"}

FOLD = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
SEP = r"[\s\-]?"
RULES = [
    ("EMAIL", r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+"),
    ("IBAN", r"(?<![A-Za-z0-9])SA\s?\d{2}(?:\s?\d){20}(?!\d)"),
    ("PHONE", r"(?<![\d+])(?:(?:\+|00)?\(?\+?966\)?" + SEP + r"0?|0)?5\d(?:" + SEP + r"\d){7}(?!\d)"),
    ("PHONE", r"(?<![\d+])(?:(?:\+|00)?966" + SEP + r"|0)1[1-7](?:" + SEP + r"\d){7}(?!\d)"),
    ("NID", r"(?<!\d)1\d{9}(?!\d)"),      # Saudi national ID
    ("IQAMA", r"(?<!\d)2\d{9}(?!\d)"),    # Saudi residence permit
    ("PLATE", r"(?<![؀-ۿ])(?:[أابحدرسصطعقكلمنوىهـ]\s){2}[أابحدرسصطعقكلمنوىهـ]\s?\d{1,4}(?!\d)"),
    ("PLATE", r"\b(?!SAR|USD|VAT)[ABJDRSXTEGKLZNUVH]{3}[\s\-]?\d{1,4}\b"),
    ("ADDRESS", r"\b[A-Z]{4}\d{4}\b"),    # Saudi national short address, e.g. RAAA1234
]
RULES = [(lab, re.compile(rx, re.I if lab == "IBAN" else 0)) for lab, rx in RULES]
# «لـيوسف» -> «ل يوسف»: a preposition glued with a tatweel becomes its own word (same length).
GLUED = re.compile(r"(?<![\wـ])([وف]?[وفبلك])ـ")
CLITIC = re.compile(r"(?:[وفبلك]|وب|ول|لل|[وفبلك]ـ)?")
CHUNK = 300  # characters per model window (well under 128 tokens)


class Redactor:
    def __init__(self, model_dir=HERE, threads=1, threshold=0.3):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(Path(model_dir) / "model.int8pc.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.tok.enable_truncation(128)
        self.threshold = threshold

    # ---------------------------------------------------------------- model
    def _model_spans(self, text):
        enc = self.tok.encode(text)
        n = len(enc.ids)
        feed = {"input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.ones((1, n), dtype=np.int64),
                "token_type_ids": np.zeros((1, n), dtype=np.int64)}
        logits = self.sess.run(None, feed)[0][0]
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        spans = []
        for m in re.finditer(r"\S+", text):
            toks = [i for i, (s, e) in enumerate(enc.offsets) if s < e and s < m.end() and m.start() < e]
            hot = [i for i in toks if 1 - p[i, 0] >= self.threshold]
            if not hot:
                continue
            scores = {t: p[hot, TAGS.index("B-" + t)].sum() + p[hot, TAGS.index("I-" + t)].sum() for t in TYPES}
            label = max(scores, key=scores.get)
            begin = p[hot[0], TAGS.index("B-" + label)] > p[hot[0], TAGS.index("I-" + label)]
            start = max(m.start(), enc.offsets[hot[0]][0])
            if not CLITIC.fullmatch(text[m.start():start]):
                start = m.start()
            end = m.end()
            while end > start and not text[end - 1].isalnum():
                end -= 1
            if end <= start:
                continue
            last = spans[-1] if spans else None
            if last and last["label"] == label and not begin and text[last["end"]:start].strip() == "":
                last["end"] = end
            else:
                spans.append({"start": start, "end": end, "label": label})
        return spans

    # ---------------------------------------------------------------- detect
    def detect(self, text):
        folded = text.translate(FOLD)
        found = []
        for label, rx in RULES:
            for m in rx.finditer(folded):
                if not any(s["start"] < m.end() and m.start() < s["end"] for s in found):
                    found.append({"start": m.start(), "end": m.end(), "label": label})
        view = GLUED.sub(r"\1 ", text)
        pos = 0
        while pos < len(view):
            end = min(len(view), pos + CHUNK)
            if end < len(view):
                cut = view.rfind(" ", pos + CHUNK // 2, end)
                end = cut if cut > pos else end
            for s in self._model_spans(view[pos:end]):
                s = {**s, "start": s["start"] + pos, "end": s["end"] + pos}
                if s["label"] not in MODEL_TYPES:
                    # the regex rules own the formats; a letters-only hit is a mis-typed name
                    if any(c.isdigit() or c == "@" for c in view[s["start"]:s["end"]].translate(FOLD)):
                        continue
                    if any(r["start"] < s["end"] and s["start"] < r["end"] for r in found):
                        continue
                    s["label"] = "PERSON"
                clash = [o for o in found if o["start"] < s["end"] and s["start"] < o["end"]]
                if clash:
                    o = clash[0]
                    o["start"], o["end"] = min(o["start"], s["start"]), max(o["end"], s["end"])
                else:
                    found.append(s)
            pos = max(end - 40, pos + 1) if end < len(view) else end
        return sorted(found, key=lambda s: s["start"])

    # ---------------------------------------------------------------- redact / restore
    def redact(self, text, mapping=None):
        """Returns (masked_text, mapping). Re-use `mapping` across the fields of one request."""
        mapping = mapping if mapping is not None else {"by_value": {}, "values": {}, "count": {}}
        spans = self.detect(text)
        # mask every other occurrence of a found value too (with or without a glued prefix)
        for s in list(spans):
            value = text[s["start"]:s["end"]]
            if len(value) < 3:
                continue
            for m in re.finditer(r"(?<!\w)" + CLITIC.pattern + "(" + re.escape(value) + r")(?!\w)", text):
                a, b = m.start(1), m.end(1)
                if not any(o["start"] < b and a < o["end"] for o in spans):
                    spans.append({"start": a, "end": b, "label": s["label"]})
        out, pos = [], 0
        for s in sorted(spans, key=lambda s: s["start"]):
            value = text[s["start"]:s["end"]]
            key = (s["label"], re.sub(r"\s+", " ", value.translate(FOLD)).strip())
            if key not in mapping["by_value"]:
                n = mapping["count"].get(s["label"], 0) + 1
                mapping["count"][s["label"]] = n
                ph = f"[{s['label']}_{n}]"
                mapping["by_value"][key] = ph
                mapping["values"][ph] = value
            out += [text[pos:s["start"]], mapping["by_value"][key]]
            pos = s["end"]
        out.append(text[pos:])
        return "".join(out), mapping

    @staticmethod
    def restore(reply, mapping):
        """Put real values back. An unknown placeholder raises (fail closed)."""
        def back(m):
            ph = f"[{m.group(1).upper()}_{m.group(2)}]"
            if ph not in mapping["values"]:
                raise ValueError(f"unknown placeholder {m.group(0)}")
            return mapping["values"][ph]
        return re.sub(r"\[\s*(" + "|".join(TYPES) + r")\s*[_\- ]\s*(\d+)\s*\]", back, reply, flags=re.I)


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "العميل محمد عبدالله الغامدي جواله 0551234567 وإيميله m.ghamdi@gmail.com"
    r = Redactor()
    masked, mapping = r.redact(text)
    print("masked  :", masked)
    reply = "Sure, we will call " + next(iter(mapping["values"]), "") + " today."
    print("restored:", r.restore(reply, mapping))
