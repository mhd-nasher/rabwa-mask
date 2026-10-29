# rabwa-mask

**rabwa-mask** is a small, CPU-only model that finds personal data in **Arabic (Gulf / Saudi dialect) and English** text, plus a
reversible redaction helper. Use it to mask names, phone numbers, IDs and addresses **before** you send text to an
external LLM API, then put the real values back into the reply.

```
in   : العميل محمد عبدالله الغامدي جواله 0551234567 وإيميله m.ghamdi@gmail.com
out  : العميل [PERSON_1] جواله [PHONE_1] وإيميله [EMAIL_1]
reply: Sure, we will call [PERSON_1] today.   →   Sure, we will call محمد عبدالله الغامدي today.
```

## The problem it solves

Teams that pass customer notes, chat messages or support tickets to a hosted LLM leak personal data in free text.
Regexes catch phone numbers and e-mails, but not names. And Arabic names are hard:

- they come in 2–4 parts, with بن / بنت, ال, and kunyas such as أبو فهد / أم خالد;
- prepositions glue onto them («لفيصل», «وبندر», «للغامدي», «لـيوسف»);
- many names are also ordinary words (جميل, كريم, أمل, نور);
- places, banks and streets carry people's names (طريق الملك عبدالعزيز, مصرف الراجحي).

General multilingual NER and small chat LLMs miss too many of these. This model was fine-tuned for exactly this
job, and it runs in a few milliseconds on a CPU, so the text never leaves your server unmasked.

## What it detects

| label | what | found by |
|---|---|---|
| `PERSON` | full / partial names, kunyas, English spellings | model |
| `ADDRESS` | a person's home address (district, street, building), Saudi short address | model + rule |
| `PHONE` | Saudi mobiles and landlines in any format (05…, +966…, 00966…, spaces, dashes, Arabic-Indic digits) | rules |
| `EMAIL` | e-mail addresses | rules |
| `NID` / `IQAMA` | Saudi national ID (starts with 1) / iqama (starts with 2) | rules |
| `IBAN` | Saudi IBAN (SA + 22 digits, spaced or not) | rules |
| `PLATE` | Saudi car plates, Arabic or Latin letters | rules |

The model also tags the fixed formats, but regexes are exact there, so the helper lets the rules own them and only
takes names and addresses from the model.

## Quick start

```bash
git clone https://github.com/mhd-nasher/rabwa-mask && cd rabwa-mask
curl -L -o model/model.int8pc.onnx \
  https://github.com/mhd-nasher/rabwa-mask/releases/download/v1.0.0/model.int8pc.onnx
pip install -r requirements.txt
python pii_redact.py "Please call Sara Al-Harbi on +966 55 123 4567, her IBAN is SA44 2000 0001 2345 6789 1234"
```

```python
from pii_redact import Redactor

r = Redactor(threads=1)                    # load once, reuse
masked, mapping = r.redact(user_text)      # send `masked` to the LLM
answer = r.restore(llm_reply, mapping)     # real values back; unknown placeholders raise
```

- **One mapping per request.** Pass the same `mapping` to `redact()` for every field of one request. Keep it in
  memory only and drop it afterwards.
- **Repeated values share a placeholder.** The same value gets the same placeholder, and every other occurrence in
  the text is masked too.
- **Fail closed.** If the model can't be loaded or raises, don't send the text.

## Results

These are measured on a held-out **synthetic** test set of 780 Gulf-Arabic / English sentences: CRM-style notes,
assistant questions and WhatsApp-style messages, built from random combinations of common names. 179 of the
sentences contain no PII at all, and there are 545 hard negatives: company, bank, place, street and project names,
plus words that are also names. The test sentence templates and a quarter of the names were never seen in training.

"Fully masked" means every letter of the entity is covered; masking half a name counts as a leak.

| | fully masked | precision | recall |
|---|---|---|---|
| names (`PERSON`, 537) | **97.2%** | 0.93 | 0.98 |
| addresses (`ADDRESS`, 110) | 99.1% | 0.74 | 0.99 |
| phone / e-mail / ID / iqama / IBAN / plate (326) | 100% | 1.00 | 1.00 |
| all PII | 98.4% | | |
| hard negatives wrongly masked | 10 of 545 (1.8%) | | |

- **Adding a name list helps.** Combined with a list of common first and family names and a short list of your own
  brand/product names that must never be masked, name masking reached **99.4%** in our tests.
- **The data is synthetic.** Real text will be messier. Test the model on your own data before relying on it.

**Speed and size** (`pii_redact.Redactor.detect`, Apple M3 CPU; a server CPU will differ):

| | 1 thread | 4 threads |
|---|---|---|
| short message (~55 chars) | ~12 ms | ~5 ms |
| 1,500 characters | ~330 ms | ~105 ms |

The model file is 105 MB (int8, per-channel; same accuracy as fp32). A process running it needs about 250 MB of RAM.

**Compared with alternatives** on the same test set, each one combined with the same regex rules and name list:

| model | names fully masked | short-request time (1 thread) | size |
|---|---|---|---|
| this model | 99.4% | ~12 ms | 105 MB |
| CAMeLBERT-NER, off the shelf | 82% | similar | 414 MB |
| GLiNER multi v2.1, zero-shot | 92% | ~36 ms | 1.1 GB |
| Qwen2.5-3B-Instruct, prompted | 87% | ~4 s | 1.9 GB |

## Known limits

- **One-word names that are also common words** can be missed: «يعطيك العافية أمل».
- **Some places named after people** get over-masked, for example streets named after princes or a hospital named
  after a doctor. Over-masking is the safe direction.
- **Only Saudi formats** are covered by the phone, ID, IBAN and plate rules. Add your own rules for other countries.
- **Text is processed in 300-character windows** (the model reads up to 128 tokens at a time), so long documents are
  slower.

## Files

| file | |
|---|---|
| `model/model.int8pc.onnx` | the model (download from the release) |
| `model/model.safetensors` | full-precision PyTorch weights for further fine-tuning (release asset) |
| `model/config.json`, `tokenizer.json`, `vocab.txt`, … | config and tokenizer (standard BERT WordPiece, no lower-casing) |
| `pii_redact.py` | detection + reversible redaction helper |

The labels are BIO tags: `O`, `B-/I-PERSON`, `PHONE`, `EMAIL`, `NID`, `IQAMA`, `IBAN`, `PLATE`, `ADDRESS`. The
model also works from Node.js with `onnxruntime-node` and a plain BERT WordPiece tokenizer.

## Training

The model is fine-tuned from
[CAMeL-Lab/bert-base-arabic-camelbert-mix-ner](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix-ner)
on 1,600 synthetic sentences, for 6 epochs, at lr 5e-5, on Apple MPS (about one minute). Three training seeds gave
the same accuracy within 0.3 points. No real personal data was used; every name in the training data is a random
combination of common given and family names.

## License

Apache License 2.0, the same as the base model. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
