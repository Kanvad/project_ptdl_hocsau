import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable

from datasets import Dataset, DatasetDict, load_dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
IGNORE_LABEL_ID = -100


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 1
    sorted_vals = sorted(values)
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    pos = (len(sorted_vals) - 1) * (p / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return int(round(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac))


def build_tokenizer(tokenizer_name: str, spacy_model: str) -> Callable[[str], list[str]]:
    if tokenizer_name == "spacy":
        try:
            import spacy  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "spaCy is not installed. Install it first or use --tokenizer whitespace."
            ) from exc

        try:
            nlp = spacy.load(spacy_model, disable=["parser", "tagger", "ner", "lemmatizer"])
        except OSError as exc:
            raise RuntimeError(
                f"Cannot load spaCy model '{spacy_model}'. Install it or use --tokenizer whitespace."
            ) from exc

        def spacy_tokenize(text: str) -> list[str]:
            return [tok.text for tok in nlp(text)]

        return spacy_tokenize

    def whitespace_tokenize(text: str) -> list[str]:
        return text.split()

    return whitespace_tokenize


def normalize_token(token: str, lowercase: bool) -> str:
    tok = token.lower() if lowercase else token
    # Replace contiguous digit spans with '@' to reduce vocabulary size.
    tok = re.sub(r"\d+", "@", tok)
    return tok


def _pick_existing_key(candidates: list[str], keys: set[str]) -> str | None:
    for key in candidates:
        if key in keys:
            return key
    return None


def _to_label_str(value, label_decoder: Callable[[int], str] | None) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int) and label_decoder is not None:
        return label_decoder(value)
    return str(value)


def parse_hf_split(
    split_ds: Dataset,
    split_name: str,
    tokenize: Callable[[str], list[str]],
    lowercase: bool,
) -> list[dict]:
    keys = set(split_ds.column_names)

    # Case 0: raw line format from PubMed_200k_RCT where each row has only "text".
    if keys == {"text"}:
        records: list[dict] = []
        current_id = None
        current_sentences: list[list[str]] = []
        current_labels: list[str] = []

        def flush_raw() -> None:
            nonlocal current_sentences, current_labels
            if current_id is not None and current_sentences:
                records.append(
                    {
                        "id": str(current_id),
                        "sentences": current_sentences,
                        "labels": current_labels,
                    }
                )
            current_sentences = []
            current_labels = []

        for idx, row in enumerate(split_ds):
            line = str(row.get("text", "")).strip()
            if not line:
                continue

            if line.startswith("###"):
                flush_raw()
                raw_id = line[3:].strip()
                current_id = raw_id if raw_id else f"{split_name}-{idx}"
                continue

            if "\t" not in line:
                continue

            label, sentence = line.split("\t", 1)
            sent_tokens = [
                normalize_token(tok, lowercase=lowercase)
                for tok in tokenize(sentence)
                if str(tok).strip()
            ]
            if not sent_tokens:
                continue

            if current_id is None:
                current_id = f"{split_name}-{idx}"

            current_labels.append(label.strip())
            current_sentences.append(sent_tokens)

        flush_raw()
        return records

    sentence_list_key = _pick_existing_key(["sentences", "abstract_text", "text"], keys)
    label_list_key = _pick_existing_key(["labels", "abstract_label", "label", "target"], keys)
    id_key = _pick_existing_key(["id", "abstract_id", "doc_id", "document_id", "pmid"], keys)

    label_decoder: Callable[[int], str] | None = None
    if label_list_key is not None and label_list_key in split_ds.features:
        feature = split_ds.features[label_list_key]
        names = getattr(feature, "names", None)
        if names is not None and hasattr(feature, "int2str"):
            label_decoder = feature.int2str

    # Case 1: each row is already a full abstract (list of sentences + list of labels).
    if sentence_list_key is not None and label_list_key is not None and len(split_ds) > 0:
        sample_val = split_ds[0][sentence_list_key]
        if isinstance(sample_val, list):
            records: list[dict] = []
            for idx, row in enumerate(split_ds):
                raw_sentences = row[sentence_list_key]
                raw_labels = row[label_list_key]

                if not isinstance(raw_sentences, list) or not isinstance(raw_labels, list):
                    continue
                if len(raw_sentences) != len(raw_labels):
                    continue

                tokens_per_sent: list[list[str]] = []
                labels: list[str] = []
                for sent, label in zip(raw_sentences, raw_labels):
                    sent_tokens = [
                        normalize_token(tok, lowercase=lowercase)
                        for tok in tokenize(str(sent))
                        if str(tok).strip()
                    ]
                    if not sent_tokens:
                        continue
                    tokens_per_sent.append(sent_tokens)
                    labels.append(_to_label_str(label, label_decoder))

                if not tokens_per_sent:
                    continue

                rec_id = str(row[id_key]) if id_key is not None else f"{split_name}-{idx}"
                records.append({"id": rec_id, "sentences": tokens_per_sent, "labels": labels})
            return records

    # Case 2: each row is one sentence, need to group rows by abstract id or line_number reset.
    text_key = _pick_existing_key(["text", "sentence", "sent", "abstract_text"], keys)
    label_key = _pick_existing_key(["label", "target", "abstract_label", "labels"], keys)
    line_key = _pick_existing_key(["line_number", "sentence_id", "sent_id"], keys)
    group_key = _pick_existing_key(["abstract_id", "doc_id", "document_id", "pmid", "id"], keys)

    if text_key is None or label_key is None:
        raise RuntimeError(
            f"Unsupported split schema for '{split_name}'. Available columns: {sorted(keys)}"
        )

    records: list[dict] = []
    current_group = None
    current_sentences: list[list[str]] = []
    current_labels: list[str] = []

    def flush() -> None:
        nonlocal current_sentences, current_labels
        if current_sentences:
            rec_id = str(current_group) if current_group is not None else f"{split_name}-{len(records)}"
            records.append({"id": rec_id, "sentences": current_sentences, "labels": current_labels})
        current_sentences = []
        current_labels = []

    for idx, row in enumerate(split_ds):
        if group_key is not None:
            row_group = row[group_key]
        elif line_key is not None:
            line_number = row[line_key]
            row_group = f"{split_name}-{idx}" if int(line_number) == 0 else current_group
        else:
            row_group = f"{split_name}-{idx}"

        if current_group is None:
            current_group = row_group
        elif row_group != current_group:
            flush()
            current_group = row_group

        text = str(row[text_key])
        label = _to_label_str(row[label_key], label_decoder)
        sent_tokens = [
            normalize_token(tok, lowercase=lowercase)
            for tok in tokenize(text)
            if str(tok).strip()
        ]
        if not sent_tokens:
            continue

        current_sentences.append(sent_tokens)
        current_labels.append(label)

    flush()
    return records


def get_split(dataset_dict: DatasetDict, candidates: list[str]) -> tuple[str, Dataset]:
    for name in candidates:
        if name in dataset_dict:
            return name, dataset_dict[name]
    raise RuntimeError(
        f"Cannot find split among {candidates}. Available splits: {list(dataset_dict.keys())}"
    )


def build_vocab(train_records: list[dict], min_freq: int) -> dict[str, int]:
    counter = Counter()
    for record in train_records:
        for sent in record["sentences"]:
            counter.update(sent)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    kept_tokens = [tok for tok, freq in counter.items() if freq >= min_freq]
    kept_tokens.sort(key=lambda t: (-counter[t], t))

    for tok in kept_tokens:
        vocab[tok] = len(vocab)
    return vocab


def build_label_mapping(train_records: list[dict]) -> dict[str, int]:
    labels = sorted({label for rec in train_records for label in rec["labels"]})
    return {label: idx for idx, label in enumerate(labels)}


def infer_max_lengths(train_records: list[dict], sent_pct: float, word_pct: float) -> tuple[int, int]:
    sentence_counts = [len(rec["sentences"]) for rec in train_records]
    word_counts = [len(sent) for rec in train_records for sent in rec["sentences"]]

    max_sentences = max(1, percentile(sentence_counts, sent_pct))
    max_words = max(1, percentile(word_counts, word_pct))
    return max_sentences, max_words


def encode_record(
    record: dict,
    vocab: dict[str, int],
    label2id: dict[str, int],
    max_sentences: int,
    max_words: int,
) -> dict:
    pad_id = vocab[PAD_TOKEN]
    unk_id = vocab[UNK_TOKEN]

    input_ids = [[pad_id for _ in range(max_words)] for _ in range(max_sentences)]
    word_mask = [[0 for _ in range(max_words)] for _ in range(max_sentences)]
    sentence_mask = [0 for _ in range(max_sentences)]
    label_ids = [IGNORE_LABEL_ID for _ in range(max_sentences)]

    for i, (label, sentence) in enumerate(zip(record["labels"], record["sentences"])):
        if i >= max_sentences:
            break

        sentence_mask[i] = 1
        label_ids[i] = label2id[label]

        for j, token in enumerate(sentence):
            if j >= max_words:
                break
            input_ids[i][j] = vocab.get(token, unk_id)
            word_mask[i][j] = 1

    return {
        "id": record["id"],
        "input_ids": input_ids,
        "word_mask": word_mask,
        "sentence_mask": sentence_mask,
        "label_ids": label_ids,
    }


def save_json(path: Path, content: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=True, indent=2)


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PubMed_200k_RCT from Hugging Face for Hierarchical BiLSTM + CRF")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="zj88zj/PubMed_200k_RCT",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional Hugging Face dataset config name",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory")
    parser.add_argument("--min-freq", type=int, default=3, help="Minimum token frequency to keep in vocabulary")
    parser.add_argument("--max-sentences", type=int, default=None, help="Max sentences per abstract")
    parser.add_argument("--max-words", type=int, default=None, help="Max words per sentence")
    parser.add_argument(
        "--sent-percentile",
        type=float,
        default=95.0,
        help="Percentile to infer max sentences if --max-sentences is not provided",
    )
    parser.add_argument(
        "--word-percentile",
        type=float,
        default=95.0,
        help="Percentile to infer max words if --max-words is not provided",
    )
    parser.add_argument("--tokenizer", choices=["whitespace", "spacy"], default="whitespace")
    parser.add_argument("--spacy-model", type=str, default="en_core_sci_sm")
    parser.add_argument("--no-lowercase", action="store_true", help="Disable lowercase normalization")
    args = parser.parse_args()

    tokenize = build_tokenizer(args.tokenizer, args.spacy_model)
    lowercase = not args.no_lowercase

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    if not isinstance(dataset, DatasetDict):
        raise RuntimeError("Expected a dataset with split dictionary (train/validation/test)")

    train_split_name, train_split = get_split(dataset, ["train"])
    dev_split_name, dev_split = get_split(dataset, ["validation", "dev"])
    test_split_name, test_split = get_split(dataset, ["test"])

    train_records = parse_hf_split(train_split, train_split_name, tokenize, lowercase)
    dev_records = parse_hf_split(dev_split, dev_split_name, tokenize, lowercase)
    test_records = parse_hf_split(test_split, test_split_name, tokenize, lowercase)

    vocab = build_vocab(train_records, min_freq=args.min_freq)
    label2id = build_label_mapping(train_records)

    inferred_max_sentences, inferred_max_words = infer_max_lengths(
        train_records,
        sent_pct=args.sent_percentile,
        word_pct=args.word_percentile,
    )
    max_sentences = args.max_sentences or inferred_max_sentences
    max_words = args.max_words or inferred_max_words

    encoded_train = [encode_record(r, vocab, label2id, max_sentences, max_words) for r in train_records]
    encoded_dev = [encode_record(r, vocab, label2id, max_sentences, max_words) for r in dev_records]
    encoded_test = [encode_record(r, vocab, label2id, max_sentences, max_words) for r in test_records]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.output_dir / "train.jsonl", encoded_train)
    save_jsonl(args.output_dir / "dev.jsonl", encoded_dev)
    save_jsonl(args.output_dir / "test.jsonl", encoded_test)

    save_json(args.output_dir / "vocab.json", vocab)
    save_json(args.output_dir / "label2id.json", label2id)

    id2label = {str(v): k for k, v in label2id.items()}
    save_json(args.output_dir / "id2label.json", id2label)

    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "input_splits": {
            "train": train_split_name,
            "dev": dev_split_name,
            "test": test_split_name,
        },
        "num_examples": {
            "train": len(train_records),
            "dev": len(dev_records),
            "test": len(test_records),
        },
        "vocab_size": len(vocab),
        "num_labels": len(label2id),
        "max_sentences": max_sentences,
        "max_words": max_words,
        "min_freq": args.min_freq,
        "tokenizer": args.tokenizer,
        "lowercase": lowercase,
        "digit_replacement": "\\d+ -> @",
        "ignore_label_id": IGNORE_LABEL_ID,
    }
    save_json(args.output_dir / "metadata.json", metadata)

    print("Preprocessing completed")
    print(f"Dataset          : {args.dataset_name} (config={args.dataset_config})")
    print(f"Output directory : {args.output_dir}")
    print(
        f"Examples         : train={len(train_records)}, dev={len(dev_records)}, test={len(test_records)}"
    )
    print(f"Vocab size       : {len(vocab)}")
    print(f"Labels           : {label2id}")
    print(f"Padding          : max_sentences={max_sentences}, max_words={max_words}")


if __name__ == "__main__":
    main()
