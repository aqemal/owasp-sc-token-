import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

SEED_VALUE = 7  # change to 7, etc. for additional runs

# ---------- Load your JSON data ----------
data_path = Path("data/first_dataset.json")
raw = json.loads(data_path.read_text())
print("Loaded", len(raw), "examples")

# Convert to HF Dataset
dataset = Dataset.from_list(raw)

# Train/validation split (80/20) at contract level
dataset = dataset.train_test_split(test_size=0.2, seed=7)
ds = DatasetDict(
    {
        "train": dataset["train"],
        "validation": dataset["test"],
    }
)

# Validation label counts (for information only)
val_counts = Counter()
for ex in ds["validation"]:
    val_counts.update(l for l in ex["labels"] if l >= 0)
print("Validation label counts:", val_counts)

# ---------- Oversample SC01 examples in training ----------
train_list = list(ds["train"])
sc01_examples = [ex for ex in train_list if 1 in ex["labels"]]

print("Train examples:", len(train_list))
print("SC01 examples in train:", len(sc01_examples))

oversample_factor = 2  # duplicates each SC01 example 2 extra times
train_oversampled = train_list + sc01_examples * oversample_factor
print("Oversampled train examples:", len(train_oversampled))

ds = DatasetDict(
    {
        "train": Dataset.from_list(train_oversampled),
        "validation": ds["validation"],
    }
)

# ---------- Labels ----------
label_list = ["O", "SC01", "SC05", "SC08"]  # 0, 1, 2, 3
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label_list)

# ---------- Tokenizer & model ----------
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# ---------- Metrics (token-level, with per-class F1) ----------
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    pred_flat = []
    label_flat = []
    for pred_row, lab_row in zip(predictions, labels):
        for p_i, l_i in zip(pred_row, lab_row):
            if l_i == -100:
                continue
            pred_flat.append(int(p_i))
            label_flat.append(int(l_i))

    pred_flat = np.array(pred_flat)
    label_flat = np.array(label_flat)

    accuracy = (pred_flat == label_flat).mean()

    positive = label_flat > 0
    pred_positive = pred_flat > 0

    tp_all = np.logical_and(pred_positive, positive).sum()
    fp_all = np.logical_and(pred_positive, ~positive).sum()
    fn_all = np.logical_and(~pred_positive, positive).sum()

    def prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    precision, recall, f1 = prf(tp_all, fp_all, fn_all)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

    label_ids = {"SC01": 1, "SC05": 2, "SC08": 3}
    for name, c in label_ids.items():
        support = (label_flat == c).sum()
        tp = np.logical_and(pred_flat == c, label_flat == c).sum()
        fp = np.logical_and(pred_flat == c, label_flat != c).sum()
        fn = np.logical_and(pred_flat != c, label_flat == c).sum()
        p_c, r_c, f1_c = prf(tp, fp, fn)

        metrics[f"{name}_precision"] = p_c
        metrics[f"{name}_recall"] = r_c
        metrics[f"{name}_f1"] = f1_c
        metrics[f"{name}_support"] = int(support)

    return metrics

# ---------- Class weights ----------
class_counts = torch.tensor([24474, 354, 716, 461], dtype=torch.float)
class_weights = 1.0 / torch.sqrt(class_counts)
class_weights = class_weights / class_weights.mean()
class_weights = class_weights.to(torch.float32)

# ---------- Weighted Trainer ----------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )

        loss = loss_fct(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )

        if return_outputs:
            return loss, outputs
        return loss

# ---------- Training arguments ----------
training_args = TrainingArguments(
    output_dir=f"codebert-sc05-checkpoints-seed{7}",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    seed=SEED_VALUE,
)

# ---------- Trainer ----------
trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ---------- Main ----------
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(f"codebert-sc05-model-seed{SEED_VALUE}")
    print(f"Training finished. Model saved to codebert-sc05-model-seed{7}")

    print("Evaluating on validation set...")
    metrics = trainer.evaluate()
    print(metrics)

    Path("results").mkdir(exist_ok=True)
    with open(f"results/metrics_sc05_seed{7}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to results/metrics_sc05_seed{SEED_VALUE}.json")