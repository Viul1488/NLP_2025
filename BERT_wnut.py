import datasets
import random
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import torch
import numpy as np
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer

# --- Configuration ---
MODEL_CHECKPOINT = "bert-base-uncased"
DATASET_NAME = "conll2003"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
WEIGHT_DECAY = 0.01
LABEL_ALL_TOKENS = False # Standard practice for NER with subwords

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# --- Load Dataset and Get Labels ---
print(f"Loading dataset: {DATASET_NAME}")
raw_datasets = datasets.load_dataset(DATASET_NAME)
label_list = raw_datasets["train"].features["ner_tags"].feature.names
num_labels = len(label_list)
print(f"Number of labels: {num_labels}")
print(f"Label list: {label_list}")

# --- Initialize Tokenizer ---
print(f"Loading tokenizer for: {MODEL_CHECKPOINT}")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)

# --- Tokenize and Align Labels ---
def tokenize_and_align_labels(examples, label_all_tokens=LABEL_ALL_TOKENS):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )
    labels = []
    for i, label_sequence in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None: # Special tokens get -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx: # First token of a new word
                label_ids.append(label_sequence[word_idx])
            else: # Subsequent subwords of the same word
                label_ids.append(label_sequence[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Tokenizing and aligning labels...")
tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)

# --- Data Collator ---
data_collator = DataCollatorForTokenClassification(tokenizer)

# --- Metric for Evaluation ---
metric = datasets.load_metric("seqeval")

def compute_metrics(eval_preds):
    pred_logits_batch, labels_batch = eval_preds
    pred_indices_batch = np.argmax(pred_logits_batch, axis=2)

    # Convert indices to label strings, removing -100s
    true_label_strings = []
    pred_label_strings = []

    for i in range(labels_batch.shape[0]):  # Iterate over sentences in the batch
        sentence_true_labels = []
        sentence_pred_labels = []
        for j in range(labels_batch.shape[1]):  # Iterate over tokens in the sentence
            if labels_batch[i, j] != -100:
                sentence_true_labels.append(label_list[labels_batch[i, j]])
                sentence_pred_labels.append(label_list[pred_indices_batch[i, j]])
        
        # Only add if there are actual (non-padded) tokens
        if sentence_true_labels: # Avoids adding empty lists if a sequence was all -100 (e.g. only CLS/SEP)
            true_label_strings.append(sentence_true_labels)
            pred_label_strings.append(sentence_pred_labels)
    
    if not true_label_strings: # Handle cases where no valid labels were found after filtering
        print("Warning: No valid sequences for seqeval after filtering.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    results = metric.compute(predictions=pred_label_strings, references=true_label_strings)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# --- Subset Creation ---
def create_subsets(dataset, fractions):
    subsets_with_info = []
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices) # Shuffle once for all subsets

    for fraction in fractions:
        subset_size = int(fraction * total_size)
        subset_indices = indices[:subset_size]
        subsets_with_info.append({
            "subset": Subset(dataset, subset_indices),
            "fraction": fraction,
            "size": subset_size
        })
    return subsets_with_info

# --- Training and Evaluation on Subsets ---
def train_and_evaluate_on_subset_entry(train_subset_info, eval_dataset):
#    print(f"\n--- Training on subset: {train_subset_info['fraction']*100:.0f}% of data ({train_subset_info['size']} samples) ---")
    
    # IMPORTANT: Re-initialize the model for each subset to ensure fair comparison
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels
    )
    model.to(device) # Move model to the correct device

    # Training arguments - can be defined once if they don't change per subset
    # Or, if you need to adjust epochs/steps based on subset size, define here
    training_args = TrainingArguments(
        output_dir=f"test-ner-",
        eval_strategy="epoch", # Evaluate at the end of each epoch
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=100, # Log training loss more frequently
        save_strategy="epoch", # Potentially save checkpoints
        load_best_model_at_end=True, # If you want the best model from training
        metric_for_best_model="f1", # Use F1 to determine the best model
        report_to="none" # Disable wandb/tensorboard reporting for simplicity here
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset_info,
        eval_dataset=eval_dataset, # Use the full validation set for comparable evaluation
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    trainer.train()
    
    print("Evaluating final model on validation set...")
    eval_result = trainer.evaluate(eval_dataset=eval_dataset) # Explicitly evaluate on the chosen eval_dataset
    #print(f"Evaluation results for subset {train_subset_info['fraction']*100:.0f}%: {eval_result}")
    return eval_result

# --- Main Execution for Learning Curve ---
fractions = [0.2, 0.4] # Example fractions
# fractions = [0.01, 0.05, 0.1, 0.2] # For quicker testing on small fractions
print("Creating training subsets...")
train_subsets_info = create_subsets(tokenized_datasets["train"], fractions)

f1_scores_on_validation = []
all_metrics_on_validation = []
#eval_result = train_and_evaluate_on_subset_entry(train_subsets_info, tokenized_datasets["validation"])
for subset_info in train_subsets_info:
    eval_results = train_and_evaluate_on_subset_entry(subset_info, tokenized_datasets["validation"])
    eval_f1 = eval_results["eval_f1"]
    f1_scores_on_validation.append(eval_f1)
    all_metrics_on_validation.append(eval_results)


#print("\n--- Learning Curve Metrics on Validation Set ---")
for frac, metrics in zip(fractions, all_metrics_on_validation):
    print(f"Fraction: {frac*100:.0f}% - "
          f"Precision: {metrics['eval_precision']:.4f}, "
          f"Recall: {metrics['eval_recall']:.4f}, "
          f"F1 Score: {metrics['eval_f1']:.4f}, "
          f"Accuracy: {metrics['eval_accuracy']:.4f}")
#print(    f"Precision: {eval_result['eval_precision']:.4f}, "
#          f"Recall: {eval_result['eval_recall']:.4f}, "
#          f"F1 Score: {eval_result['eval_f1']:.4f}, "
#          f"Accuracy: {eval_result['eval_accuracy']:.4f}")



# --- Plotting ---
def plot_performance(fractions_plot, f1_scores_plot, dataset_name_plot="Validation"):
    plt.figure(figsize=(10, 6))
    plt.plot([f * 100 for f in fractions_plot], f1_scores_plot, marker='o', linestyle='-')
    plt.xlabel('Training Data Fraction (%)')
    plt.ylabel(f'F1 Score on {dataset_name_plot} Set')
    plt.title(f'BERT Model Performance vs. Training Sample Size ({DATASET_NAME})')
    plt.xticks([f * 100 for f in fractions_plot])
    plt.grid(True)
    plt.ylim(0, 1) # F1 score is between 0 and 1
    plt.show()

#plot_performance(fractions, f1_scores_on_validation)

# --- Optional: Final Evaluation on Test Set (after selecting best fraction or using full data) ---
# If you want to report final test scores, train a model on the full training set (or best fraction)
# and then evaluate on tokenized_datasets["test"]
# Make sure not to use the test set for any hyperparameter tuning or model selection during experiments.

# Example for full training and testing:
# print("\n--- Training on FULL training data and evaluating on TEST set ---")
# model_full_train = AutoModelForTokenClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels)
# model_full_train.to(device)

# full_train_args = TrainingArguments(
#     output_dir="test-ner-full",
#     evaluation_strategy = "epoch",
#     learning_rate=LEARNING_RATE,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
#     num_train_epochs=NUM_TRAIN_EPOCHS,
#     weight_decay=WEIGHT_DECAY,
#     logging_steps=200,
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     report_to="none"
# )

# trainer_full = Trainer(
#     model=model_full_train,
#     args=full_train_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"], # Use validation set during full training for early stopping/best model
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )

# trainer_full.train()
# print("Evaluating on TEST set...")
# test_results = trainer_full.evaluate(eval_dataset=tokenized_datasets["test"])
# print(f"Test Set Results: {test_results}")