import os
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss

# 1. Load the dataset from your JSONL file
# Replace with your actual path if different (e.g. "./insurance_retrieval_dataset.jsonl")
dataset_path = "/content/insurance_retrieval_dataset1.jsonl"
dataset = load_dataset("json", data_files=dataset_path, split="train")

# 2. Rename 'query' column to 'anchor' to match Sentence Transformers standard expectations
# Column names MUST be: 'anchor', 'positive', 'negative'
dataset = dataset.rename_column("query", "anchor")

# If you want to do a train/val split (highly recommended for evaluation):
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

# 3. Load the pre-trained embedding model
model_id = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_id)

# 4. Define the loss function
# MultipleNegativesRankingLoss accepts triplets (anchor, positive, negative)
# and uses the third element as a hard negative.
train_loss = MultipleNegativesRankingLoss(model)

# 5. Define the training arguments
training_args = SentenceTransformerTrainingArguments(
    output_dir="./all-mpnet-base-v2-insurance-tuned", # Output directory for model & checkpoints
    num_train_epochs=3,                              # Train for 3 epochs
    per_device_train_batch_size=8,                   # Batch size (adjust based on GPU memory)
    learning_rate=2e-5,                              # Fine-tuning learning rate
    warmup_ratio=0.1,                                # Warmup over 10% of training steps
    weight_decay=0.01,
    logging_steps=10,                                # Log metrics every 10 steps
    eval_strategy="epoch",                           # Evaluate at the end of each epoch
    save_strategy="epoch",                           # Save checkpoint at the end of each epoch
    load_best_model_at_end=True,                     # Keep the best model weights
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",                                # Change to "tensorboard" or "wandb" if desired
)

# 6. Initialize the Trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
)

# 7. Start the training process
print("Starting training...")
trainer.train()

# 8. Save the final model
final_model_path = "./all-mpnet-base-v2-insurance-final"
model.save(final_model_path)
print(f"Model successfully trained and saved to: {final_model_path}")
