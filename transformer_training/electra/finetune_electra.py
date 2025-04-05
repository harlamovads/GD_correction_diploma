import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    ElectraTokenizerFast,
    ElectraForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, fbeta_score

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for l in label if l != -100]
        for label in labels
    ]
    
    # Flatten the predictions and labels
    flat_predictions = np.concatenate(true_predictions)
    flat_labels = np.concatenate(true_labels)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_labels, flat_predictions, average='binary', zero_division=0
    )
    accuracy = accuracy_score(flat_labels, flat_predictions)
    
    # Calculate F0.5 score (weighs precision more than recall)
    f0_5 = fbeta_score(flat_labels, flat_predictions, beta=0.5)
    
    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        flat_labels, flat_predictions, average=None, zero_division=0
    )
    
    # F0.5 per class
    f0_5_per_class = []
    for i in range(len(precision_per_class)):
        if precision_per_class[i] == 0 and recall_per_class[i] == 0:
            f0_5_per_class.append(0)
        else:
            # F0.5 formula: (1 + 0.5²) * (precision * recall) / (0.5² * precision + recall)
            beta_squared = 0.5 ** 2
            f0_5_per_class.append(
                (1 + beta_squared) * precision_per_class[i] * recall_per_class[i] / 
                (beta_squared * precision_per_class[i] + recall_per_class[i]) if 
                (beta_squared * precision_per_class[i] + recall_per_class[i]) > 0 else 0
            )
    
    # Create detailed metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f0.5': f0_5,
        'precision_class_0': precision_per_class[0],
        'recall_class_0': recall_per_class[0],
        'f1_class_0': f1_per_class[0],
        'f0.5_class_0': f0_5_per_class[0],
        'support_class_0': support_per_class[0],
    }
    
    # Only add class 1 metrics if we have both classes in the data
    if len(precision_per_class) > 1:
        metrics.update({
            'precision_class_1': precision_per_class[1],
            'recall_class_1': recall_per_class[1],
            'f1_class_1': f1_per_class[1],
            'f0.5_class_1': f0_5_per_class[1],
            'support_class_1': support_per_class[1],
        })
    
    return metrics

def main():
    # Initialize Weights & Biases
    wandb.init(project="error-detection-electra", name="electra-token-classification")
    
    # Load the dataset
    dataset_dir = "electra_token_classification_dataset"
    # If running in Colab, check if the path exists or needs to be adjusted
    if not os.path.exists(dataset_dir) and os.path.exists("/content/drive/MyDrive/" + dataset_dir):
        dataset_dir = "/content/drive/MyDrive/" + dataset_dir
    
    dataset = load_from_disk(dataset_dir)
    print(f"Loaded dataset from {dataset_dir}")
    print(f"Training examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    
    # Initialize tokenizer
    model_name = "google/electra-small-discriminator"
    tokenizer = ElectraTokenizerFast.from_pretrained(model_name)
    
    # Initialize model
    model = ElectraForTokenClassification.from_pretrained(
        model_name,
        num_labels=2  # Binary classification: correct or incorrect
    )
    print(f"Model loaded: {model_name}")
    
    # Create data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True
    )
    
    # Calculate training steps for reasonable evaluation frequency
    # Aim for ~5 evaluations during the entire training
    batch_size = 32  # Adjust based on available GPU memory
    num_train_examples = len(dataset["train"])
    train_steps_per_epoch = num_train_examples // batch_size
    total_epochs = 3
    
    # We want approximately 5 evaluations in total
    eval_steps = max(train_steps_per_epoch // 2, 1)
    
    # Define output directory
    output_dir = "electra_error_detection_model"
    if os.path.exists("/content/drive/MyDrive"):
        output_dir = "/content/drive/MyDrive/" + output_dir
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=total_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f0.5",  # Prioritize precision over recall
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        report_to="wandb",  # Enable Weights & Biases logging
        save_total_limit=2,  # Only keep the 2 best checkpoints
        fp16=True,  # Use mixed precision training if available
        dataloader_num_workers=2,  # Parallelize data loading
        gradient_accumulation_steps=2,  # Accumulate gradients for effective larger batch size
    )
    
    # Early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Log final evaluation results to Weights & Biases
    wandb.log({"final_" + k: v for k, v in eval_results.items()})
    
    # Save the model
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Save the best checkpoint explicitly
    best_model_path = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    
    # Get the path to the best checkpoint
    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint:
        print(f"Best checkpoint was: {best_checkpoint}")
        
        # The trainer already loaded the best model when load_best_model_at_end=True
        # So we can just save the current model to the best_model_path
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        print(f"Best model saved to {best_model_path}")
    
    # Finish Weights & Biases run
    wandb.finish()
    
    print("Training complete!")

if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("Running in Google Colab")
    except:
        IN_COLAB = False
        print("Not running in Google Colab")
    
    # Set the device to GPU if available
    import torch
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU")
    
    main()