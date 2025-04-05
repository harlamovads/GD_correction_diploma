import json
import os
from transformers import ElectraTokenizerFast
from datasets import Dataset, DatasetDict
import numpy as np

def read_jsonl(file_path):
    """Read jsonl file and return a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON at line {i+1}: {str(e)}")
                    print(f"Problematic line: {line[:100]}...")
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found. Please make sure the path is correct.")
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {str(e)}")
        
    return data

def align_tokens_and_labels(text, tags, tokenizer):
    """
    Tokenize text and align error spans with tokens.
    Returns tokenized input and binary labels (0: correct, 1: incorrect).
    """
    # Tokenize the text
    tokenized_input = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)
    
    # Initialize labels (0 for correct tokens)
    labels = np.zeros(len(tokenized_input["input_ids"]))
    
    # Extract offset mapping to map character positions to token positions
    offset_mapping = tokenized_input.pop("offset_mapping")
    
    # Special tokens (like [CLS], [SEP]) have empty offset mapping (0,0)
    # We should mark these with -100 so they are ignored during loss calculation
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:  # Special token
            labels[i] = -100
    
    # Map character-level error spans to token-level labels
    for tag in tags:
        try:
            error_start = int(tag["span_start"])
            error_end = int(tag["span_end"])
            
            # Find which tokens correspond to the error span
            for i, (start, end) in enumerate(offset_mapping):
                # Skip special tokens 
                if start == 0 and end == 0:
                    continue
                    
                # If the token overlaps with the error span, mark it as incorrect (1)
                if max(start, error_start) < min(end, error_end):
                    labels[i] = 1
        except (ValueError, TypeError) as e:
            # Log the problematic tag and continue with the next one
            continue
    
    # Convert labels to list
    tokenized_input["labels"] = labels.tolist()
    
    return tokenized_input

def process_dataset(data, tokenizer):
    """Process the entire dataset."""
    processed_data = []
    skipped_items = 0
    
    for i, item in enumerate(data):
        try:
            # Check if required keys exist
            if "text" not in item:
                print(f"Warning: Item {i} missing 'text' key. Skipping.")
                skipped_items += 1
                continue
                
            if "tags" not in item:
                print(f"Warning: Item {i} missing 'tags' key. Treating as all correct tokens.")
                item["tags"] = []
            
            text = item["text"]
            tags = item["tags"]
            
            # Validate tags
            valid_tags = []
            for tag in tags:
                if "span_start" not in tag or "span_end" not in tag:
                    continue
                    
                if not tag["span_start"] or not tag["span_end"]:
                    continue
                    
                try:
                    # Ensure span values are valid integers
                    start = int(tag["span_start"])
                    end = int(tag["span_end"])
                    
                    # Ensure spans are within text bounds
                    if 0 <= start < end <= len(text):
                        valid_tags.append(tag)
                except (ValueError, TypeError):
                    continue
            
            # Align tokens and create labels
            processed_item = align_tokens_and_labels(text, valid_tags, tokenizer)
            processed_data.append(processed_item)
            
        except Exception as e:
            print(f"Error processing item {i}: {str(e)}")
            skipped_items += 1
    
    print(f"Processed {len(processed_data)} items, skipped {skipped_items} items")
    return processed_data

def main():
    # Path to your jsonl file
    input_file = "dataset.jsonl"
    
    # Load the data
    print(f"Loading data from {input_file}...")
    data = read_jsonl(input_file)
    print(f"Loaded {len(data)} examples")
    
    # Check data structure
    if len(data) > 0:
        print("Sample data structure:")
        sample = data[0]
        print(f"Keys in sample: {list(sample.keys())}")
        if "tags" in sample:
            print(f"Number of tags in first sample: {len(sample['tags'])}")
            if len(sample["tags"]) > 0:
                print(f"Sample tag structure: {sample['tags'][0]}")
    
    # Initialize the ELECTRA tokenizer
    tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")
    print("Tokenizer loaded")
    
    # Process the dataset
    print("Processing dataset...")
    processed_data = []
    skipped_count = 0
    
    for i, item in enumerate(data):
        try:
            processed_item = align_tokens_and_labels(item["text"], item["tags"], tokenizer)
            processed_data.append(processed_item)
            
            # Print progress every 100 items
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(data)} examples...")
                
        except Exception as e:
            skipped_count += 1
            print(f"Error processing item {i}: {str(e)}")
            print(f"Problematic item: {item}")
            continue
    
    print(f"Dataset processing complete. Processed {len(processed_data)} examples, skipped {skipped_count} examples.")
    
    # Convert to Hugging Face Dataset format
    dataset = Dataset.from_list(processed_data)
    
    # Split into train and validation sets (80/20 split)
    train_test_split = dataset.train_test_split(test_size=0.2)
    
    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test']
    })
    
    # Save the dataset
    output_dir = "electra_token_classification_dataset"
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    
    print(f"Dataset saved to {output_dir}")
    print(f"Training examples: {len(dataset_dict['train'])}")
    print(f"Validation examples: {len(dataset_dict['validation'])}")
    
    # Show a sample
    print("\nSample example:")
    sample_idx = 0
    sample = dataset_dict['train'][sample_idx]
    
    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
    
    # Print tokens and their corresponding labels
    print("Tokens and labels:")
    for token, label in zip(tokens, sample['labels']):
        if label == 1:
            print(f"Token: {token}, Label: INCORRECT")
        else:
            print(f"Token: {token}, Label: CORRECT")

if __name__ == "__main__":
    main()