import random
import typer
import spacy
from pathlib import Path
from spacy.tokens import DocBin
from spacy.util import load_model
from spacy.training.example import Example


def split_spacy_file(input_file, output_dir, train_split=0.7, dev_split=0.2, test_split=0.1, seed=42):
    """
    Split a single .spacy file into train, dev, and test sets with specified ratios.
    
    Args:
        input_file (str): Path to the input .spacy file
        output_dir (str): Directory to save the split files
        train_split (float): Proportion for training set (default: 0.7)
        dev_split (float): Proportion for development set (default: 0.2)
        test_split (float): Proportion for test set (default: 0.1)
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: Paths to the train, dev, and test files
    """
    if abs(train_split + dev_split + test_split - 1.0) > 1e-10:
        raise ValueError(f"Split proportions must sum to 1.0, got {train_split + dev_split + test_split}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    doc_bin = DocBin().from_disk(input_file)
    docs = list(doc_bin.get_docs(spacy.blank("en").vocab))
    random.seed(seed)
    random.shuffle(docs)
    n_docs = len(docs)
    train_end = int(n_docs * train_split)
    dev_end = train_end + int(n_docs * dev_split)
    train_docs = docs[:train_end]
    dev_docs = docs[train_end:dev_end]
    test_docs = docs[dev_end:] 
    print(f"Splitting {n_docs} documents into:")
    print(f"  - Train: {len(train_docs)} docs ({len(train_docs)/n_docs:.1%})")
    print(f"  - Dev:   {len(dev_docs)} docs ({len(dev_docs)/n_docs:.1%})")
    print(f"  - Test:  {len(test_docs)} docs ({len(test_docs)/n_docs:.1%})")
    train_file = output_path / "train.spacy"
    dev_file = output_path / "dev.spacy"
    test_file = output_path / "test.spacy"
    train_doc_bin = DocBin()
    for doc in train_docs:
        train_doc_bin.add(doc)
    train_doc_bin.to_disk(train_file)    
    dev_doc_bin = DocBin()
    for doc in dev_docs:
        dev_doc_bin.add(doc)
    dev_doc_bin.to_disk(dev_file)    
    test_doc_bin = DocBin()
    for doc in test_docs:
        test_doc_bin.add(doc)
    test_doc_bin.to_disk(test_file)
    return str(train_file), str(dev_file), str(test_file)


def evaluate_model(model_path, eval_data_path, spans_key="sc"):
    """
    Evaluate a trained spaCy model on a dataset and print detailed metrics.
    
    Args:
        model_path: Path to the trained model
        eval_data_path: Path to the evaluation .spacy file
        spans_key: The key used for span categorization in your model
        
    Returns:
        dict: Evaluation scores
    """
    print(f"Loading model from {model_path}")
    nlp = spacy.load(model_path)
    print(f"Loading evaluation data from {eval_data_path}")
    doc_bin = DocBin().from_disk(eval_data_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    examples = []
    for gold in docs:
        pred = nlp.make_doc(gold.text)
        example = Example(pred, gold)
        examples.append(example)
    scores = nlp.evaluate(examples)
    print("\n=== EVALUATION RESULTS ===")
    print(f"Spans F-score: {scores['spans_sc_f']:.4f}")
    print(f"Spans Precision: {scores['spans_sc_p']:.4f}")
    print(f"Spans Recall: {scores['spans_sc_r']:.4f}")
    if "spancat" in nlp.pipe_names:
        spancat = nlp.get_pipe("spancat")
        detailed_scores = spancat.scorer.score_spans(
            examples, 
            spans_key=spans_key
        )        
        print("\n=== DETAILED SCORES BY LABEL ===")
        for label, scores_dict in detailed_scores.items():
            if label.startswith('spans_'):
                continue
            print(f"\nLabel: {label}")
            print(f"  F-score: {scores_dict['f']:.4f}")
            print(f"  Precision: {scores_dict['p']:.4f}")
            print(f"  Recall: {scores_dict['r']:.4f}")
    
    return scores


def train_model(config_path, output_dir, train_file, dev_file, test_file=None, spans_key="sc", use_gpu=-1):
    """
    Train a spaCy model using the specified config and data files, then evaluate it.
    
    Args:
        config_path (str): Path to the training config file
        output_dir (str): Directory to save trained model
        train_file (str): Path to the training data
        dev_file (str): Path to the development data
        test_file (str, optional): Path to the test data for final evaluation
        spans_key (str): Key used for spans in the model
        use_gpu (int): GPU device ID, -1 for CPU
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_path = Path(train_file)
    dev_path = Path(dev_file)
    config_path = Path(config_path)
    overrides = {
        "paths.train": str(train_path),
        "paths.dev": str(dev_path),
    }
    from spacy.cli.train import train
    print("\n=== STARTING MODEL TRAINING ===")
    train(
        config_path=config_path,
        output_path=output_path,
        overrides=overrides,
        use_gpu=use_gpu
    )    
    print(f"\nModel trained and saved to {output_path}")
    print("\n=== EVALUATING MODEL ON DEV SET ===")
    dev_scores = evaluate_model(output_path, dev_file, spans_key)
    if test_file:
        print("\n=== EVALUATING MODEL ON TEST SET ===")
        test_scores = evaluate_model(output_path, test_file, spans_key)
        dev_f = dev_scores['spans_sc_f']
        test_f = test_scores['spans_sc_f']
        diff = dev_f - test_f
        print("\n=== OVERFITTING ANALYSIS ===")
        print(f"Dev F-score: {dev_f:.4f}")
        print(f"Test F-score: {test_f:.4f}")
        print(f"Difference: {diff:.4f}")
        if diff > 0.05:
            print("WARNING: Model may be overfitting (>5% difference between dev and test scores)")
        elif diff < -0.05:
            print("UNUSUAL: Model performs better on test set than dev set. Check your data splits.")
        else:
            print("Model generalization looks good (similar performance on dev and test sets)")


def main(
    input_file: str = typer.Argument(..., help="Path to input .spacy file"),
    output_dir: str = typer.Argument(..., help="Directory to save split files and model"),
    config_path: str = typer.Argument(..., help="Path to spaCy training config file"),
    train_split: float = typer.Option(0.7, help="Proportion for training set"),
    dev_split: float = typer.Option(0.2, help="Proportion for development set"),
    test_split: float = typer.Option(0.1, help="Proportion for test set"),
    spans_key: str = typer.Option("sc", help="Key used for spans in the model"),
    use_gpu: int = typer.Option(-1, help="GPU device ID, -1 for CPU"),
    skip_train: bool = typer.Option(False, help="Skip training and only evaluate an existing model"),
    seed: int = typer.Option(42, help="Random seed for reproducibility")
):
    """
    Split a single .spacy file into train/dev/test sets, train a spaCy model, and evaluate it.
    """
    splits_dir = Path(output_dir) / "splits"
    model_dir = Path(output_dir) / "model"
    if not skip_train:
        train_file, dev_file, test_file = split_spacy_file(
            input_file, 
            str(splits_dir), 
            train_split, 
            dev_split, 
            test_split, 
            seed
        )
        train_model(
            config_path, 
            str(model_dir), 
            train_file, 
            dev_file, 
            test_file,
            spans_key,
            use_gpu
        )
        print("\nDone! Files saved to:")
        print(f"  - Train: {train_file}")
        print(f"  - Dev:   {dev_file}")
        print(f"  - Test:  {test_file}")
        print(f"  - Model: {model_dir}")
    else:
        train_file = str(splits_dir / "train.spacy")
        dev_file = str(splits_dir / "dev.spacy")
        test_file = str(splits_dir / "test.spacy")
        if not model_dir.exists():
            print(f"Error: Model directory {model_dir} does not exist. Cannot evaluate.")
            return
        print("\n=== EVALUATING EXISTING MODEL ===")
        print("\n=== DEV SET EVALUATION ===")
        dev_scores = evaluate_model(str(model_dir), dev_file, spans_key)
        print("\n=== TEST SET EVALUATION ===")
        test_scores = evaluate_model(str(model_dir), test_file, spans_key)
        dev_f = dev_scores['spans_sc_f']
        test_f = test_scores['spans_sc_f']
        diff = dev_f - test_f
        print("\n=== OVERFITTING ANALYSIS ===")
        print(f"Dev F-score: {dev_f:.4f}")
        print(f"Test F-score: {test_f:.4f}")
        print(f"Difference: {diff:.4f}")
        if diff > 0.05:
            print("WARNING: Model may be overfitting (>5% difference between dev and test scores)")
        elif diff < -0.05:
            print("UNUSUAL: Model performs better on test set than dev set. Check your data splits.")
        else:
            print("Model generalization looks good (similar performance on dev and test sets)")


if __name__ == "__main__":
    typer.run(main)