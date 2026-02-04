import torch
from models import GPTModel
from utils import get_tokenizer, text_to_token_ids, token_ids_to_text, format_input, generate
from config import get_model_config


def load_model(model_path, model_name="gpt2-medium (355M)", device="cuda"):
    """Load pretrained model."""
    config = get_model_config(model_name)
    model = GPTModel(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, config


def run_inference(model, config, instruction, input_text="", device="cuda", max_new_tokens=256):
    """Run inference on the model."""
    tokenizer = get_tokenizer()
    
    entry = {"instruction": instruction, "input": input_text, "output": ""}
    formatted_input = format_input(entry)
    
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(formatted_input, tokenizer).to(device),
        max_new_tokens=max_new_tokens,
        context_size=config["context_length"],
        eos_id=50256
    )
    
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(formatted_input):]
        .replace("### Response:", "")
        .strip()
    )
    
    return response_text


def main():
    # Configuration
    MODEL_PATH = "/content/drive/MyDrive/gpt2-medium355M-sft.pth"
    MODEL_NAME = "gpt2-medium (355M)"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    
    # Load model
    print("Loading model...")
    model, config = load_model(MODEL_PATH, MODEL_NAME, DEVICE)
    print("Model loaded successfully!")
    
    # Example 1: Synonyms
    print("\n" + "="*50)
    print("Example 1: Synonyms")
    print("="*50)
    instruction = "Write the synonyms of given word"
    input_text = "Evil"
    response = run_inference(model, config, instruction, input_text, DEVICE)
    print(f"Instruction: {instruction}")
    print(f"Input: {input_text}")
    print(f"Response: {response}")
    
    # Example 2: Physics question
    print("\n" + "="*50)
    print("Example 2: Physics Question")
    print("="*50)
    instruction = "What is the formula for speed?"
    response = run_inference(model, config, instruction, "", DEVICE)
    print(f"Instruction: {instruction}")
    print(f"Response: {response}")


if __name__ == "__main__":
    torch.manual_seed(123)
    main()
