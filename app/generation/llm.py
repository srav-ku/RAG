import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from app.config import CONFIG

# Singleton, same pattern as embedder and reranker
_model = None
_tokenizer = None


def get_llm():
    """Loads the LLM in 4-bit quantized form on first call, reuses it after."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print(f"[LLM] Loading {CONFIG.llm_model} in 4-bit...")

        # Configuration for 4-bit quantized loading
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # computation still happens in float16 for stability
            bnb_4bit_quant_type="nf4"              # a quantization scheme designed for neural network weights
        )

        _tokenizer = AutoTokenizer.from_pretrained(CONFIG.llm_model)
        _model = AutoModelForCausalLM.from_pretrained(
            CONFIG.llm_model,
            quantization_config=bnb_config,
            device_map="auto"  # automatically places the model on GPU if available
        )
        print("[LLM] Model loaded.")

    return _model, _tokenizer


def generate_text(prompt: str, max_new_tokens: int = 300) -> str:
    """
    Generates a response to a prompt using the loaded LLM.
    Uses the model's chat template to format the prompt correctly.
    """
    model, tokenizer = get_llm()

    # Wrap our raw prompt in the chat format this model expects
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():  # we're not training, so no need to track gradients - saves memory
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,   # low temperature = more focused, less random answers - good for factual RAG
            pad_token_id=tokenizer.eos_token_id
        )

    # The output includes our original prompt tokens too - slice them off, keep only the new text
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()
