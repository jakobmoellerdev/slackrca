# rca/analyzer.py
import logging
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

login(token="***REMOVED***")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load a pre-trained model and tokenizer from Hugging Face
model_name = "CohereForAI/c4ai-command-r-plus-4bit" # TODO change to IBM model, e.g. granite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log the start of the model loading process
logger.info("Loading the tokenizer and model...")

model = AutoModelForCausalLM.from_pretrained(model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.to(device)
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
logger.info(f"Model loaded and running on {device}.")

tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
logger.info("Tokenizer loaded.")

def analyze_thread(messages):
    all_text = ' '.join([msg['text'] for msg in messages])
    prompt = (
        "You are an AI assistant specialized in identifying and analyzing root causes from conversation logs. "
        "Given the following conversation, provide a comprehensive root cause analysis. Be detailed and structured "
        "in your response, covering all aspects of the conversation. Consider technical details, communication issues, "
        "and any other relevant factors.\n\n"
        f"Conversation:\n{all_text}\n\n"
        "Root Cause Analysis:\n---\n"
    )

    # Log the start of the inference process
    logger.info("Generating response...")

    # Format message with the command-r-plus chat template
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

    # Generate response
    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3,
    )
    summary = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

    # Log the completion of the inference process
    logger.info("Response generated.")

    # Post-process the output to remove the prompt
    if "---" in summary:
        summary = summary.split("---")[1].strip()

    return summary