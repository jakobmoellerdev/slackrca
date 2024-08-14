# rca/analyzer.py
import logging
import os

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.cache_utils import HybridCache

torch.set_float32_matmul_precision('high')


token = os.getenv("HUGGINGFACE_TOKEN")

if token is None:
    raise ValueError("HUGGINGFACE_TOKEN not found. Please set the environment variable according to your access.")

login(token=token)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load a pre-trained model and tokenizer from Hugging Face
model_name = "google/gemma-2-2b-it"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log the start of the model loading process
logger.info("Loading the tokenizer and model...")
model = AutoModelForCausalLM.from_pretrained(model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.to(device)
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
logger.info(f"Model loaded and running on {device}.")

tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
logger.info("Tokenizer loaded.")

# set-up k/v cache
past_key_values = HybridCache(
    config=model.config,
    max_batch_size=1,
    max_cache_len=model.config.max_position_embeddings,
    device=model.device,
    dtype=model.dtype
)
# enable passing kv cache to generate
model._supports_cache_class = True
model.generation_config.cache_implementation = None

# Initialize the text generation pipeline with the model and tokenizer
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def analyze_thread(messages):
    all_text = ' '.join([msg['text'] for msg in messages])

    template = "You are an AI assistant specialized in identifying and analyzing root causes from conversation logs. " \
            "Given the following conversation, provide a comprehensive root cause analysis. Be detailed and structured " \
            "in your response, covering all aspects of the conversation. Consider technical details, communication issues, " \
            "and any other relevant factors. Your answer will be used to generate a PDF from html so output everything as a formatted HTML Document." \
            "Any other format is unacceptable because we will use it for machine parsing. Do not wrap the output in markdown" \
            "\n\n" \
            "Translate this conversation into an RCA:" \
            f"\n{all_text}"
    
    # Log the start of the inference process
    logger.info("Generating response...")
    prompt = [{"role": "user", "content": template}]
    output = text_generator(prompt, max_new_tokens=2048, num_return_sequences=1, past_key_values=past_key_values)

    # Log the completion of the inference process
    logger.info("Response generated.")

    content = output[0]['generated_text'][1]['content']

    return content


with open('rca/RootCauseAnalysis_RCA_Template.html') as f:
    rca_template = f.read()


def convert_to_template(summary):
    template = ("You are a Converter that takes a given Input HTML and converts it to a Root Cause Analysis Summary based on a given template"
                "The input HTML will be a rough version of the RCA which you will have to adapt to the template."
                "The output should be a formatted HTML Document that is structured and detailed."
                "The output should be in the format of the template."
                "Any other format is unacceptable because we will use it for machine parsing."
                "Do not wrap the output in markdown."
                "\n\n"
                "Use the following template to convert the HTML to an RCA Summary:"
                f"\n{rca_template}"
                "\n\n"
                "Convert this HTML to an RCA Summary (also in HTML) according to the template above, raw output, no wrapping in markdown:"
                f"\n{summary}")

    # Log the start of the inference process
    logger.info("Generating template conversion...")
    prompt = [{"role": "user", "content": template}]
    output = text_generator(prompt, max_length=8192, num_return_sequences=1, past_key_values=past_key_values)

    # Log the completion of the inference process
    logger.info("Conversion completed.")

    content = output[0]['generated_text'][1]['content']

    return content