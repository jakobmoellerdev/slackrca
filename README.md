# SlackRCA - A Slack Bot that generates RCA documents based on LLM text inference

## Overview

SlackRCA is a Slack bot designed to generate Root Cause Analysis (RCA) documents by leveraging Large Language Models (LLMs) for text inference. The bot integrates with Slack to receive conversation threads, analyzes the content to identify root causes, and generates a detailed PDF report.
This PDF report can be shared with stakeholders to provide a detailed summary of the conversation thread and the root cause analysis.

## TLDR

- **Slack Integration**: SlackRCA integrates with Slack to receive conversation threads.
- **Text Analysis**: SlackRCA analyzes the conversation threads to identify root causes.
- **PDF Generation**: SlackRCA generates a detailed PDF report with the summary and root cause analysis. It uses a template to format the report, while a raw text is appended as well for reference.

There is an example of a generated PDF [here](./RCA_Summary.pdf), based on the following mocked conversation to showcase the conversion:
```
mock_messages = [
        {"text": "We encountered an error with the database connection."},
        {"text": "The issue seems to be related to network latency."},
        {"text": "Further investigation revealed a misconfiguration in the firewall settings."}
]
```

Further examples with more detailed slack threads are expected.

## Open TODOs

- [ ] Implement Slack Integration fully and test against instance (once Slack Access granted)
- [ ] Implement Anonymizer / Desensitizer for Slack messages
- [ ] Incorporate Flash Attention for better inference results (currently facing some issues in the model implementation)
- [ ] Speed up inferencing via optimized sampling
- [ ] Optimize Tokenizer (using generic model tokenizer for now)
- [ ] Once the webhook is integrated, remove the raw output, change to file, and find a pdf storage medium
- [ ] Implement Containerization for the application, and setup an e2e test guide for an AWS machine with a powerful enough GPU

## Running Locally

You will need either a powerful enough GPU or a lot of time to run the current model for inferencing. This was tested
on a RTX4090 and a summary generation could still take up to 1 minute. The model is not optimized for speed yet.

Pre-requisites for running the demo case:
- Tested on Python 3.8+
- See `requirements.txt` for the required packages
- A powerful enough GPU with recent bfloat16 precision support

## Design Details / WIP

The Tool uses the model `google/gemma-2-2b-it` for text inference and template conversion (Subject to change, the model will likely be an IBM model with at least 128k context sizing). The model is a large language model trained on a diverse range of text data and can be used for various natural language processing tasks, including text generation, text classification, and text summarization.
The model is not-yet fine-tuned on the generation (TBD via some reference RCAs) to generate root cause analysis reports based on the input text data.
The tool always generates the PDF according to the following segments:

1. **Summary**: A brief summary of the conversation thread.
2. **Root Cause Analysis**: A detailed root cause analysis based on the conversation thread.
3. **Raw Text**: The raw text of the conversation thread for reference.
4. **Observed Conversation Reference**: An observed conversation reference for the user to understand what the model used to generate the RCA.

## Inference Details

The current model makes use of [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) to accelerate inferencing.

## Dataflow

```
       +-------------------+
       |       User        |
       +--------+----------+
                |
                v
+---------------+--------------+
|     Slack Webhook Endpoint   |
|       (FastAPI Application)  |
+---------------+--------------+
                |                                        
                v                                        
+---------------+--------------+
|      Analyze Thread          |
|       (Generate Summary)     |
|   - Input: Slack messages    |
|   - Output: Summary          |
|   - Steps:                   |
|     1. Concatenate messages  |
|     2. Format into prompt    |
|     3. Generate summary      |
+---------------+--------------+
                |
                v
+---------------+--------------+
|      Generate PDF            |
|       (Create Report)        |
|   - Input: Summary           |
|   - Output: PDF              |
|   - Steps:                   |
|     1. Load HTML template    |
|     2. Replace placeholders  |
|     3. Convert to PDF        |
+---------------+--------------+
                |
                v
+---------------+--------------+
|       Return PDF to User     |
|       (FastAPI Application)  |
+---------------+--------------+
                ^
                |
+---------------+--------------+
|     HTML Template            |
|    (Stored Locally)          |
+---------------+--------------+

```

## Disclaimer

Every generated RCA document should be reviewed by a human expert before sharing it with stakeholders. The tool is designed to assist in generating RCA documents based on conversation threads but may not always provide accurate or complete root cause analysis. It is recommended to validate the generated RCA documents and make necessary adjustments before sharing them with stakeholders.
Every generated RCA has a disclaimer as its first page that cannot be removed.