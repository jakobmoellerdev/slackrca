# rca/analyzer.py
import openai

openai.api_key = 'your-openai-api-key'

def analyze_thread(messages):
    all_text = ' '.join([msg['text'] for msg in messages])
    prompt = f"Summarize the root cause analysis from the following conversation:\n\n{all_text}"

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()
