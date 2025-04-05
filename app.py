import gradio as gr
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline

# Summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

# Translation model
mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")

lang_code_map = {
    "English": "en_XX",
    "French": "fr_XX",
    "German": "de_DE",
    "Hindi": "hi_IN",
    "Spanish": "es_XX",
    "Chinese": "zh_CN",
    "Arabic": "ar_AR",
    "Russian": "ru_RU",
    "Tamil": "ta_IN",
    "Malayalam": "ml_IN"
}

def summarize_text(text):
    result = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return result[0]['summary_text']

def translate_text(text, target_language):
    mbart_tokenizer.src_lang = "en_XX"
    encoded = mbart_tokenizer(text, return_tensors="pt")
    generated_tokens = mbart_model.generate(
        **encoded,
        forced_bos_token_id=mbart_tokenizer.lang_code_to_id[lang_code_map[target_language]]
    )
    return mbart_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]



def handle_task(task, input_text, target_language):
    if task == "Summarization":
        return summarize_text(input_text)
    elif task == "Translation":
        return translate_text(input_text, target_language)
    return "Invalid task."

# Gradio UI
gr.Interface(
    fn=handle_task,
    inputs=[
        gr.Dropdown(["Summarization", "Translation"], label="Task"),
        gr.Textbox(label="Enter your text"),
        gr.Dropdown(list(lang_code_map.keys()), label="Target Language (for translation)", value="French")
    ],
    outputs=gr.Textbox(label="Output"),
    title="Multilingual Summarization & Translation",
    description="Powered by BART and mBART models"
).launch()
