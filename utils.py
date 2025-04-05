from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")


mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")

mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")


lang_code_map = {
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
    tokenizer = mbart_tokenizer
    model = mbart_model

    tokenizer.src_lang = "en_XX"
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_language]
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
