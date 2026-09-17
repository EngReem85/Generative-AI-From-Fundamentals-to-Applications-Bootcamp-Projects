import spaces

import torch
import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,          # إضافة لدعم GPT
)


# ============================================================
# Device
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Transformer Playground")
print(f"Device: {DEVICE}")
print("=" * 60)


# ============================================================
# Models
# ============================================================

print("\nLoading DistilBERT...")
sentiment_tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

sentiment_model.to(DEVICE)
sentiment_model.eval()


print("\nLoading BERT...")
mlm_tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)

mlm_model = AutoModelForMaskedLM.from_pretrained(
    "google-bert/bert-base-uncased"
)

mlm_model.to(DEVICE)
mlm_model.eval()


print("\nLoading DistilBART...")
summarization_tokenizer = AutoTokenizer.from_pretrained(
    "sshleifer/distilbart-cnn-12-6"
)

summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
    "sshleifer/distilbart-cnn-12-6"
)

summarization_model.to(DEVICE)
summarization_model.eval()


print("\nLoading GPT-2...")                           # تحميل نموذج GPT-2
gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")
gpt_model.to(DEVICE)
gpt_model.eval()
# إضافة رمز الحشو لتجنب التحذيرات
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token


print("\nAll models loaded successfully.")
print("=" * 60)


# ============================================================
# 1. DistilBERT - Sentiment Classification
# ============================================================

@spaces.GPU(duration=30)
def analyze_sentiment(text):

    if not text or not text.strip():
        return "Please enter some text."

    inputs = sentiment_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = sentiment_model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    negative_score = probabilities[0].item()
    positive_score = probabilities[1].item()

    label = "POSITIVE" if positive_score > negative_score else "NEGATIVE"
    score = max(positive_score, negative_score)

    return (
        f"Prediction: {label}\n"
        f"Confidence: {score:.2%}\n\n"
        f"Positive: {positive_score:.2%}\n"
        f"Negative: {negative_score:.2%}"
    )


# ============================================================
# 2. BERT - Masked Language Modeling
# ============================================================

@spaces.GPU(duration=30)
def predict_mask(text):

    if "[MASK]" not in text:
        return "Please include [MASK] in your sentence."

    inputs = mlm_tokenizer(
        text,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    mask_token_index = torch.where(
        inputs["input_ids"] == mlm_tokenizer.mask_token_id
    )[1]

    if len(mask_token_index) == 0:
        return "No [MASK] token detected."

    with torch.no_grad():
        outputs = mlm_model(**inputs)

    logits = outputs.logits

    mask_logits = logits[0, mask_token_index[0]]

    top_tokens = torch.topk(
        mask_logits,
        k=5
    ).indices

    results = []

    for token_id in top_tokens:
        token = mlm_tokenizer.decode([token_id])
        results.append(token.strip())

    return "\n".join(
        [
            "Top predictions:",
            "",
            *[
                f"{i + 1}. {token}"
                for i, token in enumerate(results)
            ]
        ]
    )


# ============================================================
# 3. DistilBART - Summarization
# ============================================================

@spaces.GPU(duration=60)
def summarize_text(text):

    if not text or not text.strip():
        return "Please enter some text."

    inputs = summarization_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        summary_ids = summarization_model.generate(
            **inputs,
            max_length=120,
            min_length=30,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

    summary = summarization_tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


# ============================================================
# 4. GPT-2 - Text Generation (Decoder‑only)
# ============================================================

@spaces.GPU(duration=60)
def generate_text(
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """
    توليد نصوص باستخدام GPT-2.
    """
    if not prompt or not prompt.strip():
        return "Please enter a prompt."

    inputs = gpt_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = gpt_model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=gpt_tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    generated_text = gpt_tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return generated_text


# ============================================================
# Interface
# ============================================================

with gr.Blocks(
    title="Transformer Playground"
) as demo:

    gr.Markdown(
        """
        # 🧠 Transformer Playground
        ### Transformer ≠ GPT
        Explore four different Transformer architectures:
        **DistilBERT → Encoder-only → Classification**
        **BERT → Encoder-only → Masked Language Modeling**
        **DistilBART → Encoder-Decoder → Summarization**
        **GPT-2 → Decoder-only → Text Generation**
        """
    )

    # --------------------------------------------------------
    # DistilBERT
    # --------------------------------------------------------

    with gr.Tab("1️⃣ DistilBERT — Sentiment"):

        gr.Markdown(
            """
            ## Encoder-only Transformer
            DistilBERT reads the input and produces
            representations useful for classification.
            **Task:** Sentiment Classification
            """
        )

        sentiment_input = gr.Textbox(
            label="Enter text",
            placeholder="I really enjoyed this movie!",
            lines=4
        )

        sentiment_button = gr.Button(
            "Analyze Sentiment"
        )

        sentiment_output = gr.Textbox(
            label="Result",
            lines=6
        )

        sentiment_button.click(
            fn=analyze_sentiment,
            inputs=sentiment_input,
            outputs=sentiment_output
        )

    # --------------------------------------------------------
    # BERT
    # --------------------------------------------------------

    with gr.Tab("2️⃣ BERT — Mask Prediction"):

        gr.Markdown(
            """
            ## Encoder-only Transformer
            BERT can understand the surrounding context
            of a word by looking at both directions.
            **Task:** Masked Language Modeling
            Write `[MASK]` where you want BERT to predict a word.
            """
        )

        mask_input = gr.Textbox(
            label="Sentence",
            placeholder="The capital of France is [MASK].",
            lines=3
        )

        mask_button = gr.Button(
            "Predict Mask"
        )

        mask_output = gr.Textbox(
            label="Top Predictions",
            lines=8
        )

        mask_button.click(
            fn=predict_mask,
            inputs=mask_input,
            outputs=mask_output
        )

    # --------------------------------------------------------
    # DistilBART
    # --------------------------------------------------------

    with gr.Tab("3️⃣ DistilBART — Summarization"):

        gr.Markdown(
            """
            ## Encoder-Decoder Transformer
            The encoder understands the input,
            while the decoder generates the output.
            **Task:** Text Summarization
            """
        )

        summary_input = gr.Textbox(
            label="Long Text",
            placeholder="Paste a long article or paragraph here...",
            lines=10
        )

        summary_button = gr.Button(
            "Summarize"
        )

        summary_output = gr.Textbox(
            label="Summary",
            lines=8
        )

        summary_button.click(
            fn=summarize_text,
            inputs=summary_input,
            outputs=summary_output
        )

    # --------------------------------------------------------
    # GPT-2 (جديد)
    # --------------------------------------------------------

    with gr.Tab("4️⃣ GPT — Text Generation"):

        gr.Markdown(
            """
            ## Decoder-only Transformer
            GPT generates text autoregressively, predicting
            the next token given the previous ones.
            **Task:** Open‑ended Text Generation
            """
        )

        with gr.Row():
            gpt_prompt = gr.Textbox(
                label="Prompt",
                placeholder="Once upon a time,",
                lines=3,
                scale=3
            )

        with gr.Row():
            gpt_max_len = gr.Slider(
                label="Max new tokens",
                minimum=10,
                maximum=200,
                value=80,
                step=5
            )
            gpt_temp = gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1
            )
            gpt_top_p = gr.Slider(
                label="Top‑p (nucleus sampling)",
                minimum=0.5,
                maximum=1.0,
                value=0.9,
                step=0.05
            )

        gpt_button = gr.Button("Generate")
        gpt_output = gr.Textbox(
            label="Generated Text",
            lines=10
        )

        gpt_button.click(
            fn=generate_text,
            inputs=[gpt_prompt, gpt_max_len, gpt_temp, gpt_top_p],
            outputs=gpt_output
        )

    # --------------------------------------------------------
    # Architecture Map
    # --------------------------------------------------------

    gr.Markdown(
        """
        ---
        ## 🏗️ Transformer Architecture Map
        | Architecture | Model | Task |
        |---|---|---|
        | Encoder-only | DistilBERT | Classification |
        | Encoder-only | BERT | Masked Language Modeling |
        | Encoder-Decoder | DistilBART | Summarization |
        | Decoder-only | **GPT-2** | **Text Generation** |
        ### Main idea
        > **Transformer is a family of architectures, not a single model.**
        The architecture you choose depends on the task.
        """
    )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    demo.launch()