import gradio as gr
from transformers import pipeline

LABEL_MAP = {
    "unripe watermelon": "🌱 미숙",
    "perfectly ripe watermelon": "🍉 적당함",
    "overripe watermelon": "🍂 과숙",
}
CANDIDATE_LABELS = [
    "unripe watermelon",
    "perfectly ripe watermelon",
    "overripe watermelon",
]

classifier = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32"
)


def predict(image):
    if image is None:
        return "이미지를 업로드해주세요."

    results = classifier(
        image,
        candidate_labels=CANDIDATE_LABELS,
        multi_label=False,
    )

    if isinstance(results, list) and len(results) > 0:
        best = results[0]
        label = best.get("label", "")
        score = best.get("score", 0.0)
        korean_label = LABEL_MAP.get(label, "알 수 없음")
        return f"{korean_label} ({score*100:.1f}%)"

    return "판별할 수 없습니다. 다시 시도해주세요."

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="수박 사진 업로드"),
    outputs=gr.Text(label="판별 결과"),
    title="🍉 수박 생장 정도 판별 시스템",
    description="Hugging Face zero-shot-image-classification을 이용해 수박 이미지를 판별합니다.",
)

if __name__ == "__main__":
    iface.launch(share=True)