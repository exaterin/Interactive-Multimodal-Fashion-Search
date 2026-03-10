import torch
from sentence_transformers import SentenceTransformer


class GMEEncoder:
    def __init__(self, model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"):
        self.device = "cpu"

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            trust_remote_code=True,
        )

        self.model = self.model.to(torch.float32)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"GME loaded on CPU with float32, dim={self.dim}")

    def encode_text(self, texts, prompt=None):
        if isinstance(texts, str):
            texts = [texts]

        if prompt is not None:
            inputs = [{"text": t, "prompt": prompt} for t in texts]
        else:
            inputs = texts

        return self.model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_image(self, images):
        if isinstance(images, str):
            images = [images]

        inputs = [{"image": img} for img in images]

        return self.model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_fused(self, texts, images):
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(images, str):
            images = [images]

        if len(texts) != len(images):
            raise ValueError("texts and images must have same length")

        inputs = [{"text": t, "image": i} for t, i in zip(texts, images)]

        return self.model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )