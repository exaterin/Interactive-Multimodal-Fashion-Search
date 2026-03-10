import os
from pathlib import Path
import numpy as np
from tqdm import tqdm 
import torch
from sentence_transformers import SentenceTransformer

class GMEEncoder:
    def __init__(self, model_name="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"):
        self.device = "cpu"

        # Load model with remote code
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            trust_remote_code=True
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
            show_progress_bar=False
        )

    def encode_image(self, images):
        if isinstance(images, str):
            images = [images]

        inputs = [{"image": img} for img in images]

        return self.model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def encode_fused(self, texts, images):
        inputs = [{"text": t, "image": i} for t, i in zip(texts, images)]

        return self.model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )



def list_images(folder: str, exts=(".jpg", ".jpeg", ".png")):
    folder = Path(folder)
    paths = []
    for ext in exts:
        paths.extend(folder.rglob(f"*{ext}"))
        paths.extend(folder.rglob(f"*{ext.upper()}"))
    # stable order
    paths = sorted(set(paths))
    return [str(p) for p in paths]

def embed_folder_images(
    encoder,
    folder: str,
    out_file: str = "image_embeddings_gme.npz",
    batch_size: int = 16,
):
    image_paths = list_images(folder)
    if not image_paths:
        raise ValueError(f"No images found in folder: {folder}")

    all_embs = []
    for i in tqdm(range(0, len(image_paths), batch_size),
                  desc="Embedding images",
                  unit="batch"):

        batch_paths = image_paths[i:i + batch_size]
        embs = encoder.encode_image(batch_paths)
        all_embs.append(embs)

    embeddings = np.vstack(all_embs).astype(np.float32)

    np.savez_compressed(
        out_file,
        embeddings=embeddings,
        paths=np.array(image_paths, dtype=object),
        dim=np.array([embeddings.shape[1]], dtype=np.int32),
    )

    print(f"Saved {len(image_paths)} embeddings to: {out_file}")
    print(f"Shape: {embeddings.shape} (N, D)")
    return out_file

# ---- usage ----
encoder = GMEEncoder()
folder = "data/images"
embed_folder_images(encoder, folder, out_file="embs.npz", batch_size=8)