from fashion_clip.fashion_clip import FashionCLIP
import os
import torch
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

image_folder = "images"
batch_size = 256

# collect image paths
image_paths = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# keep filenames separately
filenames = [os.path.basename(p) for p in image_paths]

print(f"Processing {len(image_paths)} images")

model = FashionCLIP("fashion-clip")
model.model = model.model.to(device)

embeddings = []

for i in tqdm(range(0, len(image_paths), batch_size)):
    batch_paths = image_paths[i:i + batch_size]
    batch_emb = model.encode_images(batch_paths, batch_size=len(batch_paths))
    embeddings.extend(batch_emb)

embeddings = np.array(embeddings)

# create folder if not exists
os.makedirs("embeddings", exist_ok=True)

# save embeddings
np.save("embeddings/fashion_embeddings.npy", embeddings)

# save filenames
np.save("embeddings/image_filenames.npy", np.array(filenames))

print("Embeddings saved!")
print("Embeddings shape:", embeddings.shape)
print("Filenames saved:", len(filenames))