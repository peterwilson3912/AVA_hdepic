from embeddings.BaseEmbeddingModel import BaseEmbeddingModel
from transformers import AutoModel
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image


class JinaCLIPv2(BaseEmbeddingModel):
    def __init__(self, model_type="jinaai/jina-clip-v2", device="cuda"):
        self.device = device if device else "cuda"
        self.model = AutoModel.from_pretrained(
            model_type,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        ).to(self.device).eval()
        self.embedding_dim = 1024

    def get_image_features(self, images):
        batch_size = min(len(images), 64)
        return self._encode_images_in_batches(images, batch_size)

    def get_text_features(self, texts):
        batch_size = min(len(texts), 64)
        return self._encode_texts_in_batches(texts, batch_size)

    @torch.no_grad()
    def _encode_texts_in_batches(self, texts, batch_size):
        text_features = []
        num_batches = len(texts) // batch_size + int(len(texts) % batch_size > 0)
        for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Encoding texts"):
            batch = texts[i:i + batch_size]
            features = self.model.encode_text(batch, truncate_dim=self.embedding_dim)
            if isinstance(features, torch.Tensor):
                features = features.cpu().float().numpy()
            text_features.append(features)
        return np.concatenate(text_features, axis=0)

    @torch.no_grad()
    def _encode_images_in_batches(self, images, batch_size):
        image_features = []
        num_batches = len(images) // batch_size + int(len(images) % batch_size > 0)
        if isinstance(images[0], str):
            images = [Image.open(img).convert("RGB") for img in images]
        for i in tqdm(range(0, len(images), batch_size), total=num_batches, desc="Encoding images"):
            batch = images[i:i + batch_size]
            features = self.model.encode_image(batch, truncate_dim=self.embedding_dim)
            if isinstance(features, torch.Tensor):
                features = features.cpu().float().numpy()
            image_features.append(features)
        return np.concatenate(image_features, axis=0)
