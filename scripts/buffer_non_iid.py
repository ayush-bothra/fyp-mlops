"""
Non-IID buffer selection evaluation.

Tests whether UnsupervisedSigLIPBuffer actually does better than a naive
reservoir sampling baseline on a REALISTIC stream ordering -- one where
the camera looks at one object continuously before moving to the next,
not a shuffled i.i.d. stream. Ground-truth category labels are used
ONLY for scoring the result afterward, never for selection itself.

Runs on CPU by default so it can be run alongside a GPU-bound job.

Usage:
    python -m scripts.buffer_noniid_evaluation --dataset data/raw/core50/core50_128x128
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.buffer.selector import CoresetBuffer
from src.data.core50 import scan_core50

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def load_cpu_vision_encoder(model_id):
    processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    return model, processor


def extract_embedding(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    pixel_values = pixel_values.view(-1, 3, pixel_values.shape[-2], pixel_values.shape[-1])
    with torch.no_grad():
        vision_outputs = model.model.vision_model(pixel_values=pixel_values)
    pooled = vision_outputs.last_hidden_state.mean(dim=1)
    return pooled.squeeze(0).numpy()


def build_non_iid_stream(samples, session_id, frames_per_object):
    session_samples = [s for s in samples if s.session_id == session_id]
    by_object = defaultdict(list)
    for sample in session_samples:
        by_object[sample.object_id].append(sample)

    stream = []
    for object_id in sorted(by_object):
        object_samples = sorted(by_object[object_id], key=lambda s: s.frame_id)
        stream.extend(object_samples[:frames_per_object])
    return stream


def reservoir_update(buffer_items, new_item, capacity, seen_count):
    if len(buffer_items) < capacity:
        buffer_items.append(new_item)
        return
    replace_index = random.randint(0, seen_count - 1)
    if replace_index < capacity:
        buffer_items[replace_index] = new_item


def buffer_selector_update(selector, embedding, item):
    selector.evaluate_and_add(embedding, item)

def summarize(buffer_items, label_fn, name):
    labels = [label_fn(item) for item in buffer_items]
    counts = Counter(labels)
    total = len(labels)
    proportions = np.array([count / total for count in counts.values()])
    entropy = -np.sum(proportions * np.log(proportions + 1e-12))
    max_class_fraction = max(counts.values()) / total if total > 0 else 0.0
    print(f"{name}: distinct_categories={len(counts)} entropy={entropy:.3f} max_class_fraction={max_class_fraction:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/raw/core50/core50_128x128")
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--frames-per-object", type=int, default=40)
    parser.add_argument("--capacity", type=int, default=200)
    args = parser.parse_args()

    samples = scan_core50(args.dataset)
    stream = build_non_iid_stream(samples, args.session, args.frames_per_object)
    print(f"non-IID stream length: {len(stream)} samples across session {args.session}")

    model, processor = load_cpu_vision_encoder(MODEL_ID)
    embedding_dim = None
    
    reservoir_buffer = []
    selector = None
    
    for index, sample in enumerate(stream):
        embedding = extract_embedding(model, processor, sample.path)
        if embedding_dim is None:
            embedding_dim = embedding.shape[0]
            selector = CoresetBuffer(capacity=args.capacity, window_size=100)
    
        buffer_selector_update(selector, embedding, sample)
        reservoir_update(reservoir_buffer, sample, args.capacity, index + 1)
    
        if (index + 1) % 100 == 0:
            print(f"processed {index + 1}/{len(stream)}")
    
    label_fn = lambda sample: sample.category_name
    print()
    print(Counter(sample.category_name for sample in selector.items))
    summarize(selector.items, label_fn, "CoresetBuffer")
    summarize(reservoir_buffer, label_fn, "reservoir sampling")


if __name__ == "__main__":
    main()
