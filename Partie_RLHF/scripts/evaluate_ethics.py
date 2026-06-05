from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from adl_alignment.config import ensure_dir, load_config
from adl_alignment.data import load_ethics_subset
from adl_alignment.evaluation import predict_ethics
from adl_alignment.modeling import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rlhf.yaml")
    parser.add_argument("--model", choices=["baseline", "aligned"], default="aligned")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # 1. Le tokenizer se charge toujours sur la base d'origine
    tokenizer = load_tokenizer(cfg["model"]["base_name"], cfg["model"]["max_length"])
    
    # Détermination du device pour éviter le multi-GPU anarchique
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Chargement adaptatif du modèle sur le device unique cible
    if args.model == "aligned":
        print("🚀 Chargement du modèle ALIGNÉ (Base Qwen + Adaptateur LoRA PPO)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["base_name"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map={"text-model": device} if device == "cuda" else None # On force l'affectation sur le GPU 0
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, cfg["training"]["aligned_model_dir"])
    else:
        print("📦 Chargement du modèle INITIAL (Baseline Qwen)...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["base_name"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map={"text-model": device} if device == "cuda" else None # On force l'affectation sur le GPU 0
        )

    dataset = load_ethics_subset(
        cfg["data"]["ethics_dataset"],
        cfg["data"]["ethics_subset"],
        cfg["data"]["ethics_split"],
        cfg["data"]["max_eval_samples"],
    )

    results = predict_ethics(model, tokenizer, dataset, device=device)
    results_dir = ensure_dir(cfg["evaluation"]["results_dir"])
    output_path = results_dir / f"ethics_{args.model}_results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
