from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from trl import RewardTrainer, RewardConfig 

# Aligner le path pour trouver adl_alignment
sys.path.append(str(Path(__file__).resolve().parents[1]))

from adl_alignment.config import ensure_dir, load_config
from adl_alignment.data import read_jsonl
from adl_alignment.modeling import load_reward_model, load_tokenizer

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rlhf.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = load_tokenizer(cfg["model"]["base_name"], cfg["model"]["max_length"])
    model = load_reward_model(cfg["model"]["base_name"], tokenizer, cfg["model"].get("lora"))
    rows = read_jsonl(cfg["data"]["processed_preferences"])
    
    # Préparation du dataset avec le texte brut attendu par TRL
    chosen_texts = []
    rejected_texts = []
    for row in rows:
        chosen_texts.append(row["chosen"])
        rejected_texts.append(row["rejected"])
        
    hf_dataset = Dataset.from_dict({
        "chosen": chosen_texts,
        "rejected": rejected_texts
    })

    # --- CONFIGURATION STABILISÉE POUR EMPECHER L'EXPLOSION DES GRADIENTS ---
    training_args = RewardConfig(
        output_dir=cfg["training"]["reward_model_dir"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_length=cfg["model"]["max_length"],
        
        # CORRECTIFS STABILITÉ MULTI-GPU/GPU KAGGLE :
        max_grad_norm=1.0,               # Écrête les gradients trop forts (anti-explosion)
        lr_scheduler_type="cosine",      # Décroissance fluide du learning rate
        warmup_ratio=0.1,                # Démarrage en douceur pendant les 10% premières étapes
        optim="adamw_torch"              # Optimiseur standard robuste
    )

    trainer = RewardTrainer(
        model=model, 
        args=training_args, 
        train_dataset=hf_dataset,
        processing_class=tokenizer
    )
    
    # Lancement de l'entraînement
    print("🚀 Démarrage de l'entraînement du Reward Model (Version stabilisée)...")
    trainer.train()
    
    # --- BLOC DE GÉNÉRATION DE LA COURBE DE LOSS ---
    print("📈 Génération de la courbe de loss...")
    history = trainer.state.log_history
    steps = [log["step"] for log in history if "loss" in log]
    losses = [log["loss"] for log in history if "loss" in log]
    
    if steps and losses:
        ensure_dir(cfg["evaluation"]["results_dir"])
        
        # Nettoyage et conversion des types en float
        losses = [float(l) for l in losses]
        
        plt.figure(figsize=(10, 5))
        plt.plot(steps, losses, label="Loss (Modèle stabilisé)", color="blue", linewidth=2, marker='o')
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Reward Model Training Loss Curve")
        plt.grid(True)
        plt.legend()
        
        graph_path = f"{cfg['evaluation']['results_dir']}/reward_model_loss.png"
        plt.savefig(graph_path)
        print(f"✅ Courbe de loss sauvegardée avec succès dans : {graph_path}")
    else:
        print("⚠️ Impossible de générer la courbe : aucune métrique de loss trouvée dans l'historique.")
    
    # Sauvegarde finale du modèle et du tokenizer
    trainer.save_model(cfg["training"]["reward_model_dir"])
    tokenizer.save_pretrained(cfg["training"]["reward_model_dir"])
    print("💾 Modèle de récompense sauvegardé !")

if __name__ == "__main__":
    main()
