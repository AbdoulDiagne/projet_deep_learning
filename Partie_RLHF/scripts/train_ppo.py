from __future__ import annotations
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

import torch
from transformers import pipeline
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from adl_alignment.config import load_config
from adl_alignment.data import read_jsonl
from adl_alignment.modeling import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rlhf.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # Sécurité supplémentaire pour PyTorch contre la fragmentation mémoire
    import os
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    tokenizer = load_tokenizer(cfg["model"]["base_name"], cfg["model"]["max_length"])
    
    # --- CONFIGURATION DU LORA ---
    lora_cfg = cfg["model"].get("lora")
    peft_config = None
    if lora_cfg and lora_cfg.get("enabled", True):
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM"
        )

    # --- CHARGEMENT DES MODÈLES ---
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg["model"]["base_name"],
        peft_config=peft_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg["model"]["base_name"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # --- CORRECTIF MEMOIRE 1 : Activer le Gradient Checkpointing pour diviser par 2 la VRAM du modèle ---
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # --- PIPELINE POUR LE REWARD MODEL ---
    reward_pipe = pipeline(
        "text-classification",
        model=cfg["training"]["reward_model_dir"],
        tokenizer=cfg["training"]["reward_model_dir"],
        device=0 if torch.cuda.is_available() else -1,
        function_to_apply="none",
    )

    ppo_config = PPOConfig(
        learning_rate=cfg["ppo"]["learning_rate"],
        batch_size=cfg["ppo"]["batch_size"],
        mini_batch_size=cfg["ppo"]["mini_batch_size"],
        target_kl=cfg["ppo"]["target_kl"],
    )
    
    trainer = PPOTrainer(config=ppo_config, model=model, ref_model=ref_model, tokenizer=tokenizer)

    rows = read_jsonl(cfg["data"]["processed_preferences"])
    prompts = [row["prompt"] for row in rows]

    # --- STOCKAGE DES DONNÉES ---
    ppo_steps = []
    ppo_rewards = []
    
    batch_queries = []
    batch_responses = []
    batch_rewards = []

    total_steps = cfg["ppo"]["steps"]
    step_count = 0

    print(f"🚀 Début de la boucle PPO (Version optimisée VRAM) par lots de {ppo_config.batch_size}...")

    for prompt in prompts:
        if step_count >= total_steps:
            break

        # CORRECTIF MEMOIRE 2 : On force la troncature stricte à 400 tokens maximum pour laisser de la place à la génération
        query = tokenizer.encode(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=400
        ).to(trainer.accelerator.device)[0]
        
        # Génération contrôlée
        response = trainer.generate(query, max_new_tokens=80, do_sample=True, top_p=0.9)
        
        # CORRECTIF MEMOIRE 3 : Sécurité de troncature stricte pour éviter que la combinaison dépasse 512
        query_and_resp = response.squeeze()
        if query_and_resp.shape[0] > 512:
            query_and_resp = query_and_resp[:512]
            
        response_text = tokenizer.decode(query_and_resp, skip_special_tokens=True)
        
        # Score du Reward Model
        reward_value = reward_pipe(response_text)[0]["score"]
        
        # Accumulation lot
        batch_queries.append(query)
        batch_responses.append(query_and_resp[len(query):]) # On ne passe que la réponse générée au step
        batch_rewards.append(torch.tensor(reward_value))
        
        ppo_steps.append(step_count)
        ppo_rewards.append(reward_value)

        if len(batch_queries) == ppo_config.batch_size:
            # Nettoyage préventif du cache CUDA avant l'étape d'optimisation lourde
            torch.cuda.empty_cache()
            
            train_stats = trainer.step(batch_queries, batch_responses, batch_rewards)
            
            batch_queries = []
            batch_responses = []
            batch_rewards = []
            
            if step_count % 10 == 0:
                print(f"PPO step {step_count}/{total_steps}: current_reward={reward_value:.4f}")
                
            step_count += 1

    # --- TRAÇAGE ET SAUVEGARDE ---
    plt.figure(figsize=(10, 5))
    plt.plot(ppo_steps, ppo_rewards, label="Reward Value", color="green", alpha=0.6)
    
    if len(ppo_rewards) > 10:
        import numpy as np
        weights = np.ones(10) / 10
        smoothed_rewards = np.convolve(ppo_rewards, weights, mode='valid')
        plt.plot(ppo_steps[9:], smoothed_rewards, label="Trend (Moving Average)", color="darkgreen", linewidth=2)
        
    plt.xlabel("PPO Steps")
    plt.ylabel("Reward Score")
    plt.title("PPO Policy Optimization - Reward Evolution")
    plt.grid(True)
    plt.legend()
    
    Path("reports").mkdir(parents=True, exist_ok=True)
    plt.savefig("reports/ppo_reward_evolution.png")
    print("Graphique d'évolution PPO sauvegardé dans reports/ppo_reward_evolution.png")

    trainer.save_pretrained(cfg["training"]["aligned_model_dir"])


if __name__ == "__main__":
    main()
