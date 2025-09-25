#!/usr/bin/env python3
"""
GRPO Training Script with MLX Sampler for Non-Deterministic Rollouts
"""

import json
import logging
import os
import shutil
from glob import glob
from typing import List, Dict, Any

from mlx_lm import load
from web_browsing_reward import WebBrowsingReward
from grpo_trainer import GRPOTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GRPOTrainingConfig:
    """Configuration for GRPO training with MLX sampler."""
    
    def __init__(self):
        # Training parameters
        self.num_epochs = 1
        self.steps_per_epoch = 20
        self.save_every = 5
        self.eval_every = 5
        
        # Model parameters
        self.model_name = "Qwen/Qwen3-0.6B"
        self.max_tokens = 100
        
        # GRPO parameters with MLX sampler
        self.learning_rate = 5e-7  # Further reduced to prevent backwards learning
        self.rollouts_per_step = 8
        self.temperature = 0.8  # For non-deterministic generation
        self.top_p = 0.95
        self.top_k = 50
        
        # Data paths
        self.training_data_path = "data/train_data_structured.json"
        self.test_data_path = "data/test_data_structured.json"
        
        # Output paths
        self.checkpoint_dir = "training/checkpoints/grpo"
        self.output_dir = "training/outputs/grpo"
        # Final export directory for the trained model (mirrors the last checkpoint)
        self.final_model_dir = "training/models/grpo/final"

def load_training_data(data_path: str) -> List[Dict[str, Any]]:
    """Load training data from JSON file."""
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} training examples from {data_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return []

def create_base_model(model_name: str):
    """Create base Qwen model wrapper."""
    try:
        model, tokenizer = load(model_name)
        logger.info(f"Loaded base Qwen model: {model_name}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def analyze_training_performance(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze training performance."""
    rewards = stats.get('total_rewards', [])
    kl_history = stats.get('kl_history', [])
    
    if not rewards:
        return {}
    
    # Basic reward analysis
    avg_reward = sum(rewards) / len(rewards)
    min_reward = min(rewards)
    max_reward = max(rewards)
    
    # KL analysis
    avg_kl = sum(kl_history) / len(kl_history) if kl_history else 0.0
    max_kl = max(kl_history) if kl_history else 0.0
    
    # Improvement analysis
    first_half = rewards[:len(rewards)//2]
    second_half = rewards[len(rewards)//2:]
    
    first_half_avg = sum(first_half) / len(first_half) if first_half else 0
    second_half_avg = sum(second_half) / len(second_half) if second_half else 0
    
    improvement = second_half_avg - first_half_avg
    improvement_pct = (improvement / first_half_avg * 100) if first_half_avg > 0 else 0
    
    return {
        "total_steps": len(rewards),
        "average_reward": avg_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "reward_std": (sum((r - avg_reward)**2 for r in rewards) / len(rewards))**0.5,
        "average_kl": avg_kl,
        "max_kl": max_kl,
        "improvement": improvement,
        "improvement_percent": improvement_pct
    }

def train(config: GRPOTrainingConfig):
    """Main training function."""
    try:
        # Load training data
        training_data = load_training_data(config.training_data_path)
        test_data = load_training_data(config.test_data_path)
        
        if not training_data:
            logger.error("No training data loaded!")
            return
        
        # Create base model
        model, tokenizer = create_base_model(config.model_name)
        if model is None or tokenizer is None:
            logger.error("Failed to load model!")
            return
        
        # Create reward function
        reward_function = WebBrowsingReward()
        
        # Create GRPO trainer with MLX sampler
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            tasks=training_data,
            reward_function=reward_function,
            learning_rate=config.learning_rate,
            rollouts_per_step=config.rollouts_per_step,
            save_dir=config.checkpoint_dir,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k
        )
        
        logger.info("Starting GRPO training with MLX sampler...")
        logger.info(f"Training for {config.num_epochs} epochs, {config.steps_per_epoch} steps each")
        logger.info(f"Features: MLX sampler (temp={config.temperature}, top_p={config.top_p}, top_k={config.top_k})")
        logger.info(f"Model will be saved every {config.save_every} steps")
        
        # Training loop
        for epoch in range(config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")
            
            for step in range(config.steps_per_epoch):
                # Execute training step
                reward = trainer.train_step()
                
                # Log progress
                logger.info(f"Step {step + 1}/{config.steps_per_epoch}, Reward: {reward:.4f}")
                
                # Save checkpoint
                if (step + 1) % config.save_every == 0:
                    logger.info(f"Saving checkpoint at step {step + 1}...")
                    trainer.save_model(step + 1, epoch + 1)
                    
                    # Evaluate on test set
                    logger.info("Evaluating on test set...")
                    test_reward = trainer.evaluate(test_data)
                    logger.info(f"Test evaluation - Average reward: {test_reward:.4f}")
            
            # End of epoch analysis
            stats = trainer.get_training_stats()
            analysis = analyze_training_performance(stats)
            
            if analysis:
                logger.info("=== EPOCH SUMMARY ===")
                logger.info(f"Total steps: {analysis['total_steps']}")
                logger.info(f"Average reward: {analysis['average_reward']:.4f}")
                logger.info(f"Reward range: {analysis['min_reward']:.4f} - {analysis['max_reward']:.4f}")
                logger.info(f"Improvement: {analysis['improvement']:.4f} ({analysis['improvement_percent']:.1f}%)")
                logger.info(f"Average KL: {analysis['average_kl']:.4f}")
        
        # Final save
        logger.info("Training complete! Saving final model...")
        trainer.save_model(config.steps_per_epoch, config.num_epochs)

        # Export the current in-memory model directly (not by copying a checkpoint)
        try:
            final_export_dir = config.final_model_dir
            os.makedirs(final_export_dir, exist_ok=True)
            trainer.export_model(final_export_dir, step=trainer.step_count, epoch=config.num_epochs)

            # Refresh a convenient 'latest' symlink alongside the final export
            latest_link = os.path.join(os.path.dirname(final_export_dir), "latest")
            try:
                if os.path.islink(latest_link) or os.path.exists(latest_link):
                    os.unlink(latest_link)
            except Exception:
                pass
            try:
                os.symlink(final_export_dir, latest_link)
            except FileExistsError:
                pass
            except OSError:
                # Symlinks can fail on some filesystems; ignore silently
                pass

            logger.info(
                f"Exported final model to {final_export_dir} and updated 'latest' symlink"
            )
        except Exception as e:
            logger.error(f"Failed to export final model: {e}")
        
        # Final analysis
        final_stats = trainer.get_training_stats()
        final_analysis = analyze_training_performance(final_stats)
        
        if final_analysis:
            logger.info("=== FINAL TRAINING SUMMARY ===")
            logger.info(f"Total steps: {final_analysis['total_steps']}")
            logger.info(f"Final average reward: {final_analysis['average_reward']:.4f}")
            logger.info(f"Overall improvement: {final_analysis['improvement']:.4f} ({final_analysis['improvement_percent']:.1f}%)")
            logger.info(f"Final average KL: {final_analysis['average_kl']:.4f}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

def main():
    """Main entry point."""
    try:
        # Create configuration
        config = GRPOTrainingConfig()
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Start training
        train(config)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 