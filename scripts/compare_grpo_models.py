#!/usr/bin/env python3
"""
Test Trained Model vs Original Model
"""

import json
import logging

try:
    from mlx_lm import load, generate as mlx_generate
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "mlx-lm is required for scripts/compare_grpo_models.py. "
        "Install it via `pip install mlx-lm mlx mlx-metal`."
    ) from exc

from web_browsing_reward import WebBrowsingReward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_comparison():
    """Test original model vs trained model on 3 prompts."""
    
    # Load test data
    try:
        with open("data/test_data_structured.json", 'r') as f:
            test_data = json.load(f)
        logger.info(f"Loaded {len(test_data)} test examples")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return
    
    # Load models
    try:
        original_model, tokenizer = load("Qwen/Qwen3-0.6B")
        logger.info("Loaded original Qwen model")
        
        # Try to load the trained model from the latest checkpoint
        import os
        from pathlib import Path
        
        checkpoint_dir = "training/checkpoints/grpo"
        latest_checkpoint = None
        
        if os.path.exists(checkpoint_dir):
            # Find the latest checkpoint
            checkpoints = []
            for item in Path(checkpoint_dir).iterdir():
                if item.is_dir() and item.name.startswith("grpo_step_"):
                    try:
                        step_num = int(item.name.split("_")[-1])
                        checkpoints.append((step_num, str(item)))
                    except ValueError:
                        continue
            
            if checkpoints:
                latest_step, latest_checkpoint = max(checkpoints, key=lambda x: x[0])
                logger.info(f"Found latest checkpoint at step {latest_step}: {latest_checkpoint}")
                
                # Try to load the trained model weights
                try:
                    weights_file = os.path.join(latest_checkpoint, "model.safetensors.npz")
                    if os.path.exists(weights_file):
                        import mlx.core as mx
                        from mlx.utils import tree_unflatten
                        
                        # Load the base model architecture
                        trained_model, _ = load("Qwen/Qwen3-0.6B")
                        
                        # Load and apply the trained weights
                        logger.info(f"Loading trained weights from {weights_file}")
                        weights = mx.load(weights_file)
                        trained_model.update(tree_unflatten(list(weights.items())))
                        mx.eval(trained_model.parameters())
                        
                        logger.info("Successfully loaded trained model with updated weights!")
                    else:
                        logger.warning(f"Weights file not found at {weights_file}, using original model")
                        trained_model = original_model
                except Exception as e:
                    logger.error(f"Error loading trained weights: {e}")
                    trained_model = original_model
            else:
                logger.info("No checkpoints found, using original model")
                trained_model = original_model
        else:
            logger.info("No checkpoint directory found, using original model")
            trained_model = original_model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Create reward function
    reward_function = WebBrowsingReward()
    
    # Test prompts (first 3 from test data)
    test_prompts = test_data[:3]
    
    logger.info(f"\n{'='*80}")
    logger.info("MODEL COMPARISON TEST")
    logger.info(f"{'='*80}")
    
    original_total_reward = 0.0
    trained_total_reward = 0.0
    original_keywords = 0
    trained_keywords = 0
    
    web_keywords = ['search', 'find', 'article', 'news', 'website', 'browse', 'navigate', 'energy', 'renewable', 'web', 'internet']
    
    for i, test_task in enumerate(test_prompts, 1):
        prompt = test_task['prompt']
        goal = test_task.get('goal', 'Unknown goal')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST {i}: {goal}")
        logger.info(f"{'='*60}")
        logger.info(f"Prompt: {prompt[:150]}...")
        
        # Test original model
        logger.info(f"\n📋 ORIGINAL MODEL:")
        original_response = mlx_generate(original_model, tokenizer, prompt, max_tokens=100)
        original_reward = reward_function.calculate_reward(test_task, original_response)
        
        # Count keywords in original response
        original_response_lower = original_response.lower()
        original_found_keywords = [kw for kw in web_keywords if kw in original_response_lower]
        
        logger.info(f"Response: {original_response}")
        logger.info(f"Reward: {original_reward:.4f}")
        logger.info(f"Keywords: {original_found_keywords} ({len(original_found_keywords)}/11)")
        
        # Test trained model
        logger.info(f"\n🚀 TRAINED MODEL:")
        trained_response = mlx_generate(trained_model, tokenizer, prompt, max_tokens=100)
        trained_reward = reward_function.calculate_reward(test_task, trained_response)
        
        # Count keywords in trained response
        trained_response_lower = trained_response.lower()
        trained_found_keywords = [kw for kw in web_keywords if kw in trained_response_lower]
        
        logger.info(f"Response: {trained_response}")
        logger.info(f"Reward: {trained_reward:.4f}")
        logger.info(f"Keywords: {trained_found_keywords} ({len(trained_found_keywords)}/11)")
        
        # Compare responses
        reward_diff = trained_reward - original_reward
        keyword_diff = len(trained_found_keywords) - len(original_found_keywords)
        
        logger.info(f"\n📊 COMPARISON:")
        logger.info(f"Reward Difference: {reward_diff:+.4f}")
        logger.info(f"Keyword Difference: {keyword_diff:+d}")
        
        if reward_diff > 0:
            logger.info(f"✅ Trained model shows improvement in reward")
        elif reward_diff < 0:
            logger.info(f"❌ Trained model shows decline in reward")
        else:
            logger.info(f"➖ No change in reward")
        
        if keyword_diff > 0:
            logger.info(f"✅ Trained model uses more web-related keywords")
        elif keyword_diff < 0:
            logger.info(f"❌ Trained model uses fewer web-related keywords")
        else:
            logger.info(f"➖ No change in keyword usage")
        
        # Accumulate totals
        original_total_reward += original_reward
        trained_total_reward += trained_reward
        original_keywords += len(original_found_keywords)
        trained_keywords += len(trained_found_keywords)
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY COMPARISON")
    logger.info(f"{'='*80}")
    
    avg_original_reward = original_total_reward / 3
    avg_trained_reward = trained_total_reward / 3
    avg_original_keywords = original_keywords / 3
    avg_trained_keywords = trained_keywords / 3
    
    logger.info(f"Original Model:")
    logger.info(f"  Average Reward: {avg_original_reward:.4f}")
    logger.info(f"  Average Keywords: {avg_original_keywords:.2f}")
    
    logger.info(f"Trained Model:")
    logger.info(f"  Average Reward: {avg_trained_reward:.4f}")
    logger.info(f"  Average Keywords: {avg_trained_keywords:.2f}")
    
    total_reward_improvement = avg_trained_reward - avg_original_reward
    total_keyword_improvement = avg_trained_keywords - avg_original_keywords
    
    logger.info(f"\nOverall Improvement:")
    logger.info(f"  Reward: {total_reward_improvement:+.4f}")
    logger.info(f"  Keywords: {total_keyword_improvement:+.2f}")
    
    if total_reward_improvement > 0:
        logger.info(f"✅ Training improved average reward by {total_reward_improvement:.4f}")
    elif total_reward_improvement < 0:
        logger.info(f"❌ Training decreased average reward by {abs(total_reward_improvement):.4f}")
    else:
        logger.info(f"➖ No change in average reward")
    
    if total_keyword_improvement > 0:
        logger.info(f"✅ Training improved keyword usage by {total_keyword_improvement:.2f} keywords")
    elif total_keyword_improvement < 0:
        logger.info(f"❌ Training decreased keyword usage by {abs(total_keyword_improvement):.2f} keywords")
    else:
        logger.info(f"➖ No change in keyword usage")
    
    # Note about model loading
    logger.info(f"\n📝 NOTE:")
    if latest_checkpoint:
        logger.info(f"Loaded trained model weights from checkpoint: {latest_checkpoint}")
        logger.info(f"This comparison shows the actual difference between original and trained models.")
    else:
        logger.info(f"No trained checkpoints found - using original model for both tests.")
        logger.info(f"Run training first to see actual model improvements.")

if __name__ == "__main__":
    test_model_comparison() 
