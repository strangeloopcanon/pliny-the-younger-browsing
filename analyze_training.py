#!/usr/bin/env python3
"""
Analyze Previous Training Performance
"""

import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_previous_training():
    """Analyze the previous training run."""
    
    # Load the latest checkpoint
    checkpoint_dir = "training/checkpoints/web-browsing-qwen3/step_20"
    state_path = os.path.join(checkpoint_dir, "training_state.json")
    
    if not os.path.exists(state_path):
        logger.error("No training state found!")
        return
    
    with open(state_path, 'r') as f:
        state = json.load(f)
    
    rewards = state['total_rewards']
    
    logger.info(f"\n{'='*60}")
    logger.info("PREVIOUS TRAINING ANALYSIS")
    logger.info(f"{'='*60}")
    
    # Basic statistics
    total_steps = len(rewards)
    avg_reward = sum(rewards) / len(rewards)
    min_reward = min(rewards)
    max_reward = max(rewards)
    
    logger.info(f"Training Summary:")
    logger.info(f"  Total Steps: {total_steps}")
    logger.info(f"  Average Reward: {avg_reward:.4f}")
    logger.info(f"  Min Reward: {min_reward:.4f}")
    logger.info(f"  Max Reward: {max_reward:.4f}")
    logger.info(f"  Reward Range: {max_reward - min_reward:.4f}")
    
    # Performance analysis
    first_half = rewards[:len(rewards)//2]
    second_half = rewards[len(rewards)//2:]
    
    first_half_avg = sum(first_half) / len(first_half) if first_half else 0
    second_half_avg = sum(second_half) / len(second_half) if second_half else 0
    
    improvement = second_half_avg - first_half_avg
    improvement_pct = (improvement / first_half_avg * 100) if first_half_avg > 0 else 0
    
    logger.info(f"\nPerformance Analysis:")
    logger.info(f"  First Half (Steps 1-10): {first_half_avg:.4f}")
    logger.info(f"  Second Half (Steps 11-20): {second_half_avg:.4f}")
    logger.info(f"  Improvement: {improvement:+.4f}")
    logger.info(f"  Improvement %: {improvement_pct:+.1f}%")
    
    # Trend analysis
    x_values = list(range(len(rewards)))
    n = len(rewards)
    if n > 1:
        slope = (n * sum(i * r for i, r in enumerate(rewards)) - sum(x_values) * sum(rewards)) / (n * sum(x**2 for x in x_values) - sum(x_values)**2)
    else:
        slope = 0
    
    trend_direction = "improving" if slope > 0 else "declining" if slope < 0 else "stable"
    
    logger.info(f"\nTrend Analysis:")
    logger.info(f"  Trend Slope: {slope:.6f}")
    logger.info(f"  Trend Direction: {trend_direction}")
    
    # Step-by-step analysis
    logger.info(f"\nStep-by-Step Rewards:")
    for i, reward in enumerate(rewards, 1):
        logger.info(f"  Step {i:2d}: {reward:.4f}")
    
    # Checkpoint availability
    logger.info(f"\nCheckpoint Availability:")
    checkpoint_base = "training/checkpoints/web-browsing-qwen3"
    if os.path.exists(checkpoint_base):
        checkpoints = [d for d in os.listdir(checkpoint_base) if d.startswith("step_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))
        logger.info(f"  Available checkpoints: {checkpoints}")
        logger.info(f"  Can resume from step: {checkpoints[-1] if checkpoints else 'None'}")
    
    # Recommendations
    logger.info(f"\nRecommendations:")
    if improvement_pct > 5:
        logger.info(f"  ✅ Training shows improvement ({improvement_pct:+.1f}%)")
        logger.info(f"  ✅ Continue training from checkpoint")
    elif improvement_pct < -5:
        logger.info(f"  ⚠️  Training shows decline ({improvement_pct:+.1f}%)")
        logger.info(f"  ⚠️  Consider adjusting learning rate or hyperparameters")
    else:
        logger.info(f"  ➖ Training is stable ({improvement_pct:+.1f}%)")
        logger.info(f"  ➖ May need more steps or different approach")
    
    if avg_reward < 0.4:
        logger.info(f"  💡 Average reward ({avg_reward:.4f}) is below target (0.4)")
        logger.info(f"  💡 Consider longer training or better reward function")
    
    return {
        "total_steps": total_steps,
        "average_reward": avg_reward,
        "improvement_percentage": improvement_pct,
        "trend_direction": trend_direction,
        "can_resume": len(checkpoints) > 0 if 'checkpoints' in locals() else False
    }

if __name__ == "__main__":
    analyze_previous_training() 