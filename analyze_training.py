#!/usr/bin/env python3
"""
Analyze Previous Training Performance
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _latest_checkpoint(base_dir: Path) -> Path | None:
    candidates = []
    for path in base_dir.glob("step_*/"):
        try:
            step = int(path.name.split("_")[-1])
        except ValueError:
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def analyze_previous_training(checkpoint_root: Path | None = None) -> dict[str, object]:
    """Analyze the previous training run."""

    checkpoint_root = checkpoint_root or Path("training/checkpoints/web-browsing-qwen3")

    if not checkpoint_root.exists():
        logger.warning("No checkpoints directory found at %s", checkpoint_root)
        return {}

    latest_dir = _latest_checkpoint(checkpoint_root)
    if not latest_dir:
        logger.warning("No step_* checkpoints present under %s", checkpoint_root)
        return {}

    state_path = latest_dir / "training_state.json"

    if not state_path.exists():
        logger.warning("No training_state.json found in %s", latest_dir)
        return {}

    with state_path.open('r', encoding="utf-8") as f:
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
    checkpoint_base = checkpoint_root
    checkpoints = []
    for path in checkpoint_base.glob("step_*/"):
        checkpoints.append(path.name.rstrip("/"))
    checkpoints.sort(key=lambda name: int(name.split("_")[-1]))
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
        "can_resume": bool(checkpoints),
        "latest_checkpoint": latest_dir.name,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GRPO training checkpoints")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("training/checkpoints/web-browsing-qwen3"),
        help="Directory containing step_* checkpoint folders",
    )
    args = parser.parse_args()

    analyze_previous_training(args.checkpoint_root)


if __name__ == "__main__":
    main()
