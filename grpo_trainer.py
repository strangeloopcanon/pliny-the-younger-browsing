#!/usr/bin/env python3
"""
Proper GRPO Trainer with MLX Sampler for Non-Deterministic Rollouts
"""

import json
import logging
import os
from typing import List, Dict, Any, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

logger = logging.getLogger(__name__)

class UEDTaskSelector:
    """Unsupervised Environment Design task selector."""
    
    def __init__(self, tasks: List[Dict[str, Any]], difficulty_window: int = 10):
        self.tasks = tasks
        self.difficulty_window = difficulty_window
        self.task_difficulties = {}
        self.reward_history = {}
        
        # Initialize difficulties based on task characteristics
        for task in tasks:
            self.estimate_task_difficulty(task)
    
    def estimate_task_difficulty(self, task: Dict[str, Any]) -> float:
        """Estimate task difficulty based on task characteristics."""
        # Simple heuristic: longer goals are harder
        goal_length = len(task.get('goal', ''))
        return min(1.0, goal_length / 200.0)  # Normalize to [0, 1]
    
    def update_difficulty(self, task_id: str, reward: float):
        """Update task difficulty based on performance."""
        if task_id not in self.reward_history:
            self.reward_history[task_id] = []
        
        self.reward_history[task_id].append(reward)
        
        # Keep only recent rewards
        if len(self.reward_history[task_id]) > self.difficulty_window:
            self.reward_history[task_id] = self.reward_history[task_id][-self.difficulty_window:]
        
        # Update difficulty based on average reward (lower reward = higher difficulty)
        avg_reward = sum(self.reward_history[task_id]) / len(self.reward_history[task_id])
        self.task_difficulties[task_id] = max(0.0, 1.0 - avg_reward)
    
    def select_optimal_task(self, current_performance: float) -> Dict[str, Any]:
        """Select optimal task based on current performance and UED principles."""
        # Simple UED: select tasks that are slightly harder than current performance
        target_difficulty = min(1.0, current_performance + 0.1)
        
        # Find tasks close to target difficulty
        best_task = None
        best_diff = float('inf')
        
        for task in self.tasks:
            task_id = task.get('task_id', 'unknown')
            difficulty = self.task_difficulties.get(task_id, 0.5)
            diff = abs(difficulty - target_difficulty)
            
            if diff < best_diff:
                best_diff = diff
                best_task = task
        
        return best_task or self.tasks[0]

class GRPOTrainer:
    """Proper GRPO trainer with MLX sampler for non-deterministic rollouts."""
    
    def __init__(self, model, tokenizer, tasks: List[Dict[str, Any]], reward_function,
                 learning_rate=5e-6, rollouts_per_step=8, save_dir="training/checkpoints",
                 temperature=0.8, top_p=0.95, top_k=50):
        self.model = model
        self.tokenizer = tokenizer
        self.tasks = tasks
        self.reward_function = reward_function
        self.learning_rate = learning_rate
        self.rollouts_per_step = rollouts_per_step
        self.save_dir = save_dir
        
        # MLX sampler for non-deterministic generation
        self.sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        
        # Initialize optimizer
        self.optimizer = Adam(learning_rate=learning_rate)
        
        # UED task selector
        self.ued_selector = UEDTaskSelector(tasks)
        
        # Training statistics
        self.step_count = 0
        self.total_rewards = []
        self.kl_history = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Initialized GRPO trainer with MLX sampler")
        logger.info(f"  - Temperature: {temperature}")
        logger.info(f"  - Top-p: {top_p}")
        logger.info(f"  - Top-k: {top_k}")
        logger.info(f"  - Rollouts per step: {rollouts_per_step}")
        logger.info(f"  - UED Task Selection: Enabled")
    
    def compute_policy_gradients(self, prompt: str, responses: List[str], 
                               rewards: List[float], old_logits: mx.array) -> Tuple[Dict[str, mx.array], float]:
        """Compute policy gradients with proper MLX implementation."""
        try:
            # Normalize rewards (no baseline for simplicity)
            normalized_rewards = rewards
            
            # Log if we have identical rewards (shouldn't happen with sampler)
            if len(set(rewards)) == 1 and len(rewards) > 1:
                logger.warning(f"All rewards identical ({rewards[0]:.4f}) - check sampler!")
            
            def loss_fn():
                """Loss function for policy gradient computation."""
                prompt_tokens = self.tokenizer.encode(prompt)
                prompt_array = mx.array(prompt_tokens, dtype=mx.int32)
                total_loss = mx.array(0.0)
                
                valid_responses = [r for r in responses if r.strip()]
                if not valid_responses:
                    return total_loss
                
                for i, response in enumerate(valid_responses):
                    # Tokenize response
                    response_tokens = self.tokenizer.encode(response)
                    response_array = mx.array(response_tokens, dtype=mx.int32)
                    
                    # Concatenate prompt and response
                    input_ids = mx.concatenate([prompt_array, response_array], axis=0)
                    
                    # Get model logits
                    logits = self.model(input_ids[None, :])  # Add batch dimension
                    
                    # Compute cross-entropy loss
                    log_probs = nn.log_softmax(logits, axis=-1)
                    
                    # Create labels for next-token prediction (shift by 1)
                    labels = input_ids[1:]  # Remove first token, predict next tokens
                    
                    # Ensure labels and log_probs have compatible shapes
                    # log_probs: (1, seq_len, vocab_size), labels: (seq_len-1,)
                    # We need to index log_probs[0, 1:, :] to match labels
                    target_log_probs = mx.take_along_axis(
                        log_probs[0, 1:, :],  # Remove batch dim and first position
                        labels[:, None],      # Add dimension for indexing
                        axis=-1
                    ).squeeze()
                    
                    # Policy gradient loss with baseline: -(reward - baseline) * log_prob
                    baseline = sum(normalized_rewards) / len(normalized_rewards)
                    advantage = normalized_rewards[i] - baseline
                    loss = -advantage * mx.mean(target_log_probs)
                    total_loss += loss
                
                return total_loss / len(valid_responses)
            
            # Compute gradients
            loss, gradients = nn.value_and_grad(self.model, loss_fn)()
            
            # Debug: Check gradients structure
            logger.debug(f"Gradients type: {type(gradients)}")
            logger.debug(f"Gradients keys: {list(gradients.keys()) if gradients else 'Empty'}")
            
            # Handle nested gradient structure from MLX
            if gradients and 'model' in gradients:
                # Extract the actual gradients from the nested structure
                actual_gradients = gradients['model']
                logger.debug(f"Actual gradients keys: {list(actual_gradients.keys()) if actual_gradients else 'Empty'}")
                
                # Skip gradient clipping for now to avoid complex nested structure handling
                # The gradients will be passed through as-is
                logger.debug("Skipping gradient clipping due to complex nested structure")
            
            return gradients, 0.0  # KL will be computed after model update
            
            return gradients, 0.0  # KL will be computed after model update
            
        except Exception as e:
            logger.error(f"Error in policy gradient computation: {e}")
            return {}, 0.0
    
    def update_model_parameters(self, gradients: Dict[str, mx.array]):
        """Update model parameters using computed gradients."""
        try:
            # Apply optimizer step
            self.optimizer.update(self.model, gradients)
            logger.debug(f"Updated model parameters for step {self.step_count}")
            
        except Exception as e:
            logger.error(f"Error updating model parameters: {e}")
    
    def train_step(self) -> float:
        """Execute one training step with non-deterministic rollouts."""
        try:
            # Select task using UED
            current_avg_reward = sum(self.total_rewards[-10:]) / min(10, len(self.total_rewards)) if self.total_rewards else 0.5
            task = self.ued_selector.select_optimal_task(current_avg_reward)
            
            logger.info(f"Step {self.step_count + 1}: Selected task {task.get('task_id', 'unknown')}")
            
            prompt = task['prompt']
            responses = []
            rewards = []
            
            # Get old logits for KL computation
            prompt_tokens = self.tokenizer.encode(prompt)
            prompt_array = mx.array(prompt_tokens, dtype=mx.int32)
            old_logits = self.model(prompt_array[None, :])
            
            # Generate diverse responses using MLX sampler
            for i in range(self.rollouts_per_step):
                # Generate response with sampler for non-deterministic behavior
                response = mlx_generate(
                    self.model, 
                    self.tokenizer, 
                    prompt, 
                    max_tokens=100,
                    sampler=self.sampler  # This creates diversity!
                )
                responses.append(response)
                
                # Calculate reward
                reward = self.reward_function.calculate_reward(task, response)
                rewards.append(reward)
                
                logger.debug(f"Rollout {i+1}: reward = {reward:.4f}")
            
            # Update task difficulty in UED selector
            avg_reward = sum(rewards) / len(rewards)
            self.ued_selector.update_difficulty(task.get('task_id', 'unknown'), avg_reward)
            
            # Compute policy gradients
            gradients, kl_div = self.compute_policy_gradients(prompt, responses, rewards, old_logits)
            
            # Update model parameters
            self.update_model_parameters(gradients)
            
            # Compute KL divergence AFTER model update
            prompt_tokens = self.tokenizer.encode(prompt)
            prompt_array = mx.array(prompt_tokens, dtype=mx.int32)
            new_logits = self.model(prompt_array[None, :])
            
            # Simple KL approximation
            kl_div = mx.mean(mx.square(new_logits - old_logits)).item()
            
            # Update statistics
            self.step_count += 1
            self.total_rewards.append(avg_reward)
            self.kl_history.append(kl_div)
            
            logger.info(f"Step {self.step_count}: Avg reward = {avg_reward:.4f}, KL = {kl_div:.4f}")
            
            return avg_reward
            
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            return 0.0
    
    def save_model(self, step: int, epoch: int = 1):
        """Save model weights and training state."""
        try:
            checkpoint_dir = os.path.join(self.save_dir, f"grpo_step_{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save MLX model weights
            model_path = os.path.join(checkpoint_dir, "model.safetensors.npz")
            self.model.save_weights(model_path)
            logger.info(f"Saved MLX model weights to {model_path}")
            
            # Save tokenizer
            self.tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Saved tokenizer to {checkpoint_dir}")
            
            # Save training state
            state = {
                'step': step,
                'epoch': epoch,
                'total_rewards': self.total_rewards,
                'kl_history': self.kl_history,
                'learning_rate': self.learning_rate
                # Note: optimizer_state contains MLX arrays, so we skip it for JSON serialization
            }
            
            state_path = os.path.join(checkpoint_dir, "training_state.json")
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved complete checkpoint with MLX weights to {checkpoint_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def export_model(self, export_dir: str, step: int = 0, epoch: int = 0):
        """Export the current in-memory model and tokenizer to a specific directory.

        This does not rely on copying from checkpoints; it saves the active weights.
        """
        try:
            os.makedirs(export_dir, exist_ok=True)

            # Save MLX model weights
            model_path = os.path.join(export_dir, "model.safetensors.npz")
            self.model.save_weights(model_path)
            logger.info(f"Exported MLX model weights to {model_path}")

            # Save tokenizer
            self.tokenizer.save_pretrained(export_dir)
            logger.info(f"Exported tokenizer to {export_dir}")

            # Save training state snapshot
            state = {
                'step': step or self.step_count,
                'epoch': epoch,
                'total_rewards': self.total_rewards,
                'kl_history': self.kl_history,
                'learning_rate': self.learning_rate
            }
            state_path = os.path.join(export_dir, "training_state.json")
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)

            logger.info(f"Exported complete model package to {export_dir}")
        except Exception as e:
            logger.error(f"Error exporting model: {e}")

    def load_model(self, checkpoint_path: str):
        """Load model weights and training state."""
        try:
            # Load MLX model weights
            model_path = os.path.join(checkpoint_path, "model.safetensors.npz")
            self.model.load_weights(model_path)
            logger.info(f"Loaded MLX model weights from {model_path}")
            
            # Load training state
            state_path = os.path.join(checkpoint_path, "training_state.json")
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.step_count = state.get('step', 0)
                self.total_rewards = state.get('total_rewards', [])
                self.kl_history = state.get('kl_history', [])
                logger.info(f"Loaded training state from {state_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def evaluate(self, tasks: List[Dict[str, Any]]) -> float:
        """Evaluate model on test tasks."""
        try:
            total_reward = 0.0
            num_tasks = len(tasks)
            
            for task in tasks:
                prompt = task['prompt']
                
                # Generate response with sampler
                response = mlx_generate(
                    self.model, 
                    self.tokenizer, 
                    prompt, 
                    max_tokens=100,
                    sampler=self.sampler
                )
                
                reward = self.reward_function.calculate_reward(task, response)
                total_reward += reward
            
            avg_reward = total_reward / num_tasks
            logger.info(f"Test evaluation - Average reward: {avg_reward:.4f}")
            
            return avg_reward
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return 0.0
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'step_count': self.step_count,
            'total_rewards': self.total_rewards,
            'kl_history': self.kl_history,
            'avg_reward': sum(self.total_rewards) / len(self.total_rewards) if self.total_rewards else 0.0,
            'avg_kl': sum(self.kl_history) / len(self.kl_history) if self.kl_history else 0.0
        } 