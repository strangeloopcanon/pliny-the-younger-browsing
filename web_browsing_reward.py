#!/usr/bin/env python3
"""
Web Browsing Tasks for HRM + GRPO Training

Uses corrected_trajectories.json to create training examples for
multi-step web navigation and information gathering tasks.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WebBrowsingTask:
    """A web browsing task with goal and expected actions"""
    
    task_id: str
    goal: str
    expected_actions: List[Dict[str, Any]]
    expected_outcome: str
    difficulty: str  # "easy", "medium", "hard"
    domain: str  # e.g., "vercel", "github", "documentation"
    
    def to_training_format(self) -> Dict[str, Any]:
        """Convert to training format"""
        return {
            "task_id": self.task_id,
            "prompt": self._create_prompt(),
            "completion": self._create_completion(),
            "goal": self.goal,
            "expected_actions": self.expected_actions,
            "expected_outcome": self.expected_outcome,
            "difficulty": self.difficulty,
            "domain": self.domain
        }
    
    def _create_prompt(self) -> str:
        """Create the task prompt"""
        # Get starting page info from first expected action
        starting_url = ""
        if self.expected_actions:
            starting_url = self.expected_actions[0].get("target_url", "")
        
        return f"""You are an intelligent web browsing agent. Your task is to navigate through websites to accomplish specific goals.

TASK: {self.goal}

CONTEXT: You are working on a {self.domain} website. You are currently on the page: {starting_url}.

THINKING PROCESS:
1. First, analyze what you need to accomplish and break it down into logical sub-tasks
2. For each step, think about what information you need and where to find it
3. Plan your navigation path step by step with clear reasoning
4. Explain why each action helps achieve your goal

AVAILABLE ACTIONS:
- internal_navigate: Go to a specific URL or page
- click_external_link: Click on a link to another page  
- read_page: Read and understand the current page content

INSTRUCTIONS:
- Break down the task into specific, actionable steps
- For each step, explain what you're looking for and why
- Be specific about what information you expect to find
- Include reasoning for each navigation decision

Please think step by step and provide a detailed plan with reasoning for each action:"""

    def _create_completion(self) -> str:
        """Create the expected completion with step-by-step reasoning"""
        steps = []
        for i, action in enumerate(self.expected_actions, 1):
            action_type = action["type"]
            target = action.get("target_url", action.get("target", "unknown"))
            reasoning = action.get("reasoning", "")
            
            if action_type == "internal_navigate":
                if reasoning:
                    steps.append(f"Step {i}: Navigate to {target}\nReasoning: {reasoning}")
                else:
                    steps.append(f"Step {i}: Navigate to {target}\nReasoning: This page likely contains the information needed for the task")
            elif action_type == "click_external_link":
                if reasoning:
                    steps.append(f"Step {i}: Click on link to {target}\nReasoning: {reasoning}")
                else:
                    steps.append(f"Step {i}: Click on link to {target}\nReasoning: This external page may contain relevant information")
            elif action_type == "read_page":
                steps.append(f"Step {i}: Read and analyze the page content\nReasoning: Need to understand what information is available on this page")
        
        steps.append(f"Final Step: {self.expected_outcome}")
        
        return "\n".join(steps)


class WebBrowsingTaskGenerator:
    """Generates web browsing tasks from trajectories data"""
    
    def __init__(self, trajectories_file: str = "corrected_trajectories.json"):
        self.trajectories_file = trajectories_file
        self.trajectories = self._load_trajectories()
        self.task_templates = self._create_task_templates()
    
    def _load_trajectories(self) -> List[List[Dict[str, Any]]]:
        """Load trajectories from JSON file"""
        with open(self.trajectories_file, 'r') as f:
            data = json.load(f)
        return data.get("trajectories", [])
    
    def _create_task_templates(self) -> List[Dict[str, Any]]:
        """Create task templates with specific, actionable goals"""
        return [
            {
                "goal": "You need to find the pricing page and compare different subscription plans. Look for monthly vs annual pricing, feature comparisons, and any free tier limitations.",
                "pattern": ["settings", "pricing", "billing"],
                "difficulty": "easy",
                "domain": "platform"
            },
            {
                "goal": "You need to configure environment variables for your deployment. Find the deployment settings section, locate the environment variables tab, and understand how to add new variables.",
                "pattern": ["deployments", "settings", "environment"],
                "difficulty": "medium", 
                "domain": "deployment"
            },
            {
                "goal": "You need to find the API documentation to understand available endpoints. Look for the developer docs section, find the API reference, and identify authentication methods.",
                "pattern": ["docs", "api", "reference"],
                "difficulty": "medium",
                "domain": "documentation"
            },
            {
                "goal": "You need to check your repository settings and verify the Git configuration. Find the repository settings page, check branch protection rules, and review deployment settings.",
                "pattern": ["settings", "repository", "git"],
                "difficulty": "hard",
                "domain": "project"
            },
            {
                "goal": "You need to update your profile information and account preferences. Find the user settings page, locate profile editing options, and check privacy settings.",
                "pattern": ["profile", "account", "user", "settings"],
                "difficulty": "easy",
                "domain": "user"
            },
            {
                "goal": "You need to manage team members and their access permissions. Find the team management section, review current members, and understand how to add or remove team access.",
                "pattern": ["team", "members", "permissions", "management"],
                "difficulty": "medium",
                "domain": "team"
            },
            {
                "goal": "You need to check your application's performance metrics and analytics. Find the analytics dashboard, review traffic patterns, and identify performance bottlenecks.",
                "pattern": ["analytics", "dashboard", "metrics", "performance"],
                "difficulty": "medium",
                "domain": "analytics"
            },
            {
                "goal": "You need to review and update your security settings and authentication methods. Find the security configuration page, check two-factor authentication, and review login history.",
                "pattern": ["security", "auth", "authentication", "login"],
                "difficulty": "hard",
                "domain": "security"
            },
            {
                "goal": "You need to set up third-party integrations and API connections. Find the integrations page, review available services, and understand how to configure webhooks.",
                "pattern": ["integrations", "connections", "third-party", "api"],
                "difficulty": "medium",
                "domain": "integrations"
            },
            {
                "goal": "You need to create a backup of your data and export important files. Find the backup section, understand export formats, and initiate a data download.",
                "pattern": ["backup", "export", "data", "download"],
                "difficulty": "easy",
                "domain": "data"
            },
            {
                "goal": "You need to configure your notification preferences and alert settings. Find the notifications page, set up email alerts, and customize notification frequency.",
                "pattern": ["notifications", "alerts", "preferences", "email"],
                "difficulty": "easy",
                "domain": "notifications"
            },
            {
                "goal": "You need to review your billing history and update payment methods. Find the billing section, check recent invoices, and add or update payment information.",
                "pattern": ["billing", "payment", "history", "invoice"],
                "difficulty": "medium",
                "domain": "billing"
            }
        ]
    
    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain from URL"""
        if "vercel.com" in url:
            return "vercel"
        elif "github.com" in url:
            return "github"
        elif "docs." in url:
            return "documentation"
        else:
            return "general"
    
    def _find_matching_trajectory(self, task_template: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Find a trajectory that matches the task pattern"""
        pattern_keywords = task_template["pattern"]
        
        for trajectory in self.trajectories:
            # Check if trajectory contains the pattern keywords
            trajectory_text = " ".join([
                str(step.get("state", {}).get("url", "")) + " " + 
                str(step.get("state", {}).get("title", ""))
                for step in trajectory
            ]).lower()
            
            # More flexible matching - just need some keywords to match
            if any(keyword in trajectory_text for keyword in pattern_keywords):
                return trajectory
        
        # If no exact match, return any trajectory with enough steps
        for trajectory in self.trajectories:
            if len(trajectory) >= 3:  # At least 3 steps
                return trajectory
        
        return None
    
    def _create_expected_actions(self, trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create expected actions from trajectory with better reasoning"""
        actions = []
        
        for i, step in enumerate(trajectory):
            action = step.get("action", {})
            action_type = action.get("type", "")
            target_url = action.get("target_url", "")
            
            # Get page title for better context
            page_title = step.get("state", {}).get("title", "")
            
            if action_type == "internal_navigate":
                if "settings" in target_url.lower() or "settings" in page_title.lower():
                    reasoning = f"Navigate to settings page to find configuration options"
                elif "profile" in target_url.lower() or "account" in target_url.lower():
                    reasoning = f"Navigate to account/profile section to access user settings"
                elif "docs" in target_url.lower() or "documentation" in target_url.lower():
                    reasoning = f"Navigate to documentation section to find API information"
                elif "billing" in target_url.lower() or "pricing" in target_url.lower():
                    reasoning = f"Navigate to billing/pricing section to find cost information"
                elif "team" in target_url.lower() or "members" in target_url.lower():
                    reasoning = f"Navigate to team management section to access member settings"
                elif "deploy" in target_url.lower() or "environment" in target_url.lower():
                    reasoning = f"Navigate to deployment settings to configure environment variables"
                else:
                    reasoning = f"Navigate to {target_url} to continue the task"
                
                actions.append({
                    "type": "internal_navigate",
                    "target_url": target_url,
                    "reasoning": reasoning
                })
            elif action_type == "click_external_link":
                actions.append({
                    "type": "click_external_link", 
                    "target": target_url,
                    "reasoning": f"Click on external link to access {target_url} for additional information"
                })
        
        return actions
    
    def _create_expected_outcome(self, task_template: Dict[str, Any], trajectory: List[Dict[str, Any]]) -> str:
        """Create expected outcome based on task and trajectory"""
        goal = task_template["goal"]
        final_url = trajectory[-1].get("state", {}).get("url", "")
        final_title = trajectory[-1].get("state", {}).get("title", "")
        
        return f"Successfully completed the goal: {goal}. Final location: {final_title} ({final_url})"
    
    def generate_task(self, task_template: Dict[str, Any]) -> Optional[WebBrowsingTask]:
        """Generate a web browsing task from template"""
        trajectory = self._find_matching_trajectory(task_template)
        
        if not trajectory:
            return None
        
        # Extract domain from first URL
        first_url = trajectory[0].get("state", {}).get("url", "")
        domain = self._extract_domain_from_url(first_url)
        
        # Create task
        task = WebBrowsingTask(
            task_id=f"web_browse_{domain}_{random.randint(1000, 9999)}",
            goal=task_template["goal"],
            expected_actions=self._create_expected_actions(trajectory),
            expected_outcome=self._create_expected_outcome(task_template, trajectory),
            difficulty=task_template["difficulty"],
            domain=domain
        )
        
        return task
    
    def generate_training_data(self, num_tasks: int = 100) -> List[Dict[str, Any]]:
        """Generate training data with web browsing tasks"""
        training_data = []
        
        logger.info(f"Generating {num_tasks} web browsing tasks...")
        
        for i in range(num_tasks):
            # Randomly select a task template
            template = random.choice(self.task_templates)
            
            # Generate task
            task = self.generate_task(template)
            
            if task:
                training_data.append(task.to_training_format())
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1} tasks")
        
        logger.info(f"Generated {len(training_data)} web browsing tasks")
        return training_data


class WebBrowsingReward:
    """Reward function for web browsing tasks"""
    
    def __init__(self):
        self.name = "web_browsing_reward"
    
    def calculate_reward(self, task: Dict[str, Any], response: str) -> float:
        """Calculate reward for web browsing task completion with improved multifaceted evaluation"""
        reward = 0.0
        response_lower = response.lower()
        response_upper = response.upper()
        
        # 1. STRUCTURE & ORGANIZATION REWARDS (0.2 points total)
        structure_reward = 0.0
        
        # Reward explicit structure indicators
        if "CONTEXT:" in response_upper:
            structure_reward += 0.05
        if "THINKING PROCESS:" in response_upper or "PROCESS:" in response_upper:
            structure_reward += 0.05
        
        # Reward step-by-step approach
        step_patterns = [f"step {i}" for i in range(1, 10)]
        if any(pattern in response_lower for pattern in step_patterns):
            structure_reward += 0.1
        
        reward += structure_reward
        
        # 2. ACTION TYPE AWARENESS (0.3 points total)
        # Reward understanding of required action types
        expected_actions = task.get("expected_actions", [])
        action_type_reward = 0.0
        
        if expected_actions:
            action_types_mentioned = 0
            unique_action_types = set()
            
            for action in expected_actions:
                action_type = action.get("type", "")
                if action_type:
                    unique_action_types.add(action_type)
                    if action_type in response_lower:
                        action_types_mentioned += 1
            
            # Score based on coverage of unique action types
            if unique_action_types:
                type_coverage = len([t for t in unique_action_types if t in response_lower]) / len(unique_action_types)
                action_type_reward = type_coverage * 0.3
        
        reward += action_type_reward
        
        # 3. GOAL ALIGNMENT & KEYWORD MATCHING (0.25 points total)
        goal = task.get("goal", "").lower()
        goal_keywords = [word for word in goal.split() if len(word) > 3]
        goal_reward = 0.0
        
        if goal_keywords:
            goal_matches = sum(1 for keyword in goal_keywords if keyword in response_lower)
            goal_coverage = min(goal_matches / len(goal_keywords), 1.0)
            goal_reward = goal_coverage * 0.25
        
        reward += goal_reward
        
        # 4. REASONING QUALITY (0.15 points total)
        reasoning_indicators = [
            "because", "since", "therefore", "in order to", "to find", "to locate", 
            "need to", "should", "will help", "to access", "to get", "to check",
            "first", "then", "next", "analyze", "understand"
        ]
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
        reasoning_reward = 0.15 if reasoning_count >= 3 else (0.1 if reasoning_count >= 2 else 0.05 if reasoning_count >= 1 else 0)
        
        reward += reasoning_reward
        
        # 5. COHERENCE & COMPLETENESS (0.1 points total)
        # Enhanced coherence check
        english_words = ["the", "and", "to", "in", "of", "a", "is", "that", "it", "with", "as", "for", "this", "on", "be", "at", "by", "i", "not", "have", "or", "an", "they", "which", "one", "you", "had", "by", "word", "but", "not", "what", "all", "were", "we", "when", "your", "can", "said", "there", "use", "each", "which", "she", "do", "how", "their", "if", "up", "out", "many", "then", "them", "these", "so", "some", "her", "would", "make", "like", "into", "him", "time", "has", "two", "more", "go", "no", "way", "could", "my", "than", "first", "been", "call", "who", "its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may", "part"]
        
        coherent_words = sum(1 for word in english_words if word in response_lower)
        total_words = len(response_lower.split())
        
        coherence_reward = 0.0
        if total_words > 0:
            coherence_ratio = coherent_words / total_words
            # Require at least 30% coherent words and minimum 10 total words
            if coherence_ratio >= 0.3 and total_words >= 10:
                coherence_reward = 0.1
        
        reward += coherence_reward
        
        return min(reward, 1.0)


def main():
    """Test the web browsing task generator"""
    generator = WebBrowsingTaskGenerator()
    
    # Generate a few example tasks
    tasks = generator.generate_training_data(num_tasks=5)
    
    print("Generated Web Browsing Tasks:")
    print("=" * 50)
    
    for i, task in enumerate(tasks, 1):
        print(f"\nTask {i}:")
        print(f"Goal: {task['goal']}")
        print(f"Difficulty: {task['difficulty']}")
        print(f"Domain: {task['domain']}")
        print(f"Expected Actions: {len(task['expected_actions'])} steps")
        print(f"Expected Outcome: {task['expected_outcome'][:100]}...")
        print("-" * 30)
    
    # Test reward function
    reward_fn = WebBrowsingReward()
    sample_response = "Step 1: Navigate to the settings page. Step 2: Find the pricing section. Successfully completed the goal."
    
    for task in tasks[:2]:
        reward = reward_fn.calculate_reward(task, sample_response)
        print(f"Reward for task '{task['task_id']}': {reward:.2f}")


if __name__ == "__main__":
    main() 