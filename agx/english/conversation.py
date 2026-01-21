"""
Conversation Practice
=====================

AI-powered conversation practice scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random


@dataclass
class ConversationScenario:
    """Conversation practice scenario."""
    title: str
    context: str
    difficulty: str
    sample_dialogue: List[tuple[str, str]]  # List of (speaker, text)
    tips: List[str]


class ConversationPractice:
    """Practice conversational English."""
    
    SCENARIOS = [
        ConversationScenario(
            title="Introducing Yourself",
            context="Meeting someone for the first time in a professional setting",
            difficulty="beginner",
            sample_dialogue=[
                ("You", "Hello! I'm Sarah. Nice to meet you."),
                ("Other", "Hi Sarah! I'm John. Pleased to meet you too."),
                ("You", "What do you do for work?"),
                ("Other", "I'm a software engineer. How about you?"),
                ("You", "I work in marketing. It's great to meet you, John!"),
            ],
            tips=[
                "Maintain eye contact and smile",
                "Speak clearly and at a moderate pace",
                "Show genuine interest in the other person",
                "Use polite language and greetings"
            ]
        ),
        ConversationScenario(
            title="Job Interview Questions",
            context="Answering common interview questions professionally",
            difficulty="intermediate",
            sample_dialogue=[
                ("Interviewer", "Can you tell me about yourself?"),
                ("You", "I'm a recent graduate with a degree in Computer Science. I have experience in Python and web development through internships and personal projects."),
                ("Interviewer", "What are your strengths?"),
                ("You", "I'm a fast learner and work well in teams. I'm also detail-oriented and enjoy solving complex problems."),
            ],
            tips=[
                "Use the STAR method (Situation, Task, Action, Result)",
                "Be specific with examples",
                "Stay positive and professional",
                "Ask questions to show interest"
            ]
        ),
    ]
    
    def __init__(self):
        """Initialize conversation practice."""
        self.scenarios = self.SCENARIOS.copy()
    
    def get_scenario(self, difficulty: str = None) -> ConversationScenario:
        """Get a conversation scenario.
        
        Args:
            difficulty: Filter by difficulty level
            
        Returns:
            ConversationScenario
        """
        scenarios = self.scenarios
        
        if difficulty:
            scenarios = [s for s in scenarios if s.difficulty == difficulty]
        
        return random.choice(scenarios) if scenarios else self.scenarios[0]
    
    def format_scenario(self, scenario: ConversationScenario) -> str:
        """Format scenario for display.
        
        Args:
            scenario: Scenario to format
            
        Returns:
            Formatted string
        """
        lines = []
        lines.append(f"=== {scenario.title} ===\n")
        lines.append(f"Context: {scenario.context}\n")
        lines.append(f"Difficulty: {scenario.difficulty.capitalize()}\n")
        lines.append("\nSample Dialogue:")
        
        for speaker, text in scenario.sample_dialogue:
            lines.append(f"  {speaker}: \"{text}\"")
        
        lines.append("\nTips:")
        for i, tip in enumerate(scenario.tips, 1):
            lines.append(f"  {i}. {tip}")
        
        return "\n".join(lines)
