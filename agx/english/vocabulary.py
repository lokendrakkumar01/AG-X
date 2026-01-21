"""
Vocabulary Builder
==================

Daily vocabulary and word learning system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random


@dataclass
class Word:
    """Vocabulary word."""
    word: str
    pronunciation: str  # IPA notation
    part_of_speech: str
    definition: str
    synonyms: List[str]
    antonyms: List[str]
    example_sentences: List[str]
    difficulty: str


class VocabularyBuilder:
    """Build vocabulary with daily words."""
    
    # Sample vocabulary database
    WORD_DATABASE = [
        Word(
            word="eloquent",
            pronunciation="/ˈeləkwənt/",
            part_of_speech="adjective",
            definition="Fluent or persuasive in speaking or writing",
            synonyms=["articulate", "fluent", "persuasive"],
            antonyms=["inarticulate", "unclear"],
            example_sentences=[
                "She gave an eloquent speech that moved the audience.",
                "His eloquent writing style earned him many admirers."
            ],
            difficulty="advanced"
        ),
        Word(
            word="accomplish",
            pronunciation="/əˈkʌmplɪʃ/",
            part_of_speech="verb",
            definition="To achieve or complete successfully",
            synonyms=["achieve", "complete", "fulfill"],
            antonyms=["fail", "abandon"],
            example_sentences=[
                "She worked hard to accomplish her goals.",
                "The team accomplished the project ahead of schedule."
            ],
            difficulty="intermediate"
        ),
        Word(
            word="curious",
            pronunciation="/ˈkjʊəriəs/",
            part_of_speech="adjective",
            definition="Eager to know or learn something",
            synonyms=["inquisitive", "interested", "questioning"],
            antonyms=["indifferent", "uninterested"],
            example_sentences=[
                "Children are naturally curious about the world.",
                "I'm curious to know what happened next."
            ],
            difficulty="beginner"
        ),
    ]
    
    def __init__(self):
        """Initialize vocabulary builder."""
        self.words = self.WORD_DATABASE.copy()
    
    def get_daily_words(self, count: int = 5, difficulty: str = None) -> List[Word]:
        """Get daily vocabulary words.
        
        Args:
            count: Number of words to return
            difficulty: Filter by difficulty level
            
        Returns:
            List of Word objects
        """
        filtered_words = self.words
        
        if difficulty:
            filtered_words = [w for w in filtered_words if w.difficulty == difficulty]
        
        # Return random selection
        return random.sample(filtered_words, min(count, len(filtered_words)))
    
    def format_word_card(self, word: Word) -> str:
        """Format word as a study card.
        
        Args:
            word: Word to format
            
        Returns:
            Formatted string
        """
        card = f"""
╔══════════════════════════════════════════════════════════╗
║  {word.word.upper():^54}  ║
╠══════════════════════════════════════════════════════════╣
║  Pronunciation: {word.pronunciation:<41}  ║
║  Part of Speech: {word.part_of_speech:<39}  ║
║                                                          ║
║  Definition:                                             ║
║  {word.definition[:52]:<52}  ║
║                                                          ║
║  Synonyms: {', '.join(word.synonyms[:3]):<44}  ║
╚══════════════════════════════════════════════════════════╝

Example: {word.example_sentences[0] if word.example_sentences else 'N/A'}
"""
        return card.strip()
