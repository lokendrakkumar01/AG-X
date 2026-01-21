"""
AG-X 2026 Database Models
=========================

Database layer for user management, content storage, and learning analytics.
Uses SQLAlchemy with async support.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, 
    String, Text, Table, UniqueConstraint, Index
)
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

Base = declarative_base()


# =============================================================================
# Enums
# =============================================================================

class UserRole(str, enum.Enum):
    """User role enum."""
    STUDENT = "student"
    EDUCATOR = "educator"
    ADMIN = "admin"


class ContentType(str, enum.Enum):
    """Content type enum."""
    NOTE = "note"
    TUTORIAL = "tutorial"
    CODE_SNIPPET = "code_snippet"
    PROBLEM = "problem"
    PUZZLE = "puzzle"


class DifficultyLevel(str, enum.Enum):
    """Difficulty level enum."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ProblemCategory(str, enum.Enum):
    """DSA problem category enum."""
    ARRAYS = "arrays"
    STRINGS = "strings"
    LINKED_LISTS = "linked_lists"
    STACKS = "stacks"
    QUEUES = "queues"
    TREES = "trees"
    GRAPHS = "graphs"
    RECURSION = "recursion"
    SORTING = "sorting"
    SEARCHING = "searching"
    HASHING = "hashing"
    GREEDY = "greedy"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    BACKTRACKING = "backtracking"
    BIT_MANIPULATION = "bit_manipulation"


# =============================================================================
# Association Tables
# =============================================================================

user_bookmarks = Table(
    "user_bookmarks",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE")),
    Column("content_id", Integer, ForeignKey("content.id", ondelete="CASCADE")),
    Column("created_at", DateTime, server_default=func.now()),
)

problem_tags = Table(
    "problem_tags",
    Base.metadata,
    Column("problem_id", Integer, ForeignKey("problems.id", ondelete="CASCADE")),
    Column("tag", String(50)),
)


# =============================================================================
# User Models
# =============================================================================

class User(Base):
    """User model for authentication and profile."""
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Profile
    full_name: Mapped[Optional[str]] = mapped_column(String(100))
    bio: Mapped[Optional[str]] = mapped_column(Text)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Role and status
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.STUDENT)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    content: Mapped[List["Content"]] = relationship("Content", back_populates="author", cascade="all, delete-orphan")
    progress: Mapped[List["UserProgress"]] = relationship("UserProgress", back_populates="user", cascade="all, delete-orphan")
    submissions: Mapped[List["Submission"]] = relationship("Submission", back_populates="user", cascade="all, delete-orphan")
    comments: Mapped[List["Comment"]] = relationship("Comment", back_populates="user", cascade="all, delete-orphan")
    bookmarks: Mapped[List["Content"]] = relationship("Content", secondary=user_bookmarks, back_populates="bookmarked_by")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role={self.role})>"


class UserProgress(Base):
    """Track user learning progress and achievements."""
    __tablename__ = "user_progress"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Domain and topic
    domain: Mapped[str] = mapped_column(String(50), nullable=False)  # physics, chemistry, math, cs, dsa, etc.
    topic: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Progress metrics
    problems_solved: Mapped[int] = mapped_column(Integer, default=0)
    total_time_seconds: Mapped[int] = mapped_column(Integer, default=0)
    current_streak: Mapped[int] = mapped_column(Integer, default=0)
    longest_streak: Mapped[int] = mapped_column(Integer, default=0)
    
    # Achievements
    achievements: Mapped[Optional[str]] = mapped_column(Text)  # JSON array
    
    # Timestamps
    last_practiced: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="progress")
    
    __table_args__ = (
        UniqueConstraint("user_id", "domain", "topic", name="unique_user_domain_topic"),
        Index("idx_user_progress_domain", "domain"),
    )


# =============================================================================
# Content Models
# =============================================================================

class Content(Base):
    """User-generated content (notes, tutorials, code snippets)."""
    __tablename__ = "content"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    author_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Content details
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content_type: Mapped[ContentType] = mapped_column(Enum(ContentType), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Categorization
    domain: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    topic: Mapped[str] = mapped_column(String(100), nullable=False)
    difficulty: Mapped[DifficultyLevel] = mapped_column(Enum(DifficultyLevel), nullable=False)
    programming_language: Mapped[Optional[str]] = mapped_column(String(50))
    tags: Mapped[Optional[str]] = mapped_column(Text)  # JSON array
    
    # Moderation and quality
    is_approved: Mapped[bool] = mapped_column(Boolean, default=True)
    is_featured: Mapped[bool] = mapped_column(Boolean, default=False)
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    rating_sum: Mapped[int] = mapped_column(Integer, default=0)
    rating_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    author: Mapped["User"] = relationship("User", back_populates="content")
    comments: Mapped[List["Comment"]] = relationship("Comment", back_populates="content", cascade="all, delete-orphan")
    bookmarked_by: Mapped[List["User"]] = relationship("User", secondary=user_bookmarks, back_populates="bookmarks")
    
    @property
    def average_rating(self) -> float:
        if self.rating_count == 0:
            return 0.0
        return self.rating_sum / self.rating_count
    
    __table_args__ = (
        Index("idx_content_domain_topic", "domain", "topic"),
    )


# =============================================================================
# DSA Problem Models
# =============================================================================

class Problem(Base):
    """DSA problem for practice."""
    __tablename__ = "problems"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    # Problem details
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[ProblemCategory] = mapped_column(Enum(ProblemCategory), nullable=False, index=True)
    difficulty: Mapped[DifficultyLevel] = mapped_column(Enum(DifficultyLevel), nullable=False, index=True)
    
    # Problem specifics
    constraints: Mapped[str] = mapped_column(Text)
    input_format: Mapped[str] = mapped_column(Text)
    output_format: Mapped[str] = mapped_column(Text)
    examples: Mapped[str] = mapped_column(Text)  # JSON array
    
    # Solution info
    approach_explanation: Mapped[str] = mapped_column(Text)
    time_complexity: Mapped[str] = mapped_column(String(100))
    space_complexity: Mapped[str] = mapped_column(String(100))
    
    # Test cases
    test_cases: Mapped[str] = mapped_column(Text)  # JSON array
    
    # Hints
    hints: Mapped[Optional[str]] = mapped_column(Text)  # JSON array
    
    # Statistics
    submission_count: Mapped[int] = mapped_column(Integer, default=0)
    acceptance_rate: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Relationships
    submissions: Mapped[List["Submission"]] = relationship("Submission", back_populates="problem", cascade="all, delete-orphan")
    solutions: Mapped[List["Solution"]] = relationship("Solution", back_populates="problem", cascade="all, delete-orphan")


class Solution(Base):
    """Reference solutions for problems in different languages."""
    __tablename__ = "solutions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    problem_id: Mapped[int] = mapped_column(Integer, ForeignKey("problems.id", ondelete="CASCADE"), nullable=False)
    
    programming_language: Mapped[str] = mapped_column(String(50), nullable=False)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text)
    
    # Relationships
    problem: Mapped["Problem"] = relationship("Problem", back_populates="solutions")
    
    __table_args__ = (
        UniqueConstraint("problem_id", "programming_language", name="unique_problem_language_solution"),
    )


class Submission(Base):
    """User code submissions for problems."""
    __tablename__ = "submissions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    problem_id: Mapped[int] = mapped_column(Integer, ForeignKey("problems.id", ondelete="CASCADE"), nullable=False)
    
    # Submission details
    programming_language: Mapped[str] = mapped_column(String(50), nullable=False)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Results
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False)
    test_cases_passed: Mapped[int] = mapped_column(Integer, default=0)
    total_test_cases: Mapped[int] = mapped_column(Integer, default=0)
    execution_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    memory_used_kb: Mapped[Optional[float]] = mapped_column(Float)
    
    # Error info
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Timestamp
    submitted_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="submissions")
    problem: Mapped["Problem"] = relationship("Problem", back_populates="submissions")
    
    __table_args__ = (
        Index("idx_submission_user_problem", "user_id", "problem_id"),
    )


# =============================================================================
# Interaction Models
# =============================================================================

class Comment(Base):
    """Comments on content."""
    __tablename__ = "comments"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    content_id: Mapped[int] = mapped_column(Integer, ForeignKey("content.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Comment details
    body: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Thread support
    parent_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("comments.id", ondelete="CASCADE"))
    
    # Moderation
    is_approved: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Relationships
    content: Mapped["Content"] = relationship("Content", back_populates="comments")
    user: Mapped["User"] = relationship("User", back_populates="comments")
    replies: Mapped[List["Comment"]] = relationship("Comment", back_populates="parent", remote_side=[id])
    parent: Mapped[Optional["Comment"]] = relationship("Comment", back_populates="replies", remote_side=[parent_id])


# =============================================================================
# Puzzle Models
# =============================================================================

class Puzzle(Base):
    """User-created puzzles and challenges."""
    __tablename__ = "puzzles"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    creator_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    puzzle_type: Mapped[str] = mapped_column(String(50), nullable=False)  # logic, math, programming, mixed
    difficulty: Mapped[DifficultyLevel] = mapped_column(Enum(DifficultyLevel), nullable=False)
    
    # Solution
    solution: Mapped[str] = mapped_column(Text, nullable=False)  # Hidden from users
    
    # Statistics
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    solved_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Relationships
    creator: Mapped["User"] = relationship("User")


# =============================================================================
# Vocabulary Models (English Module)
# =============================================================================

class VocabularyWord(Base):
    """Daily vocabulary words."""
    __tablename__ = "vocabulary_words"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    word: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    pronunciation: Mapped[str] = mapped_column(String(200))  # IPA notation
    definition: Mapped[str] = mapped_column(Text, nullable=False)
    part_of_speech: Mapped[str] = mapped_column(String(50))
    
    # Additional info
    synonyms: Mapped[Optional[str]] = mapped_column(Text)  # JSON array
    antonyms: Mapped[Optional[str]] = mapped_column(Text)  # JSON array
    example_sentences: Mapped[str] = mapped_column(Text)  # JSON array
    
    difficulty: Mapped[DifficultyLevel] = mapped_column(Enum(DifficultyLevel), nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
