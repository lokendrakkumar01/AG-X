"""
DSA Problem Bank
================

Problem database with categorization and difficulty levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import json


class DSACategory(str, Enum):
    """DSA problem categories."""
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


@dataclass
class TestCase:
    """Test case for a problem."""
    input: str
    expected_output: str
    is_hidden: bool = False  # Hidden test cases not shown to user


@dataclass
class Problem:
    """DSA problem."""
    id: str
    title: str
    description: str
    category: DSACategory
    difficulty: str  # "beginner", "intermediate", "advanced"
    
    constraints: List[str] = field(default_factory=list)
    input_format: str = ""
    output_format: str = ""
    examples: List[Dict[str, str]] = field(default_factory=list)
    
    test_cases: List[TestCase] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    
    time_complexity: str = ""
    space_complexity: str = ""
    approach: str = ""
    
    tags: List[str] = field(default_factory=list)


class ProblemBank:
    """Manage DSA problems."""
    
    def __init__(self):
        """Initialize problem bank with sample problems."""
        self.problems: Dict[str, Problem] = {}
        self._initialize_sample_problems()
    
    def _initialize_sample_problems(self):
        """Add sample problems for demonstration."""
        
        # Two Sum problem
        two_sum = Problem(
            id="two-sum",
            title="Two Sum",
            description=(
                "Given an array of integers nums and an integer target, "
                "return indices of the two numbers such that they add up to target.\n\n"
                "You may assume that each input would have exactly one solution, "
                "and you may not use the same element twice."
            ),
            category=DSACategory.ARRAYS,
            difficulty="beginner",
            constraints=[
                "2 <= nums.length <= 10^4",
                "-10^9 <= nums[i] <= 10^9",
                "-10^9 <= target <= 10^9",
                "Only one valid answer exists."
            ],
            input_format="nums: List[int], target: int",
            output_format="List[int]",
            examples=[
                {
                    "input": "nums = [2,7,11,15], target = 9",
                    "output": "[0,1]",
                    "explanation": "Because nums[0] + nums[1] == 9, we return [0, 1]."
                },
                {
                    "input": "nums = [3,2,4], target = 6",
                    "output": "[1,2]",
                    "explanation": "Because nums[1] + nums[2] == 6, we return [1, 2]."
                }
            ],
            test_cases=[
                TestCase("[2,7,11,15]\n9", "[0, 1]"),
                TestCase("[3,2,4]\n6", "[1, 2]"),
                TestCase("[3,3]\n6", "[0, 1]"),
            ],
            hints=[
                "Use a hash map to store numbers you've seen.",
                "For each number, check if target - number exists in the hash map.",
                "Time complexity can be O(n) with this approach."
            ],
            time_complexity="O(n)",
            space_complexity="O(n)",
            approach=(
                "1. Create a hash map to store {value: index}\n"
                "2. Iterate through the array\n"
                "3. For each number, calculate complement = target - number\n"
                "4. Check if complement exists in hash map\n"
                "5. If found, return [map[complement], current_index]\n"
                "6. Otherwise, add current number to hash map"
            ),
            tags=["hash-table", "array"]
        )
        
        # Reverse String problem
        reverse_string = Problem(
            id="reverse-string",
            title="Reverse String",
            description=(
                "Write a function that reverses a string. "
                "The input string is given as an array of characters s.\n\n"
                "You must do this by modifying the input array in-place with O(1) extra memory."
            ),
            category=DSACategory.STRINGS,
            difficulty="beginner",
            constraints=[
                "1 <= s.length <= 10^5",
                "s[i] is a printable ascii character."
            ],
            input_format="s: List[str]",
            output_format="None (modify in-place)",
            examples=[
                {
                    "input": 's = ["h","e","l","l","o"]',
                    "output": '["o","l","l","e","h"]'
                },
                {
                    "input": 's = ["H","a","n","n","a","h"]',
                    "output": '["h","a","n","n","a","H"]'
                }
            ],
            test_cases=[
                TestCase('["h","e","l","l","o"]', '["o","l","l","e","h"]'),
                TestCase('["H","a","n","n","a","h"]', '["h","a","n","n","a","H"]'),
            ],
            hints=[
                "Use two pointers approach from both ends.",
                "Swap characters at left and right pointers.",
                "Move pointers towards center."
            ],
            time_complexity="O(n)",
            space_complexity="O(1)",
            approach=(
                "1. Use two pointers: left = 0, right = len(s) - 1\n"
                "2. While left < right:\n"
                "   - Swap s[left] and s[right]\n"
                "   - Increment left, decrement right"
            ),
            tags=["two-pointers", "string"]
        )
        
        # Binary Search
        binary_search = Problem(
            id="binary-search",
            title="Binary Search",
            description=(
                "Given an array of integers nums which is sorted in ascending order, "
                "and an integer target, write a function to search target in nums. "
                "If target exists, then return its index. Otherwise, return -1."
            ),
            category=DSACategory.SEARCHING,
            difficulty="beginner",
            constraints=[
                "1 <= nums.length <= 10^4",
                "-10^4 < nums[i], target < 10^4",
                "All the integers in nums are unique.",
                "nums is sorted in ascending order."
            ],
            input_format="nums: List[int], target: int",
            output_format="int",
            examples=[
                {
                    "input": "nums = [-1,0,3,5,9,12], target = 9",
                    "output": "4",
                    "explanation": "9 exists in nums and its index is 4"
                },
                {
                    "input": "nums = [-1,0,3,5,9,12], target = 2",
                    "output": "-1",
                    "explanation": "2 does not exist in nums so return -1"
                }
            ],
            test_cases=[
                TestCase("[-1,0,3,5,9,12]\n9", "4"),
                TestCase("[-1,0,3,5,9,12]\n2", "-1"),
                TestCase("[5]\n5", "0"),
            ],
            hints=[
                "Use divide and conquer approach.",
                "Compare middle element with target.",
                "Recursion or iteration both work."
            ],
            time_complexity="O(log n)",
            space_complexity="O(1) iterative, O(log n) recursive",
            approach=(
                "1. Set left = 0, right = len(nums) - 1\n"
                "2. While left <= right:\n"
                "   - Calculate mid = (left + right) // 2\n"
                "   - If nums[mid] == target, return mid\n"
                "   - If nums[mid] < target, set left = mid + 1\n"
                "   - Else set right = mid - 1\n"
                "3. Return -1 if not found"
            ),
            tags=["binary-search", "algorithm"]
        )
        
        self.problems["two-sum"] = two_sum
        self.problems["reverse-string"] = reverse_string
        self.problems["binary-search"] = binary_search
    
    def get_problem(self, problem_id: str) -> Optional[Problem]:
        """Get a problem by ID."""
        return self.problems.get(problem_id)
    
    def list_problems(
        self,
        category: Optional[DSACategory] = None,
        difficulty: Optional[str] = None
    ) -> List[Problem]:
        """List problems with optional filtering."""
        problems = list(self.problems.values())
        
        if category:
            problems = [p for p in problems if p.category == category]
        
        if difficulty:
            problems = [p for p in problems if p.difficulty == difficulty]
        
        return problems
    
    def add_problem(self, problem: Problem):
        """Add a new problem to the bank."""
        self.problems[problem.id] = problem
    
    def get_categories(self) -> List[DSACategory]:
        """Get all available categories."""
        return list(DSACategory)
    
    def get_difficulties(self) -> List[str]:
        """Get all difficulty levels."""
        return ["beginner", "intermediate", "advanced"]
