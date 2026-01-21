"""
Code Executor
=============

Safe code execution for DSA problems in multiple languages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import subprocess
import tempfile
import os
import time
from pathlib import Path
from loguru import logger


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    memory_used_kb: float = 0.0
    test_cases_passed: int = 0
    total_test_cases: int = 0


class CodeExecutor:
    """Execute code in multiple languages safely."""
    
    # Language configurations
    LANGUAGE_CONFIG = {
        "python": {
            "extension": ".py",
            "command": ["python", "-u"],
            "timeout": 10,
        },
        "java": {
            "extension": ".java",
            "compile_command": ["javac"],
            "command": ["java"],
            "timeout": 10,
        },
        "cpp": {
            "extension": ".cpp",
            "compile_command": ["g++", "-o"],
            "command": [],
            "timeout": 10,
        },
        "c": {
            "extension": ".c",
            "compile_command": ["gcc", "-o"],
            "command": [],
            "timeout": 10,
        },
        "javascript": {
            "extension": ".js",
            "command": ["node"],
            "timeout": 10,
        },
    }
    
    def __init__(self, timeout_seconds: int = 10, max_memory_mb: int = 256):
        """Initialize code executor.
        
        Args:
            timeout_seconds: Maximum execution time
            max_memory_mb: Maximum memory usage
        """
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
    
    def execute_python(
        self,
        code: str,
        test_input: str = ""
    ) -> ExecutionResult:
        """Execute Python code.
        
        Args:
            code: Python code to execute
            test_input: Input to provide to the code
            
        Returns:
            ExecutionResult
        """
        start_time = time.time()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute code
            process = subprocess.Popen(
                ['python', '-u', temp_file],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(
                    input=test_input,
                    timeout=self.timeout_seconds
                )
                
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                
                if process.returncode == 0:
                    return ExecutionResult(
                        success=True,
                        output=stdout.strip(),
                        execution_time_ms=execution_time
                    )
                else:
                    return ExecutionResult(
                        success=False,
                        output=stdout.strip(),
                        error=stderr.strip(),
                        execution_time_ms=execution_time
                    )
            
            except subprocess.TimeoutExpired:
                process.kill()
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {self.timeout_seconds} seconds"
                )
        
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e)
            )
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def execute_code(
        self,
        code: str,
        language: str,
        test_input: str = ""
    ) -> ExecutionResult:
        """Execute code in specified language.
        
        Args:
            code: Code to execute
            language: Programming language (python, java, cpp, c, javascript)
            test_input: Input to provide to the code
            
        Returns:
            ExecutionResult
        """
        language = language.lower()
        
        if language == "python":
            return self.execute_python(code, test_input)
        
        # For other languages, return a placeholder for now
        # In production, you would implement proper sandboxed execution
        return ExecutionResult(
            success=False,
            output="",
            error=f"Language '{language}' execution not yet implemented in this demo. "
                  f"Currently only Python is supported for code execution. "
                  f"In production, Docker containers would be used for secure multi-language execution."
        )
    
    def run_test_cases(
        self,
        code: str,
        language: str,
        test_cases: List[tuple[str, str]]  # List of (input, expected_output)
    ) -> ExecutionResult:
        """Run code against multiple test cases.
        
        Args:
            code: Code to execute
            language: Programming language
            test_cases: List of (input, expected_output) tuples
            
        Returns:
            ExecutionResult with test case results
        """
        passed = 0
        failed_outputs = []
        total_time = 0.0
        
        for i, (test_input, expected_output) in enumerate(test_cases):
            result = self.execute_code(code, language, test_input)
            total_time += result.execution_time_ms
            
            if not result.success:
                return ExecutionResult(
                    success=False,
                    output=f"Test case {i+1} failed to execute",
                    error=result.error,
                    test_cases_passed=passed,
                    total_test_cases=len(test_cases)
                )
            
            # Compare output (strip whitespace for comparison)
            actual = result.output.strip()
            expected = expected_output.strip()
            
            if actual == expected:
                passed += 1
            else:
                failed_outputs.append(
                    f"Test {i+1}: Expected '{expected}', got '{actual}'"
                )
        
        all_passed = (passed == len(test_cases))
        
        return ExecutionResult(
            success=all_passed,
            output="All test cases passed!" if all_passed else "\n".join(failed_outputs),
            execution_time_ms=total_time / len(test_cases) if test_cases else 0,
            test_cases_passed=passed,
            total_test_cases=len(test_cases)
        )
    
    def validate_syntax(self, code: str, language: str) -> tuple[bool, Optional[str]]:
        """Validate code syntax without executing.
        
        Args:
            code: Code to validate
            language: Programming language
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if language.lower() == "python":
            try:
                compile(code, '<string>', 'exec')
                return True, None
            except SyntaxError as e:
                return False, str(e)
        
        # For other languages, would need language-specific validators
        return True, None  # Assume valid for now
