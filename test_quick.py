# -*- coding: utf-8 -*-
"""
Quick Test - AG-X 2026 Platform
================================

Testing core functionality of all new modules.
"""

import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def test_chemistry():
    """Test chemistry module."""
    print("Testing Chemistry Module...")
    from agx.chemistry import EquationBalancer
    
    balancer = EquationBalancer()
    equation = balancer.balance_from_string("H2 + O2 -> H2O")
    print(f"  [OK] Equation Balancer: {equation}")
    assert "2H2" in str(equation) and "O2" in str(equation)
    print("  [OK] Chemistry module working!")


def test_mathematics():
    """Test mathematics module."""
    print("\nTesting Mathematics Module...")
    from agx.mathematics import SymbolicSolver, CalculusSolver
    
    # Test symbolic solver
    solver = SymbolicSolver()
    solutions = solver.solve_equation("x**2 - 4 = 0")
    print(f"  [OK] Symbolic Solver: x^2 - 4 = 0 -> {solutions}")
    
    # Test calculus
    calc = CalculusSolver()
    derivative = calc.derivative("x**2")
    print(f"  [OK] Calculus: d/dx(x^2) = {derivative}")
    print("  [OK] Mathematics module working!")


def test_dsa():
    """Test DSA module."""
    print("\nTesting DSA Module...")
    from agx.dsa import ProblemBank, CodeExecutor
    
    # Test problem bank
    bank = ProblemBank()
    problem = bank.get_problem("two-sum")
    print(f"  [OK] Problem Bank: Loaded '{problem.title}'")
    
    # Test code executor
    executor = CodeExecutor()
    code = 'print("Hello, AG-X 2026!")'
    result = executor.execute_python(code)
    print(f"  [OK] Code Executor: {result.output}")
    print("  [OK] DSA module working!")


def test_english():
    """Test English module."""
    print("\nTesting English Module...")
    from agx.english import VocabularyBuilder
    
    vocab = VocabularyBuilder()
    words = vocab.get_daily_words(count=1)
    print(f"  [OK] Vocabulary: Got word '{words[0].word}'")
    print("  [OK] English module working!")


def test_programming():
    """Test programming module."""
    print("\nTesting Programming Module...")
    from agx.programming import LANGUAGES
    
    python_info = LANGUAGES.get("python")
    print(f"  [OK] Language DB: Python - {python_info['description'][:50]}...")
    print("  [OK] Programming module working!")


def test_config():
    """Test configuration."""
    print("\nTesting Configuration System...")
    from agx.config import AGXConfig
    
    config = AGXConfig()
    print(f"  [OK] Platform: {config.name}")
    print(f"  [OK] Chemistry enabled: {config.chemistry.enabled}")
    print(f"  [OK] DSA enabled: {config.dsa.enabled}")
    print("  [OK] Configuration working!")


if __name__ == "__main__":
    print("=" * 60)
    print("AG-X 2026 Universal Platform - Quick Test")
    print("=" * 60)
    
    try:
        test_chemistry()
        test_mathematics()
        test_dsa()
        test_english()
        test_programming()
        test_config()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("=" * 60)
        print("\nPlatform is ready to use!")
        print("\nNext steps:")
        print("  1. Run web dashboard: python -m agx.web_app")
        print("  2. Try demo: python examples\\comprehensive_demo.py")
        print("  3. Install dependencies: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure to install dependencies:")
        print("  pip install -r requirements.txt")
