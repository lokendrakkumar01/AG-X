"""
AG-X 2026 Universal Platform - Comprehensive Demo
==================================================

This script demonstrates all major features of the AG-X 2026 platform.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_header(title: str):
    """Print a section header."""
    console.print(f"\n[bold cyan]=== {title} ===[/bold cyan]\n")


def demo_chemistry():
    """Demonstrate chemistry module."""
    print_header("Chemistry Module Demo")
    
    from agx.chemistry import Equation Balancer, MolecularStructure, ThermodynamicsCalculator
    
    # 1. Balance equation
    console.print("[yellow]1. Balancing Chemical Equation[/yellow]")
    balancer = EquationBalancer()
    equation = balancer.balance_from_string("H2 + O2 -> H2O")
    console.print(f"   Balanced: [green]{equation}[/green]")
    
    # 2. Create molecule
    console.print("\n[yellow]2. Molecular Structure[/yellow]")
    water = MolecularStructure.create_water()
    console.print(f"   Formula: [green]{water.get_molecular_formula()}[/green]")
    console.print(f"   Molecular Weight: [green]{water.get_molecular_weight():.2f} g/mol[/green]")
    
    # 3. Thermodynamics
    console.print("\n[yellow]3. Thermodynamics[/yellow]")
    thermo = ThermodynamicsCalculator()
    products = {"H2O(l)": 2}
    reactants = {"H2(g)": 2, "O2(g)": 1}
    delta_h = thermo.calculate_delta_h(products, reactants)
    console.print(f"   ΔH° = [green]{delta_h:.1f} kJ/mol[/green]")


def demo_mathematics():
    """Demonstrate mathematics module."""
    print_header("Mathematics Module Demo")
    
    from agx.mathematics import SymbolicSolver, CalculusSolver
    
    # 1. Solve equation
    console.print("[yellow]1. Solving Equations[/yellow]")
    solver = SymbolicSolver()
    solutions = solver.solve_equation("x**2 - 5*x + 6 = 0")
    console.print(f"   Solutions to x² - 5x + 6 = 0: [green]{', '.join(solutions)}[/green]")
    
    # 2. Calculus
    console.print("\n[yellow]2. Calculus Operations[/yellow]")
    calc = CalculusSolver()
    derivative = calc.derivative("x**3 + 2*x**2 - 5*x + 1")
    console.print(f"   d/dx(x³ + 2x² - 5x + 1) = [green]{derivative}[/green]")
    
    integral = calc.integral("2*x + 3")
    console.print(f"   ∫(2x + 3)dx = [green]{integral}[/green]")


def demo_dsa():
    """Demonstrate DSA practice system."""
    print_header("DSA Practice System Demo")
    
    from agx.dsa import ProblemBank, CodeExecutor
    
    # 1. Browse problems
    console.print("[yellow]1. Available Problems[/yellow]")
    bank = ProblemBank()
    problems = bank.list_problems()
    
    table = Table(title="DSA Problems")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="magenta")
    table.add_column("Category", style="green")
    table.add_column("Difficulty", style="yellow")
    
    for problem in problems:
        table.add_row(problem.id, problem.title, problem.category.value, problem.difficulty)
    
    console.print(table)
    
    # 2. Solve a problem
    console.print("\n[yellow]2. Solving 'Two Sum' Problem[/yellow]")
    problem = bank.get_problem("two-sum")
    console.print(f"   {problem.description[:100]}...")
    
    # Submit solution
    code = '''
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

print(two_sum([2,7,11,15], 9))
'''
    
    executor = CodeExecutor()
    result = executor.execute_python(code)
    console.print(f"   Output: [green]{result.output}[/green]")
    console.print(f"   Time complexity: [cyan]{problem.time_complexity}[/cyan]")


def demo_english():
    """Demonstrate English module."""
    print_header("English Communication Demo")
    
    from agx.english import VocabularyBuilder, ConversationPractice
    
    # 1. Daily vocabulary
    console.print("[yellow]1. Daily Vocabulary Words[/yellow]")
    vocab = VocabularyBuilder()
    words = vocab.get_daily_words(count=2)
    
    for word in words:
        console.print(f"\n   Word: [bold green]{word.word}[/bold green]")
        console.print(f"   Pronunciation: {word.pronunciation}")
        console.print(f"   Definition: {word.definition}")
    
    # 2. Conversation practice
    console.print("\n[yellow]2. Conversation Practice[/yellow]")
    conversation = ConversationPractice()
    scenario = conversation.get_scenario("beginner")
    console.print(f"   Scenario: [bold]{scenario.title}[/bold]")
    console.print(f"   Context: {scenario.context}")


def demo_programming():
    """Demonstrate programming languages module."""
    print_header("Programming Languages Demo")
    
    from agx.programming import LANGUAGES
    
    console.print("[yellow]Supported Languages[/yellow]\n")
    
    for lang_id, info in list(LANGUAGES.items())[:3]:  # Show first 3
        console.print(f"[bold green]{info['name']}[/bold green]")
        console.print(f"  Description: {info['description']}")
        console.print(f"  Use Cases: {', '.join(info['use_cases'][:3])}")
        console.print(f"  Difficulty: {info['difficulty']}\n")


def main():
    """Run all demos."""
    console.print(Panel.fit(
        "[bold cyan]AG-X 2026 Universal Educational Platform[/bold cyan]\n"
        "[yellow]Comprehensive Feature Demonstration[/yellow]",
        border_style="blue"
    ))
    
    try:
        demo_chemistry()
        demo_mathematics()
        demo_dsa()
        demo_english()
        demo_programming()
        
        console.print("\n[bold green]✓ All demos completed successfully![/bold green]")
        console.print("\n[dim]For more information, visit the web dashboard:[/dim]")
        console.print("[bold]python -m agx.web_app[/bold]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        console.print("[yellow]Make sure all dependencies are installed: pip install -r requirements.txt[/yellow]")


if __name__ == "__main__":
    main()
