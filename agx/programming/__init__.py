"""
AG-X 2026 Programming Languages Module
=======================================

Programming language knowledge base and tutorials.
"""

LANGUAGES = {
    "python": {
        "name": "Python",
        "description": "High-level, interpreted language known for readability and versatility",
        "use_cases": ["Web Development", "Data Science", "Machine Learning", "Automation", "Scripting"],
        "paradigm": "Multi-paradigm (Object-Oriented, Functional, Procedural)",
        "difficulty": "beginner",
        "hello_world": 'print("Hello, World!")',
        "key_features": [
            "Dynamic typing",
            "Extensive standard library",
            "Large ecosystem of packages (PyPI)",
            "Readable syntax",
            "Cross-platform"
        ]
    },
    "java": {
        "name": "Java",
        "description": "Class-based, object-oriented language with write once, run anywhere philosophy",
        "use_cases": ["Enterprise Applications", "Android Development", "Web Services", "Big Data"],
        "paradigm": "Object-Oriented",
        "difficulty": "intermediate",
        "hello_world": '''public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}''',
        "key_features": [
            "Platform independent (JVM)",
            "Strong type system",
            "Garbage collection",
            "Rich standard library",
            "Multithreading support"
        ]
    },
    "javascript": {
        "name": "JavaScript",
        "description": "Dynamic scripting language for web development, both client and server-side",
        "use_cases": ["Web Development", "Server-side (Node.js)", "Mobile Apps", "Desktop Apps"],
        "paradigm": "Multi-paradigm (Event-driven, Functional, Prototype-based OOP)",
        "difficulty": "beginner",
        "hello_world": 'console.log("Hello, World!");',
        "key_features": [
            "Event-driven programming",
            "First-class functions",
            "Prototype-based inheritance",
            "Asynchronous programming",
            "Runs in browser and server (Node.js)"
        ]
    },
    "cpp": {
        "name": "C++",
        "description": "Powerful, high-performance language extending C with object-oriented features",
        "use_cases": ["System Software", "Game Development", "High-Performance Computing", "Embedded Systems"],
        "paradigm": "Multi-paradigm (Object-Oriented, Procedural, Generic)",
        "difficulty": "advanced",
        "hello_world": '''#include <iostream>
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}''',
        "key_features": [
            "High performance",
            "Manual memory management",
            "Templates and generic programming",
            "Multiple programming paradigms",
            "Direct hardware access"
        ]
    },
}

__all__ = ["LANGUAGES"]
