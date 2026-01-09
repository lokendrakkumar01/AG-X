"""
Natural Language Experiment Control
=====================================

Parse natural language commands to control simulations.
"""

from __future__ import annotations
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Command:
    """Parsed command from natural language."""
    action: str
    parameters: Dict[str, Any]
    confidence: float
    original_text: str


class NLPController:
    """
    Natural Language Experiment Controller.
    
    Parses natural language commands to configure and run simulations.
    """
    
    # Command patterns
    PATTERNS = {
        "run_simulation": [
            r"run\s+(?:a\s+)?simulation",
            r"start\s+(?:a\s+)?simulation",
            r"simulate",
            r"execute\s+(?:the\s+)?simulation",
        ],
        "set_parameter": [
            r"set\s+(\w+)\s+to\s+([0-9.e+-]+)",
            r"change\s+(\w+)\s+to\s+([0-9.e+-]+)",
            r"(\w+)\s*=\s*([0-9.e+-]+)",
        ],
        "create_body": [
            r"(?:add|create)\s+(?:a\s+)?(?:body|particle|mass)",
        ],
        "show_visualization": [
            r"show\s+(?:the\s+)?(\w+)",
            r"visualize\s+(\w+)",
            r"display\s+(\w+)",
        ],
        "generate_report": [
            r"generate\s+(?:a\s+)?report",
            r"create\s+(?:a\s+)?report",
            r"export\s+results",
        ],
        "get_help": [
            r"help",
            r"what\s+can\s+(?:you|i)\s+do",
            r"commands",
        ],
        "stop": [
            r"stop",
            r"halt",
            r"pause",
        ],
    }
    
    # Parameter extraction patterns
    PARAM_PATTERNS = {
        "timesteps": r"(\d+)\s*(?:time)?steps?",
        "dt": r"dt\s*(?:=|of)?\s*([0-9.e+-]+)",
        "mass": r"mass\s*(?:=|of)?\s*([0-9.e+-]+)",
        "position": r"position\s*(?:=|at)?\s*\[?\s*([0-9.e+-]+)\s*,\s*([0-9.e+-]+)\s*(?:,\s*([0-9.e+-]+))?\s*\]?",
        "velocity": r"velocity\s*(?:=|of)?\s*\[?\s*([0-9.e+-]+)\s*,\s*([0-9.e+-]+)\s*(?:,\s*([0-9.e+-]+))?\s*\]?",
        "scenario": r"scenario\s*(?:=|:)?\s*(\w+)",
    }
    
    def __init__(self):
        self.command_history: List[Command] = []
    
    def parse(self, text: str) -> Command:
        """Parse natural language text into a command."""
        text = text.lower().strip()
        
        # Try to match action patterns
        action = "unknown"
        confidence = 0.0
        
        for cmd, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    action = cmd
                    confidence = 0.9
                    break
            if action != "unknown":
                break
        
        # Extract parameters
        parameters = self._extract_parameters(text)
        
        # Special handling for set_parameter
        if action == "unknown":
            for pattern in self.PATTERNS["set_parameter"]:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    action = "set_parameter"
                    parameters["name"] = match.group(1)
                    parameters["value"] = float(match.group(2))
                    confidence = 0.85
                    break
        
        command = Command(
            action=action,
            parameters=parameters,
            confidence=confidence,
            original_text=text,
        )
        
        self.command_history.append(command)
        return command
    
    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract parameters from text."""
        params = {}
        
        for param_name, pattern in self.PARAM_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if param_name in ["position", "velocity"]:
                    values = [float(g) for g in groups if g is not None]
                    if len(values) == 2:
                        values.append(0.0)
                    params[param_name] = values
                elif param_name == "scenario":
                    params[param_name] = groups[0]
                else:
                    params[param_name] = float(groups[0])
        
        return params
    
    def execute(self, command: Command, engine: Any = None) -> Dict[str, Any]:
        """Execute a parsed command."""
        result = {"success": False, "message": "", "data": None}
        
        if command.action == "run_simulation":
            result["message"] = "Starting simulation..."
            result["success"] = True
            if engine:
                timesteps = command.parameters.get("timesteps", 1000)
                result["data"] = engine.run_simulation(timesteps=int(timesteps))
                result["message"] = f"Simulation completed with {timesteps} timesteps"
        
        elif command.action == "set_parameter":
            name = command.parameters.get("name")
            value = command.parameters.get("value")
            result["message"] = f"Set {name} = {value}"
            result["success"] = True
            result["data"] = {name: value}
        
        elif command.action == "create_body":
            mass = command.parameters.get("mass", 1.0)
            position = command.parameters.get("position", [0, 0, 0])
            result["message"] = f"Created body with mass={mass} at position={position}"
            result["success"] = True
        
        elif command.action == "show_visualization":
            result["message"] = "Opening visualization..."
            result["success"] = True
        
        elif command.action == "generate_report":
            result["message"] = "Generating report..."
            result["success"] = True
        
        elif command.action == "get_help":
            result["message"] = self._get_help_text()
            result["success"] = True
        
        elif command.action == "stop":
            result["message"] = "Stopping simulation..."
            result["success"] = True
        
        else:
            result["message"] = f"Unknown command: {command.original_text}"
        
        return result
    
    def _get_help_text(self) -> str:
        """Generate help text."""
        return """
Available Commands:
- "Run simulation" or "Start simulation" - Begin a new simulation
- "Set [parameter] to [value]" - Modify simulation parameters
- "Add a body with mass [value]" - Create a new gravitating body
- "Show particles/spacetime/energy" - Display visualizations
- "Generate report" - Create experiment report
- "Stop" - Halt current simulation
- "Help" - Show this message

Example: "Run simulation with 5000 timesteps and dt=0.001"
"""
    
    def suggest_completion(self, partial_text: str) -> List[str]:
        """Suggest command completions."""
        suggestions = []
        partial = partial_text.lower()
        
        completions = [
            "run simulation",
            "run simulation with 1000 timesteps",
            "set mass to 1.0",
            "set dt to 0.01",
            "add a body with mass 1.0 at position [0, 0, 0]",
            "show particles",
            "show spacetime",
            "generate report",
            "help",
        ]
        
        for completion in completions:
            if completion.startswith(partial):
                suggestions.append(completion)
        
        return suggestions[:5]
