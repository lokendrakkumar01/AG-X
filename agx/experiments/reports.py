"""
Report Generator
=================

Auto-generate research-style reports in PDF and Markdown formats.
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

from jinja2 import Template
import markdown


@dataclass
class Report:
    """Generated report container."""
    title: str
    content: str
    format: str  # "markdown" or "pdf"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


MARKDOWN_TEMPLATE = """# {{ title }}

**Generated:** {{ date }}
**Experiment ID:** {{ experiment_id }}

---

## Executive Summary

{{ summary }}

---

## Configuration

```yaml
{{ config_yaml }}
```

---

## Results

### Key Metrics

| Metric | Value |
|--------|-------|
{% for metric, value in metrics.items() %}| {{ metric }} | {{ "%.4f"|format(value) if value is number else value }} |
{% endfor %}

### Simulation Statistics

{{ statistics }}

---

## Visualizations

{% for viz_name, viz_path in visualizations.items() %}
### {{ viz_name }}
![{{ viz_name }}]({{ viz_path }})

{% endfor %}

---

## Analysis

{{ analysis }}

---

## AI-Generated Insights

{{ ai_insights }}

---

## Theoretical Disclaimer

⚠️ **IMPORTANT:** All results presented in this report are from THEORETICAL SIMULATIONS only. 
No claims of real-world anti-gravity effects or violation of known physical laws are made.
These simulations are for EDUCATIONAL and RESEARCH EXPLORATION purposes only.

---

## Appendix

### Reproducibility Information

- **Random Seed:** {{ seed }}
- **Software Version:** AG-X 2026 v1.0.0
- **Config Hash:** {{ config_hash }}

"""


class ReportGenerator:
    """
    Generate research-style reports from experiment results.
    
    Supports Markdown and PDF output with equations, graphs, and AI explanations.
    """
    
    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown(self,
                         experiment: Any,
                         results: Dict[str, Any],
                         visualizations: Dict[str, str] = None,
                         ai_insights: str = "") -> Report:
        """Generate Markdown report."""
        template = Template(MARKDOWN_TEMPLATE)
        
        # Prepare config as YAML-like string
        config_yaml = self._dict_to_yaml(experiment.config) if hasattr(experiment, 'config') else ""
        
        # Generate statistics summary
        stats_lines = []
        if "chaos_analysis" in results:
            chaos = results["chaos_analysis"]
            stats_lines.append(f"- **Chaos Detected:** {chaos.get('is_chaotic', False)}")
            stats_lines.append(f"- **Lyapunov Exponent:** {chaos.get('lyapunov_exponent', 'N/A')}")
        
        content = template.render(
            title=getattr(experiment, 'name', 'AG-X Simulation Report'),
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            experiment_id=getattr(experiment, 'id', 'N/A'),
            summary=getattr(experiment, 'description', 'Gravity simulation experiment.'),
            config_yaml=config_yaml,
            metrics=getattr(experiment, 'metrics', {}),
            statistics="\n".join(stats_lines) or "No statistics available.",
            visualizations=visualizations or {},
            analysis=self._generate_analysis(results),
            ai_insights=ai_insights or "*No AI insights generated.*",
            seed=getattr(experiment, 'random_seed', 42),
            config_hash=experiment.get_hash() if hasattr(experiment, 'get_hash') else 'N/A',
        )
        
        return Report(
            title=getattr(experiment, 'name', 'Report'),
            content=content,
            format="markdown",
            metadata={"experiment_id": getattr(experiment, 'id', 'N/A')},
        )
    
    def _dict_to_yaml(self, d: Dict, indent: int = 0) -> str:
        """Convert dict to YAML-like string."""
        lines = []
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._dict_to_yaml(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)
    
    def _generate_analysis(self, results: Dict[str, Any]) -> str:
        """Generate analysis text from results."""
        lines = []
        
        if "energy_history" in results:
            energies = results["energy_history"]
            if len(energies) > 0:
                initial = energies[0]
                final = energies[-1]
                drift = abs(final - initial) / abs(initial) if initial != 0 else 0
                lines.append(f"**Energy Conservation:** {(1-drift)*100:.4f}% maintained")
        
        if "chaos_analysis" in results:
            chaos = results["chaos_analysis"]
            if chaos.get("is_chaotic"):
                lines.append("**Chaos Warning:** The system exhibits chaotic behavior.")
            else:
                lines.append("**Stability:** The system appears stable and predictable.")
        
        return "\n\n".join(lines) or "Analysis not available."
    
    def save_report(self, report: Report, filename: str = None) -> Path:
        """Save report to file."""
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if report.format == "markdown":
            filepath = self.output_dir / f"{filename}.md"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report.content)
        else:
            filepath = self.output_dir / f"{filename}.pdf"
            self._save_pdf(report, filepath)
        
        return filepath
    
    def _save_pdf(self, report: Report, filepath: Path) -> None:
        """Save report as PDF using reportlab."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            
            doc = SimpleDocTemplate(str(filepath), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Convert markdown to simple paragraphs
            for line in report.content.split("\n"):
                if line.startswith("# "):
                    story.append(Paragraph(line[2:], styles["Title"]))
                elif line.startswith("## "):
                    story.append(Paragraph(line[3:], styles["Heading2"]))
                elif line.strip():
                    story.append(Paragraph(line, styles["Normal"]))
                story.append(Spacer(1, 6))
            
            doc.build(story)
            
        except ImportError:
            # Fallback: save as markdown
            md_path = filepath.with_suffix(".md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(report.content)
    
    def generate_quick_summary(self, results: Dict[str, Any]) -> str:
        """Generate a quick one-page summary."""
        lines = [
            "# Quick Summary",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]
        
        if "final_state" in results:
            lines.append("## Final State")
            lines.append(f"- Simulation completed successfully")
        
        if results.get("metrics"):
            lines.append("\n## Key Metrics")
            for k, v in results["metrics"].items():
                lines.append(f"- **{k}:** {v:.4f}" if isinstance(v, float) else f"- **{k}:** {v}")
        
        lines.append("\n---")
        lines.append("*All results are theoretical simulations only.*")
        
        return "\n".join(lines)
