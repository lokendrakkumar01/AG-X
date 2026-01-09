"""
AG-X 2026 Web Application
==========================

Web dashboard launcher and Flask/Dash integration.
"""

import sys
from loguru import logger

from agx import DISCLAIMER
from agx.config import setup_logging, settings


def main():
    """Launch the AG-X web dashboard."""
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("AG-X 2026 - Advanced Gravity Research Simulation Platform")
    logger.info("=" * 60)
    logger.warning("All simulations are THEORETICAL and for EDUCATIONAL purposes only.")
    logger.info("=" * 60)
    
    from agx.viz.dashboard import Dashboard, DashboardConfig
    
    config = DashboardConfig(
        title="AG-X 2026 - Gravity Research Platform",
        debug=settings.debug,
    )
    
    dashboard = Dashboard(config)
    
    logger.info(f"Starting dashboard on http://{settings.web_host}:{settings.web_port}")
    logger.info("Press Ctrl+C to stop")
    
    try:
        dashboard.run(
            host=settings.web_host,
            port=settings.web_port,
            debug=settings.debug,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
