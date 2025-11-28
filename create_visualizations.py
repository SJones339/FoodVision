#!/usr/bin/env python3
"""
Quick script to create comparison visualizations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from visualization.create_comparison_plots import create_comprehensive_comparison

if __name__ == "__main__":
    print("Creating comparison visualizations...")
    create_comprehensive_comparison()
    print("\n✓ Done! Check models/model_comparison.png")

