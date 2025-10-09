#!/usr/bin/env python3
"""
Example script showing how to plot calculations separately by their source
(AFM workchain vs constrained scan) using the tagged data from gather_workchain_data.py
"""
#%%
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def load_workchain_data(filename: str) -> Dict[str, Any]:
    """Load workchain data from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def extract_energies_by_source(data: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Extract energies grouped by calculation source.
    
    Returns:
        dict: Dictionary with source types as keys and lists of energies as values
    """
    energies_by_source = {
        'afm_workchain': [],
        'constrained_scan': [],
        'unknown': []
    }
    
    for calc_data in data['calculations'].values():
        source = calc_data.get('calculation_source', 'unknown')
        
        # Extract energy from output_parameters
        if calc_data.get('output_parameters') and isinstance(calc_data['output_parameters'], dict):
            energy = calc_data['output_parameters'].get('energy')
            if energy is not None:
                energies_by_source[source].append(energy)
    
    return energies_by_source


def plot_energy_distribution(energies_by_source: Dict[str, List[float]], 
                           save_path: str = None, show: bool = True):
    """
    Create separate plots for AFM workchain and constrained scan calculations.
    
    Args:
        energies_by_source: Dictionary with energies grouped by source
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot AFM workchain results
    afm_energies = energies_by_source['afm_workchain']
    if afm_energies:
        ax1.hist(afm_energies, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.min(afm_energies), color='red', linestyle='--', alpha=0.7, label=f'Min: {np.min(afm_energies):.3f}')
        ax1.set_xlabel('Energy (Ry)')
        ax1.set_ylabel('Count')
        ax1.set_title(f'AFM Workchain Calculations\n({len(afm_energies)} calculations)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No AFM calculations found', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('AFM Workchain Calculations\n(0 calculations)')
    
    # Plot constrained scan results
    constrained_energies = energies_by_source['constrained_scan']
    if constrained_energies:
        ax2.hist(constrained_energies, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.min(constrained_energies), color='red', linestyle='--', alpha=0.7, label=f'Min: {np.min(constrained_energies):.3f}')
        ax2.set_xlabel('Energy (Ry)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Constrained Scan Calculations\n({len(constrained_energies)} calculations)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No constrained calculations found', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Constrained Scan Calculations\n(0 calculations)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()


def plot_energy_comparison(energies_by_source: Dict[str, List[float]], 
                          save_path: str = None, show: bool = True):
    """
    Create a comparison plot showing both calculation types together.
    
    Args:
        energies_by_source: Dictionary with energies grouped by source
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    afm_energies = energies_by_source['afm_workchain']
    constrained_energies = energies_by_source['constrained_scan']
    
    # Create scatter plot with different colors and markers
    if afm_energies:
        ax.scatter(range(len(afm_energies)), sorted(afm_energies), 
                  c='blue', marker='o', alpha=0.7, s=50, label=f'AFM Workchain ({len(afm_energies)})')
    
    if constrained_energies:
        ax.scatter(range(len(constrained_energies)), sorted(constrained_energies), 
                  c='green', marker='^', alpha=0.7, s=50, label=f'Constrained Scan ({len(constrained_energies)})')
    
    ax.set_xlabel('Calculation Index (sorted by energy)')
    ax.set_ylabel('Energy (Ry)')
    ax.set_title('Energy Comparison: AFM Workchain vs Constrained Scan')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    if show:
        plt.show()


def print_statistics(data: Dict[str, Any], energies_by_source: Dict[str, List[float]]):
    """Print summary statistics."""
    print("="*60)
    print("CALCULATION SOURCE ANALYSIS")
    print("="*60)
    
    # From metadata
    if 'statistics' in data and 'calculation_sources' in data['statistics']:
        print("Total calculations by source:")
        for source, count in data['statistics']['calculation_sources'].items():
            print(f"  {source}: {count}")
    
    print("\nEnergy statistics:")
    for source, energies in energies_by_source.items():
        if energies:
            print(f"\n{source.replace('_', ' ').title()}:")
            print(f"  Count: {len(energies)}")
            print(f"  Min energy: {np.min(energies):.6f} Ry")
            print(f"  Max energy: {np.max(energies):.6f} Ry")
            print(f"  Mean energy: {np.mean(energies):.6f} Ry")
            print(f"  Std dev: {np.std(energies):.6f} Ry")
        else:
            print(f"\n{source.replace('_', ' ').title()}: No calculations found")


def main():
    """Main function to demonstrate the functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot calculations by source type")
    parser.add_argument("data_file", help="JSON file from gather_workchain_data.py")
    parser.add_argument("--save-plots", help="Directory to save plots")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    
    args = parser.parse_args()
    
    # Load data
    data = load_workchain_data(args.data_file)
    
    # Extract energies by source
    energies_by_source = extract_energies_by_source(data)
    
    # Print statistics
    print_statistics(data, energies_by_source)
    
    # Create plots
    show_plots = not args.no_show
    
    if args.save_plots:
        save_dir = Path(args.save_plots)
        save_dir.mkdir(exist_ok=True)
        
        # Distribution plot
        dist_path = save_dir / "energy_distribution_by_source.png"
        plot_energy_distribution(energies_by_source, save_path=str(dist_path), show=show_plots)
        
        # Comparison plot
        comp_path = save_dir / "energy_comparison_by_source.png"
        plot_energy_comparison(energies_by_source, save_path=str(comp_path), show=show_plots)
    else:
        plot_energy_distribution(energies_by_source, show=show_plots)
        plot_energy_comparison(energies_by_source, show=show_plots)


if __name__ == "__main__":
    main()
# %%
