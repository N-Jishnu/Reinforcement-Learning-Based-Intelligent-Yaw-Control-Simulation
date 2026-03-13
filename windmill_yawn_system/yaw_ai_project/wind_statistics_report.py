"""
Wind Statistics Report Generator
================================
Generates comprehensive statistics and visualizations for the wind field model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulation.wind_field import WindField
from config import SIMULATION_DURATION, TIME_STEP


def generate_wind_statistics(data_path, output_dir="results"):
    """Generate wind field statistics and visualizations."""
    
    print("=" * 60)
    print("WIND FIELD STATISTICS REPORT")
    print("=" * 60)
    
    wind_field = WindField(
        duration=SIMULATION_DURATION,
        dt=TIME_STEP,
        data_path=data_path
    )
    
    num_samples = min(5000, wind_field.requested_steps)
    
    speeds = []
    directions = []
    
    print("\nGenerating wind samples...")
    for t in range(num_samples):
        speed, direction = wind_field.get_wind_data(t)
        speeds.append(speed)
        directions.append(direction)
    
    speeds = np.array(speeds)
    directions = np.array(directions)
    
    print("\n" + "-" * 40)
    print("WIND SPEED STATISTICS")
    print("-" * 40)
    print(f"  Mean Speed:        {np.mean(speeds):.2f} m/s")
    print(f"  Std Dev:           {np.std(speeds):.2f} m/s")
    print(f"  Min Speed:         {np.min(speeds):.2f} m/s")
    print(f"  Max Speed:         {np.max(speeds):.2f} m/s")
    print(f"  Median Speed:      {np.median(speeds):.2f} m/s")
    
    print("\n" + "-" * 40)
    print("WIND DIRECTION STATISTICS")
    print("-" * 40)
    print(f"  Mean Direction:    {np.mean(directions):.2f} deg")
    print(f"  Std Dev:           {np.std(directions):.2f} deg")
    print(f"  Min Direction:     {np.min(directions):.2f} deg")
    print(f"  Max Direction:     {np.max(directions):.2f} deg")
    
    turbulence_actual = np.std(speeds) / np.mean(speeds)
    print(f"\n  Turbulence Intensity (actual): {turbulence_actual:.3f} ({turbulence_actual*100:.1f}%)")
    
    print("\n" + "-" * 40)
    print("TURBINE OPERATING CONDITIONS")
    print("-" * 40)
    cut_in = 3.0
    cut_out = 25.0
    rated = 11.0
    
    in_region1 = np.sum(speeds < cut_in) / len(speeds) * 100
    in_region2 = np.sum((speeds >= cut_in) & (speeds < rated)) / len(speeds) * 100
    in_region3 = np.sum((speeds >= rated) & (speeds < cut_out)) / len(speeds) * 100
    in_region4 = np.sum(speeds >= cut_out) / len(speeds) * 100
    
    print(f"  Region 1 (< {cut_in} m/s):     {in_region1:.1f}% (below cut-in)")
    print(f"  Region 2 ({cut_in}-{rated} m/s):  {in_region2:.1f}% (partial load)")
    print(f"  Region 3 ({rated}-{cut_out} m/s): {in_region3:.1f}% (rated power)")
    print(f"  Region 4 (> {cut_out} m/s):    {in_region4:.1f}% (cut-out)")
    
    print("\n" + "-" * 40)
    print("MODEL PARAMETERS")
    print("-" * 40)
    print(f"  Turbulence Intensity:  {wind_field.turbulence_intensity}")
    print(f"  Gust Probability:       {wind_field.gust_probability}")
    print(f"  Gust Strength Range:   {wind_field.gust_strength}")
    print(f"  Shift Probability:      {wind_field.shift_probability}")
    
    print("\n" + "=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Wind Field Analysis", fontsize=14, fontweight='bold')
    
    axes[0, 0].plot(speeds[:1000], 'b-', alpha=0.7)
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Wind Speed (m/s)")
    axes[0, 0].set_title("Wind Speed Time Series (first 1000 steps)")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(speeds, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(speeds), color='r', linestyle='--', label=f'Mean: {np.mean(speeds):.1f}')
    axes[0, 1].set_xlabel("Wind Speed (m/s)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Wind Speed Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(directions[:1000], 'g-', alpha=0.7)
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Wind Direction (deg)")
    axes[1, 0].set_title("Wind Direction Time Series (first 1000 steps)")
    axes[1, 0].grid(True, alpha=0.3)
    
    axPolar = fig.add_subplot(2, 2, 4, projection='polar')
    dir_rad = np.radians(directions)
    axPolar.hist(dir_rad, bins=36, alpha=0.7, color='orange')
    axPolar.set_title("Wind Direction Distribution")
    
    axes[1, 1].remove()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/wind_statistics.png", dpi=150)
    print(f"\nPlots saved to {output_dir}/wind_statistics.png")
    
    return wind_field


if __name__ == "__main__":
    generate_wind_statistics("data/era5_chennai.csv")
