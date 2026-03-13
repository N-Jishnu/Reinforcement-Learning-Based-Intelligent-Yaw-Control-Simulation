"""
Power Output Validation Plots
==============================
Validates the turbine power model and generates comprehensive plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation.power_model import calculate_power
from config import (
    CUT_IN_SPEED,
    CUT_OUT_SPEED,
    RATED_SPEED,
    RATED_POWER,
    ROTOR_AREA,
    AIR_DENSITY,
    POWER_COEFFICIENT_MAX
)

print("=" * 60)
print("POWER MODEL VALIDATION")
print("=" * 60)

wind_speeds = np.linspace(0, 30, 300)
power_curve = [calculate_power(v, 0, 0) for v in wind_speeds]

theoretical_max = 0.5 * AIR_DENSITY * ROTOR_AREA * (RATED_SPEED ** 3) * POWER_COEFFICIENT_MAX

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Turbine Power Model Validation", fontsize=14, fontweight='bold')

axes[0, 0].plot(wind_speeds, [p/1e6 for p in power_curve], 'b-', linewidth=2)
axes[0, 0].axvline(CUT_IN_SPEED, color='g', linestyle='--', label=f'Cut-in: {CUT_IN_SPEED} m/s')
axes[0, 0].axvline(RATED_SPEED, color='orange', linestyle='--', label=f'Rated: {RATED_SPEED} m/s')
axes[0, 0].axvline(CUT_OUT_SPEED, color='r', linestyle='--', label=f'Cut-out: {CUT_OUT_SPEED} m/s')
axes[0, 0].axhline(RATED_POWER/1e6, color='purple', linestyle=':', label=f'Rated: {RATED_POWER/1e6:.0f} MW')
axes[0, 0].set_xlabel("Wind Speed (m/s)")
axes[0, 0].set_ylabel("Power (MW)")
axes[0, 0].set_title("Power Curve")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(0, 30)

test_speed = 10.0
yaw_errors = np.linspace(-90, 90, 180)
misalignment_power = [calculate_power(test_speed, 0, err) for err in yaw_errors]

axes[0, 1].plot(yaw_errors, [p/1e6 for p in misalignment_power], 'r-', linewidth=2)
axes[0, 1].axvline(0, color='g', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel("Yaw Error (deg)")
axes[0, 1].set_ylabel("Power (MW)")
axes[0, 1].set_title(f"Yaw Misalignment Loss @ {test_speed} m/s")
axes[0, 1].grid(True, alpha=0.3)

power_at_0 = calculate_power(test_speed, 0, 0)
axes[0, 1].axhline(power_at_0/1e6 * 0.95, color='orange', linestyle=':', label='95% of max')
for thresh in [15, 30, 45]:
    p_thresh = calculate_power(test_speed, 0, thresh)
    loss_pct = (1 - p_thresh/power_at_0) * 100
    axes[0, 1].annotate(f'{loss_pct:.1f}% loss\nat {thresh}°', 
                        xy=(thresh, p_thresh/1e6),
                        xytext=(thresh+15, p_thresh/1e6-0.3),
                        fontsize=8,
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
axes[0, 1].legend()

axes[1, 0].plot(yaw_errors, [np.cos(np.radians(err))**3 * 100 for err in yaw_errors], 'g-', linewidth=2)
axes[1, 0].set_xlabel("Yaw Error (deg)")
axes[1, 0].set_ylabel("Alignment Factor (%)")
axes[1, 0].set_title("Alignment Factor vs Yaw Error")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(0, 105)

speeds_for_3d = np.linspace(3, 15, 50)
errors_for_3d = np.linspace(0, 60, 50)
S, E = np.meshgrid(speeds_for_3d, errors_for_3d)
P = np.array([calculate_power(s, 0, e) for s, e in zip(S.ravel(), E.ravel())]).reshape(S.shape)

im = axes[1, 1].contourf(S, E, P/1e6, levels=20, cmap='viridis')
axes[1, 1].set_xlabel("Wind Speed (m/s)")
axes[1, 1].set_ylabel("Yaw Error (deg)")
axes[1, 1].set_title("Power Surface (MW)")
plt.colorbar(im, ax=axes[1, 1], label="Power (MW)")

plt.tight_layout()
plt.savefig("results/power_validation.png", dpi=150)
print("Power validation plot saved to results/power_validation.png")

print("\n" + "-" * 40)
print("POWER MODEL SUMMARY")
print("-" * 40)
print(f"  Rated Power:      {RATED_POWER/1e6:.2f} MW")
print(f"  Theoretical Max: {theoretical_max/1e6:.2f} MW @ {RATED_SPEED} m/s")
print(f"  Cut-in Speed:    {CUT_IN_SPEED} m/s")
print(f"  Rated Speed:     {RATED_SPEED} m/s")
print(f"  Cut-out Speed:   {CUT_OUT_SPEED} m/s")
print(f"  Rotor Area:      {ROTOR_AREA:.1f} m²")
print(f"  Air Density:     {AIR_DENSITY} kg/m³")
print(f"  Max Cp:          {POWER_COEFFICIENT_MAX}")

power_at_rated = calculate_power(RATED_SPEED, 0, 0)
power_at_8 = calculate_power(8.0, 0, 0)
power_at_12 = calculate_power(12.0, 0, 0)

print("\n" + "-" * 40)
print("VALIDATION CHECKPOINTS")
print("-" * 40)
print(f"  @ 8 m/s (Region 2):  {power_at_8/1e6:.3f} MW")
print(f"  @ {RATED_SPEED} m/s (Region 3): {power_at_rated/1e6:.3f} MW (rated: {RATED_POWER/1e6:.1f} MW)")
print(f"  @ 12 m/s (Region 3): {power_at_12/1e6:.3f} MW")
print(f"  @ 0° yaw error:      {calculate_power(10, 0, 0)/1e6:.3f} MW")
print(f"  @ 30° yaw error:     {calculate_power(10, 0, 30)/1e6:.3f} MW ({(1-calculate_power(10,0,30)/calculate_power(10,0,0))*100:.1f}% loss)")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
