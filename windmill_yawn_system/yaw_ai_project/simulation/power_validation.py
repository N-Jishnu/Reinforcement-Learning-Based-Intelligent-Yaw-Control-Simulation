import numpy as np
import matplotlib.pyplot as plt
from simulation.power_model import calculate_power

def main():

    wind_speed = 10.0  # constant test speed
    wind_direction = 0.0

    misalignments = np.linspace(-90, 90, 180)
    powers = []

    for error in misalignments:
        yaw_angle = wind_direction - error
        power = calculate_power(wind_speed, wind_direction, yaw_angle)
        powers.append(power)

    plt.figure()
    plt.plot(misalignments, np.array(powers) / 1e6)
    plt.xlabel("Yaw Misalignment (degrees)")
    plt.ylabel("Power (MW)")
    plt.title("Power vs Yaw Misalignment Validation")
    plt.grid(True)

    plt.savefig("results/power_curve_validation.png")
    print("Power validation plot saved → results/power_curve_validation.png")

if __name__ == "__main__":
    main()