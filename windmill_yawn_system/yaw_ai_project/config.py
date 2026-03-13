
# Simulation Parameters
SIMULATION_DURATION = 1000  # seconds
TIME_STEP = 1.0  # seconds

# Physics Constants
AIR_DENSITY = 1.225  # kg/m^3

# Turbine Specifications (Based on a generic 2MW turbine)
RATED_POWER = 2.0e6  # Watts
ROTOR_DIAMETER = 80.0  # meters
ROTOR_AREA = 3.14159 * (ROTOR_DIAMETER / 2) ** 2
CUT_IN_SPEED = 3.0  # m/s
RATED_SPEED = 11.0  # m/s
CUT_OUT_SPEED = 25.0  # m/s
POWER_COEFFICIENT_MAX = 0.45  # Betz limit is 0.59, practical is lower

# Yaw Control Parameters
MAX_YAW_RATE = 2.0  # degrees per second (mechanical limit)
MAX_YAW_ACCELERATION = 0.1 # degrees per second squared
YAW_ERROR_THRESHOLD = 5.0  # degrees

# Turbine Dynamics Parameters
YAW_INERTIA = 0.15  # smoothing factor (0 = no inertia, 0.3 = heavy inertia)
ACTUATION_DELAY = 3  # timesteps of delay (simulates motor/controller lag)
