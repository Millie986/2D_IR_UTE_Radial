import math

ACTIVE_NUCLEUS = '1H'

GYROMAGNETIC_RATIOS = {
    '1H': 42.578e6,
    '2H': 6.536e6,
    '3He': -32.434e6,
    '7Li': 16.548e6,
    '13C': 10.708e6,
    '15N': -4.315e6,
    '17O': -5.772e6,
    '19F': 40.069e6,
    '23Na': 11.262e6,
    '31P': 17.235e6,
    '35Cl': 2.624e6,
    '39K': 1.989e6,
    '129Xe': -11.777e6
}  # In Hz/T

GAMMA, GAMMA_RAD = GYROMAGNETIC_RATIOS[ACTIVE_NUCLEUS], GYROMAGNETIC_RATIOS[ACTIVE_NUCLEUS] * 2 * math.pi

EPSILON = 2.22e-16
