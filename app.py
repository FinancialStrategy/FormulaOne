# =========================
# CONFIG
# =========================

st.set_page_config(page_title="F1 Tire Heat Grid (Synthetic)", layout="wide")

VEHICLES_DEFAULT = [
    "Red Bull RB20",
    "Ferrari SF-24",
    "McLaren MCL38",
    "Mercedes W15",
    "Aston Martin AMR24",
    "Alpine A524",
    "Williams FW46",
    "RB VCARB 01",
    "Sauber C44",
    "Haas VF-24",
]

# Tire radius options (meters) — user can pick from combo box
RADIUS_OPTIONS_M = [0.305, 0.315, 0.325, 0.335, 0.345]


# =========================
# QUANTITATIVE SYNTHETIC MODEL
# =========================

@dataclass(frozen=True)
class VehicleParams:
    name: str
    # Scales for heat input from dynamics
    k_speed: float   # speed^2 contribution
    k_brake: float   # braking contribution
    k_latg: float    # lateral G contribution
    # Load balance affects how much heat goes into each axle
    front_bias: float  # 0..1, higher => fronts heat more
    # Baseline cooling coefficient
    cool_base: float
    # Baseline offset (car-specific)
    offset: float


def _vehicle_param_bank(vehicles: List[str], seed: int) -> Dict[str, VehicleParams]:
    """
    Deterministic per-run parameter generation so vehicles feel distinct.
    """
    rng = np.random.default_rng(seed + 101)
    bank = {}
    for v in vehicles:
        # Keep ranges plausible, not "real"
        k_speed = rng.uniform(0.0016, 0.0024)
        k_brake = rng.uniform(10.0, 16.0)
        k_latg = rng.uniform(14.0, 22.0)
        front_bias = rng.uniform(0.42, 0.58)
        cool_base = rng.uniform(0.055, 0.085)
        offset = rng.uniform(-3.0, 3.0)
        bank[v] = VehicleParams(
            name=v,
            k_speed=k_speed,
            k_brake=k_brake,
            k_latg=k_latg,
            front_bias=front_bias,
            cool_base=cool_base,
            offset=offset,
        )
    return bank


def _simulate_track_dynamics(
    n_steps: int,
    seed: int,
    track_temp_c: float,
) -> pd.DataFrame:
    """
    Synthetic lap dynamics: speed, brake, lateral_g.
    Designed to look "race-like": straights + corners + braking spikes.
    """
    rng = np.random.default_rng(seed + 202)

    t = np.arange(n_steps)
    phase = 2.0 * np.pi * t / max(1, n_steps)

    # Speed profile: baseline + harmonics + noise
    base_speed = 220 + 55 * np.sin(phase) + 25 * np.sin(3 * phase + 0.7)
    noise = rng.normal(0, 6, size=n_steps)
    speed_kph = np.clip(base_speed + noise, 70, 350)

    # Braking spikes near "corners": use a few gaussian pulses
    brake = np.zeros(n_steps, dtype=float)
    corner_centers = (np.linspace(0.08, 0.92, 8) * n_steps).astype(int_*_
