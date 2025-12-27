# app.py
# Streamlit Cloud-ready: Synthetic F1 tire heat grid with vehicle + tire radius selectors
# Fixes included:
# - Clean astype(int) corner centers
# - Seed widget max/value consistency (no StreamlitValueAboveMaxError)
# - Cloud-safe caching + robust typing

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="F1 Tire Heat Grid (Synthetic)", layout="wide")

# -----------------------------
# Defaults
# -----------------------------
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

RADIUS_OPTIONS_M = [0.305, 0.315, 0.325, 0.335, 0.345]
TIRE_ORDER = ["FL", "FR", "RL", "RR"]
ZONES = ["Inner", "Middle", "Outer"]


# -----------------------------
# Quantitative synthetic model
# -----------------------------
@dataclass(frozen=True)
class VehicleParams:
    name: str
    k_speed: float
    k_brake: float
    k_latg: float
    front_bias: float
    cool_base: float
    offset: float


def vehicle_param_bank(vehicles: List[str], seed: int) -> Dict[str, VehicleParams]:
    rng = np.random.default_rng(seed + 101)
    bank: Dict[str, VehicleParams] = {}
    for v in vehicles:
        bank[v] = VehicleParams(
            name=v,
            k_speed=float(rng.uniform(0.0016, 0.0024)),
            k_brake=float(rng.uniform(10.0, 16.0)),
            k_latg=float(rng.uniform(14.0, 22.0)),
            front_bias=float(rng.uniform(0.42, 0.58)),
            cool_base=float(rng.uniform(0.055, 0.085)),
            offset=float(rng.uniform(-3.0, 3.0)),
        )
    return bank


def simulate_track_dynamics(n_steps: int, seed: int, track_temp_c: float) -> pd.DataFrame:
    """
    Synthetic track dynamics:
    - speed profile: sinusoids + noise
    - braking spikes: gaussian pulses
    - lat_g correlated with braking + mild oscillation
    """
    rng = np.random.default_rng(seed + 202)

    t = np.arange(n_steps)
    phase = 2.0 * np.pi * t / max(1, n_steps)

    base_speed = 220 + 55 * np.sin(phase) + 25 * np.sin(3 * phase + 0.7)
    speed_kph = np.clip(base_speed + rng.normal(0, 6, size=n_steps), 70, 350)

    brake = np.zeros(n_steps, dtype=float)
    # ✅ FIX: clean astype(int)
    corner_centers = (np.linspace(0.08, 0.92, 8) * n_steps).astype(int)

    for c in corner_centers:
        width = int(rng.integers(12, 26))
        amp = float(rng.uniform(0.55, 1.00))
        idx = np.arange(n_steps)
        brake += amp * np.exp(-0.5 * ((idx - c) / width) ** 2)

    brake = np.clip(brake + rng.normal(0, 0.03, n_steps), 0, 1)

    lat_g = 0.8 + 2.2 * brake + 0.35 * np.sin(2 * phase + 1.3) + rng.normal(0, 0.08, n_steps)
    lat_g = np.clip(lat_g, 0.2, 4.5)

    # Track temp factor slightly shifts operating envelope
    track_factor = 1.0 + (track_temp_c - 35.0) * 0.004

    return pd.DataFrame(
        {
            "step": t,
            "speed_kph": speed_kph,
            "brake": brake,
            "lat_g": lat_g,
            "track_factor": track_factor,
        }
    )


def tire_targets(
    dyn: pd.DataFrame,
    vp: VehicleParams,
    radius_m: float,
    ambient_c: float,
    track_temp_c: float,
    track_dir_bias: float,
) -> Dict[str, np.ndarray]:
    """
    Compute per-tire target temperatures as a function of dynamics:
      heat_in ~ k_speed*v^2 + k_brake*brake + k_latg*latg^1.15
    with axle bias, left-right bias, and mild radius effect.
    """
    v_ms = dyn["speed_kph"].to_numpy() / 3.6
    brake = dyn["brake"].to_numpy()
    lat_g = dyn["lat_g"].to_numpy()
    tf = dyn["track_factor"].to_numpy()

    heat_in = (vp.k_speed * (v_ms**2) + vp.k_brake * brake + vp.k_latg * (lat_g**1.15)) * tf

    radius_peak = 1.0 - (radius_m - 0.305) * 0.25
    radius_peak = float(np.clip(radius_peak, 0.88, 1.02))

    base = (0.62 * track_temp_c + 0.38 * ambient_c) + vp.offset

    front_share = vp.front_bias
    rear_share = 1.0 - front_share

    # Track direction bias: [-1..+1]; positive makes left a bit hotter
    left_share = 0.5 + 0.06 * track_dir_bias
    right_share = 1.0 - left_share

    # Extra: braking heats fronts more; traction/speed heats rears slightly
    front_boost = 1.0 + 0.25 * brake
    vmax = max(1e-9, float(v_ms.max()))
    rear_boost = 1.0 + 0.18 * (v_ms / vmax)

    return {
        "FL": base + radius_peak * heat_in * front_share * left_share * front_boost,
        "FR": base + radius_peak * heat_in * front_share * right_share * front_boost,
        "RL": base + radius_peak * heat_in * rear_share * left_share * rear_boost,
        "RR": base + radius_peak * heat_in * rear_share * right_share * rear_boost,
    }


def evolve_temperature(
    targets: Dict[str, np.ndarray],
    vp: VehicleParams,
    radius_m: float,
    seed: int,
    ambient_c: float,
) -> Dict[str, np.ndarray]:
    """
    OU-like / AR(1)-like evolution:
      T_{t+1} = T_t + k*(target - T_t) - cool*(T_t - ambient) + noise
    Larger radius => higher inertia => slower response and slightly slower cooling.
    """
    rng = np.random.default_rng(seed + 303)

    n = len(next(iter(targets.values())))
    out: Dict[str, np.ndarray] = {}

    inertia = float(np.clip(1.0 + (radius_m - 0.305) * 7.5, 1.0, 1.35))
    k_resp = 0.16 / inertia
    cool = vp.cool_base / inertia

    for tire, tgt in targets.items():
        T = np.zeros(n, dtype=float)
        T[0] = float(tgt[0] - rng.uniform(8, 14))
        eps_scale = float(rng.uniform(0.20, 0.55))

        for i in range(n - 1):
            noise = float(rng.normal(0, eps_scale))
            T[i + 1] = T[i] + k_resp * (tgt[i] - T[i]) - cool * (T[i] - ambient_c) + noise

        out[tire] = np.clip(T, 55, 135)

    return out


def surface_from_core(core: Dict[str, np.ndarray], seed: int) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build 3-zone surface (Inner/Middle/Outer) from core temperature with mild gradients + noise.
    """
    rng = np.random.default_rng(seed + 404)
    surf: Dict[str, Dict[str, np.ndarray]] = {}

    for tire, T in core.items():
        n = len(T)
        grad = float(rng.uniform(1.2, 3.8) * (1 if tire in ["FL", "RL"] else -1))
        wobble = 0.8 * np.sin(np.linspace(0, 5 * np.pi, n) + float(rng.uniform(0, 2 * np.pi)))

        inner = np.clip(T + grad + wobble + rng.normal(0, 0.35, n), 50, 140)
        mid = np.clip(T + 0.15 * wobble + rng.normal(0, 0.30, n), 50, 140)
        outer = np.clip(T - grad + 0.6 * wobble + rng.normal(0, 0.35, n), 50, 140)

        surf[tire] = {"Inner": inner, "Middle": mid, "Outer": outer}

    return surf


@st.cache_data(show_spinner=False)
def generate_data(
    vehicles: List[str],
    radius_m: float,
    n_laps: int,
    points_per_lap: int,
    seed: int,
    ambient_c: float,
    track_temp_c: float,
) -> pd.DataFrame:
    """
    Long-form dataset:
    vehicle, radius_m, lap, step_in_lap, global_step, tire, zone, temp_c + dynamics
    """
    n_steps = n_laps * points_per_lap
    dyn = simulate_track_dynamics(n_steps=n_steps, seed=seed, track_temp_c=track_temp_c)

    bank = vehicle_param_bank(vehicles, seed=seed)
    track_dir_bias = float(np.sin((radius_m - 0.305) * 12.0))  # deterministic mild bias

    frames = []
    global_step = np.arange(n_steps)
    lap = (global_step // points_per_lap) + 1
    step_in_lap = (global_step % points_per_lap) + 1

    for i, v in enumerate(vehicles):
        vp = bank[v]
        targets = tire_targets(
            dyn=dyn,
            vp=vp,
            radius_m=radius_m,
            ambient_c=ambient_c,
            track_temp_c=track_temp_c,
            track_dir_bias=track_dir_bias,
        )
        core = evolve_temperature(targets, vp=vp, radius_m=radius_m, seed=seed + i * 29, ambient_c=ambient_c)
        surface = surface_from_core(core, seed=seed + i * 31)

        for tire in TIRE_ORDER:
            for zone in ZONES:
                temp = surface[tire][zone]
                frames.append(
                    pd.DataFrame(
                        {
                            "vehicle": v,
                            "radius_m": radius_m,
                            "lap": lap,
                            "step_in_lap": step_in_lap,
                            "global_step": global_step,
                            "tire": tire,
                            "zone": zone,
                            "temp_c": temp,
                            "speed_kph": dyn["speed_kph"].to_numpy(),
                            "brake": dyn["brake"].to_numpy(),
                            "lat_g": dyn["lat_g"].to_numpy(),
                        }
                    )
                )

    return pd.concat(frames, ignore_index=True)


# -----------------------------
# Plot helpers
# -----------------------------
def tire_tile_heatmap(z: np.ndarray, title: str, zmin: float, zmax: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=["Left", "Right"],
            y=["Front", "Rear"],
            zmin=zmin,
            zmax=zmax,
            showscale=False,
            hovertemplate="Position=%{y} %{x}<br>Temp=%{z:.1f}°C<extra></extra>",
        )
    )

    names = np.array([["FL", "FR"], ["RL", "RR"]])
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=["Left", "Right"][j],
                y=["Front", "Rear"][i],
                text=f"{names[i, j]}<br><b>{z[i, j]:.1f}°C</b>",
                showarrow=False,
                font=dict(size=12),
            )

    fig.update_layout(
        title=title,
        height=260,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
    return fig


def timeseries_plot(df_focus: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for tire in TIRE_ORDER:
        d = df_focus[df_focus["tire"] == tire].sort_values("global_step")
        fig.add_trace(
            go.Scatter(
                x=d["global_step"],
                y=d["temp_c"],
                mode="lines",
                name=tire,
                hovertemplate="Step=%{x}<br>Temp=%{y:.1f}°C<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Global step",
        yaxis_title="Temp (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("🏎️ F1 Tire Heat Grid — Synthetic Quant Model")
st.caption("Synthetic tire heat map generator (NOT real telemetry). Vehicles share the same track dynamics for comparison.")

with st.sidebar:
    st.header("Controls")

    # ✅ FIX: max/value consistency (no StreamlitValueAboveMaxError)
    DEFAULT_SEED = 20_251_227
    MAX_SEED = 2_147_483_647  # safe 32-bit signed max
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=MAX_SEED,
        value=min(DEFAULT_SEED, MAX_SEED),
        step=1,
    )

    vehicles = st.multiselect(
        "Vehicles (select one or more)",
        options=VEHICLES_DEFAULT,
        default=VEHICLES_DEFAULT[:6],
    )

    radius_m = st.selectbox(
        "Tire radius (m)",
        options=RADIUS_OPTIONS_M,
        index=2,
        format_func=lambda x: f"{x:.3f} m",
    )

    zone = st.selectbox("Surface zone", options=ZONES, index=1)

    n_laps = st.slider("Laps", min_value=3, max_value=35, value=12, step=1)
    points_per_lap = st.slider("Points per lap", min_value=40, max_value=240, value=120, step=10)

    ambient_c = st.slider("Ambient temp (°C)", min_value=10.0, max_value=45.0, value=27.0, step=0.5)
    track_temp_c = st.slider("Track temp (°C)", min_value=20.0, max_value=65.0, value=42.0, step=0.5)

    st.divider()
    window_mode = st.radio("Heat tile aggregation", ["Last N steps", "Specific lap"], index=0)

    last_n: Optional[int] = None
    selected_lap: Optional[int] = None

    if window_mode == "Last N steps":
        last_n = st.slider("N steps", 20, min(800, n_laps * points_per_lap), 240, 10)
    else:
        selected_lap = st.slider("Lap #", 1, n_laps, min(5, n_laps))

    st.divider()
    focus_vehicle = st.selectbox("Focus vehicle (time-series)", options=vehicles if vehicles else VEHICLES_DEFAULT)

if not vehicles:
    st.warning("Select at least one vehicle from the sidebar.")
    st.stop()

df_all = generate_data(
    vehicles=vehicles,
    radius_m=float(radius_m),
    n_laps=int(n_laps),
    points_per_lap=int(points_per_lap),
    seed=int(seed),
    ambient_c=float(ambient_c),
    track_temp_c=float(track_temp_c),
)

df = df_all[df_all["zone"] == zone].copy()

# robust range to keep consistent color scaling across tiles
zmin = float(df["temp_c"].quantile(0.02))
zmax = float(df["temp_c"].quantile(0.98))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Vehicles selected", len(vehicles))
c2.metric("Radius", f"{radius_m:.3f} m")
c3.metric("Zone", zone)
c4.metric("Rows", f"{len(df):,}")

st.subheader("Tire Heat Grid (selected vehicles)")

if window_mode == "Last N steps" and last_n is not None:
    df_slice = df[df["global_step"] >= df["global_step"].max() - int(last_n) + 1]
    slice_label = f"Last {last_n} steps"
elif window_mode == "Specific lap" and selected_lap is not None:
    df_slice = df[df["lap"] == int(selected_lap)]
    slice_label = f"Lap {selected_lap}"
else:
    df_slice = df
    slice_label = "All"

ncols = 3 if len(vehicles) >= 4 else 2
cols = st.columns(ncols)

for i, v in enumerate(vehicles):
    d = df_slice[df_slice["vehicle"] == v]
    agg = d.groupby("tire")["temp_c"].mean()

    z = np.array(
        [
            [float(agg.get("FL", np.nan)), float(agg.get("FR", np.nan))],
            [float(agg.get("RL", np.nan)), float(agg.get("RR", np.nan))],
        ],
        dtype=float,
    )

    fig = tire_tile_heatmap(
        z=z,
        title=f"{v}<br><sup>{slice_label} • {zone} • r={radius_m:.3f}m</sup>",
        zmin=zmin,
        zmax=zmax,
    )
    with cols[i % ncols]:
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Focus: Tire Temperature Time-Series")

df_focus = df[df["vehicle"] == focus_vehicle].copy()
fig_ts = timeseries_plot(df_focus, title=f"{focus_vehicle} — {zone} temps over time (r={radius_m:.3f}m)")
st.plotly_chart(fig_ts, use_container_width=True)

with st.expander("Preview data"):
    st.dataframe(df_all.head(200), use_container_width=True)

csv_bytes = df_all.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download dataset (CSV)",
    data=csv_bytes,
    file_name=f"f1_tire_heat_{zone}_r{radius_m:.3f}m.csv",
    mime="text/csv",
)
