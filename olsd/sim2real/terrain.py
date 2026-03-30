"""Procedural MuJoCo terrain XML generation."""

from __future__ import annotations

import math

from olsd.schema import TerrainType


def generate_terrain_xml(
    terrain_type: TerrainType,
    size: tuple[float, float] = (20.0, 20.0),
    params: dict | None = None,
) -> str:
    """
    Generate a MuJoCo XML snippet for a supported terrain type.

    The returned string is meant to be inserted inside ``<worldbody>``.
    """
    params = params or {}
    friction = params.get("friction", 1.0)
    rgba = params.get("rgba", "0.15 0.15 0.18 1")
    width, length = size

    if terrain_type == TerrainType.FLAT:
        return (
            f'<geom name="ground" type="plane" size="{width} {length} 0.1" '
            f'rgba="{rgba}" condim="3" friction="{friction} 0.5 0.01" '
            'conaffinity="1" contype="1"/>'
        )

    if terrain_type == TerrainType.SLOPE:
        angle_deg = float(params.get("angle_deg", 15.0))
        approach_length = float(params.get("approach_length", 1.0))
        ramp_length = float(params.get("ramp_length", 3.0))
        thickness = float(params.get("thickness", 0.2))
        half_height = max(0.05, thickness / 2.0)
        ramp_center_x = ramp_length / 2.0
        ramp_center_z = (ramp_length * 0.5 * math.tan(math.radians(angle_deg))) - half_height
        return (
            f'<geom name="slope_approach" type="box" pos="{approach_length / 2.0 - approach_length} 0 {-half_height}" '
            f'size="{approach_length / 2.0} {length} {half_height}" rgba="{rgba}" '
            f'condim="3" friction="{friction} 0.5 0.01" conaffinity="1" contype="1"/>\n'
            f'<geom name="slope_ground" type="box" pos="{ramp_center_x} 0 {ramp_center_z}" '
            f'size="{ramp_length / 2.0} {length} {half_height}" euler="0 {angle_deg} 0" '
            f'rgba="{rgba}" condim="3" friction="{friction} 0.5 0.01" '
            'conaffinity="1" contype="1"/>'
        )

    if terrain_type == TerrainType.STAIRS:
        step_height = float(params.get("step_height", 0.1))
        step_width = float(params.get("step_width", 0.3))
        n_steps = int(params.get("n_steps", 8))
        y_half = float(params.get("stair_span_y", 1.6))
        geoms = [
            (
                f'<geom name="stairs_base" type="box" pos="0 0 {-step_height}" '
                f'size="{width} {length} {step_height}" rgba="{rgba}" '
                f'condim="3" friction="{friction} 0.5 0.01" conaffinity="1" contype="1"/>'
            )
        ]
        for idx in range(n_steps):
            x_pos = (idx * step_width) + (step_width / 2.0)
            z_pos = ((idx + 1) * step_height / 2.0) - step_height
            geoms.append(
                f'<geom name="stair_{idx}" type="box" pos="{x_pos} 0 {z_pos}" '
                f'size="{step_width / 2.0} {y_half} {(idx + 1) * step_height / 2.0}" '
                f'rgba="{rgba}" condim="3" friction="{friction} 0.5 0.01" '
                'conaffinity="1" contype="1"/>'
            )
        return "\n".join(geoms)

    raise ValueError(
        f"Unsupported terrain_type={terrain_type.value}. "
        "Supported: flat, slope, stairs."
    )
