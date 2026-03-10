"""
Shared PyWake utility functions for data generation.

This module consolidates duplicated PyWake-related code patterns across the codebase.
"""

from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.examples.data.dtu10mw import power_curve as power_curve_dtu10mw
from py_wake.literature import Nygaard_2022
from py_wake.site._site import UniformSite
from py_wake.wind_farm_models import All2AllIterative, PropagateDownwind

# Default configuration for graph conversion (used in multiple files)
DEFAULT_TO_GRAPH_KWS = {
    "connectivity": "delaunay",
    "add_edge": "cartesian",
    "rel_wd": None,
}

# Default grid parameters (in rotor diameters)
DEFAULT_GRID_PARAMS = {
    "x_upstream": 10.0,
    "y_margin": 5.0,
}


def get_turbine_settings(turbine_class=DTU10MW):
    """
    Get turbine settings for inflow generation.

    Args:
        turbine_class: Turbine class to use (default: DTU10MW)

    Returns:
        tuple: (turbine_settings dict, turbine instance, diameter)
            turbine_settings contains: cutin_u, cutout_u, height_above_ground
    """
    wt = turbine_class()
    cut_in = power_curve_dtu10mw[:, 0].min()
    cut_out = power_curve_dtu10mw[:, 0].max()

    turbine_settings = {
        "cutin_u": cut_in,
        "cutout_u": cut_out,
        "height_above_ground": wt.hub_height(),
    }

    return turbine_settings, wt, wt.diameter()


def create_wind_farm_model(
    site,
    wt,
    deficit_model,
    superposition_model,
    turbulence_model,
    wind_farm_model_type,
    blockage_model=None,
):
    """
    Factory function for creating PyWake wind farm models.

    Args:
        site: PyWake site instance
        wt: Turbine instance
        deficit_model: Wake deficit model
        superposition_model: Wake superposition model
        turbulence_model: Turbulence model
        wind_farm_model_type: 'All2AllIterative' or 'PropagateDownwind'
        blockage_model: Blockage model (only used for All2AllIterative)

    Returns:
        Wind farm model instance

    Raises:
        ValueError: If wind_farm_model_type is not recognized
    """
    if wind_farm_model_type == "All2AllIterative":
        return All2AllIterative(
            site,
            wt,
            wake_deficitModel=deficit_model,
            blockage_deficitModel=blockage_model,
            superpositionModel=superposition_model,
            turbulenceModel=turbulence_model,
        )
    elif wind_farm_model_type == "PropagateDownwind":
        return PropagateDownwind(
            site,
            wt,
            wake_deficitModel=deficit_model,
            superpositionModel=superposition_model,
            turbulenceModel=turbulence_model,
        )
    else:
        raise ValueError(
            f"Unknown wind_farm_model: {wind_farm_model_type}. "
            "Use 'All2AllIterative' or 'PropagateDownwind'"
        )


def create_wind_farm_model_fresh(
    wake_config: dict = None,
    # Keep existing params as fallback for backward compatibility
    deficit_model=None,
    superposition_model=None,
    turbulence_model=None,
    wind_farm_model_type=None,
    blockage_model=None,
):
    """
    Create a wind farm model with fresh site and turbine instances.

    Useful for memory-safe processing where fresh instances prevent state accumulation.

    Args:
        wake_config: Configuration dict. If contains {"use_nygaard_2022": True},
                     returns the official TurbOPark model (Nygaard_2022). Otherwise uses
                     component-based configuration from the dict or direct arguments.
        deficit_model: Wake deficit model (fallback if wake_config not provided)
        superposition_model: Wake superposition model (fallback)
        turbulence_model: Turbulence model (fallback)
        wind_farm_model_type: 'All2AllIterative' or 'PropagateDownwind' (fallback)
        blockage_model: Blockage model (only used for All2AllIterative)

    Returns:
        tuple: (wind_farm_model, site, wt) - model instance plus site and turbine
               for access to turbine diameter etc.
    """
    site = UniformSite()
    wt = DTU10MW()

    # Check for Nygaard_2022 (complete TurbOPark model)
    if wake_config and wake_config.get("use_nygaard_2022", False):
        return Nygaard_2022(site, wt), site, wt

    # Extract params from wake_config if provided, else use direct args
    if wake_config:
        deficit_model = wake_config.get("deficit_model", deficit_model)
        superposition_model = wake_config.get("superposition_model", superposition_model)
        turbulence_model = wake_config.get("turbulence_model", turbulence_model)
        wind_farm_model_type = wake_config.get("wind_farm_model", wind_farm_model_type)
        blockage_model = wake_config.get("blockage_model", blockage_model)

    wf_model = create_wind_farm_model(
        site,
        wt,
        deficit_model,
        superposition_model,
        turbulence_model,
        wind_farm_model_type,
        blockage_model,
    )
    return wf_model, site, wt


def create_layout_stats_dict(n_wt, layout_type=None, wt_spacing=None, wake_model=None):
    """
    Create a standardized layout statistics dictionary.

    Args:
        n_wt: Number of wind turbines
        layout_type: Type of layout (e.g., 'cluster', 'single string')
        wt_spacing: Turbine spacing in rotor diameters
        wake_model: Name of the wake model used

    Returns:
        dict: Layout stats dictionary in the format {"layout_stats": {...}}
    """
    stats = {"n_wt": n_wt}

    if layout_type is not None:
        stats["layout_type"] = layout_type
    if wt_spacing is not None:
        stats["wt_spacing"] = wt_spacing
    if wake_model is not None:
        stats["wake_model"] = wake_model

    return {"layout_stats": stats}
