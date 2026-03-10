"""
Example script for loading saved layouts and inflows.

This demonstrates how to reuse layouts/inflows with different PyWake configurations
or grid settings without regenerating the layouts.
"""

import numpy as np
from py_wake.examples.data.dtu10mw import DTU10MW

from utils.resume import load_layouts_and_inflows  # noqa: F401 - re-exported


def print_dataset_summary(layouts, layout_metadata, layout_inflows):
    """Print a summary of the loaded dataset."""
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Number of layouts: {len(layouts)}")
    print(f"Inflows per layout: {layout_inflows[0].shape[0]}")
    print(f"Total samples: {len(layouts) * layout_inflows[0].shape[0]}")
    print()

    # Layout type distribution
    print("Layout type distribution:")
    unique_types, counts = np.unique(layout_metadata["types"], return_counts=True)
    for type_, count in zip(unique_types, counts):
        print(f"  {type_:20s}: {count:4d} ({count / len(layouts) * 100:.1f}%)")
    print()

    # Turbine count statistics
    n_turbines = np.array(layout_metadata["n_turbines"])
    print("Turbine count statistics:")
    print(f"  Range: {n_turbines.min()} - {n_turbines.max()}")
    print(f"  Mean:  {n_turbines.mean():.1f}")
    print(f"  Std:   {n_turbines.std():.1f}")
    print()

    # Turbine count by layout type
    print("Turbine count by layout type:")
    for type_ in unique_types:
        mask = np.array(layout_metadata["types"]) == type_
        n_turb_type = n_turbines[mask]
        print(
            f"  {type_:20s}: {n_turb_type.min():3d} - {n_turb_type.max():3d} (mean: {n_turb_type.mean():.1f})"
        )
    print()

    # Spacing statistics
    spacings = np.array(layout_metadata["spacings"])
    print("Spacing statistics (in diameters):")
    print(f"  Range: {spacings.min():.2f}D - {spacings.max():.2f}D")
    print(f"  Mean:  {spacings.mean():.2f}D")
    print()

    # Inflow statistics
    all_ws = np.concatenate([inflows[:, 0] for inflows in layout_inflows])
    all_ti = np.concatenate([inflows[:, 1] for inflows in layout_inflows])
    print("Inflow statistics:")
    print(f"  Wind speed range: {all_ws.min():.1f} - {all_ws.max():.1f} m/s")
    print(f"  Wind speed mean:  {all_ws.mean():.1f} m/s")
    print(f"  TI range:         {all_ti.min():.3f} - {all_ti.max():.3f}")
    print(f"  TI mean:          {all_ti.mean():.3f}")
    print("=" * 80)


if __name__ == "__main__":
    # Example usage
    data_dir = "./data/turbopark_10layouts_test"

    print("Loading layouts and inflows...")
    layouts, layout_metadata, layout_inflows = load_layouts_and_inflows(data_dir)

    print_dataset_summary(layouts, layout_metadata, layout_inflows)

    # Example: Access specific layout
    print("\nExample - Layout 0:")
    print(f"  Type: {layout_metadata['types'][0]}")
    print(f"  Spacing: {layout_metadata['spacings'][0]:.2f}D")
    print(f"  Number of turbines: {layout_metadata['n_turbines'][0]}")
    print(f"  Layout shape: {layouts[0].shape}")
    print(f"  Inflows shape: {layout_inflows[0].shape}")
    print("  First 3 turbine positions (in D):")
    print(layouts[0][:3])

    # Convert to meters
    wt = DTU10MW()
    D = wt.diameter()
    print(f"\n  First 3 turbine positions (in meters, D={D:.1f}m):")
    print(layouts[0][:3] * D)

    print("\n  Inflow conditions:")
    print(f"    Wind speeds (m/s): {layout_inflows[0][:, 0]}")
    print(f"    TI values:         {layout_inflows[0][:, 1]}")
