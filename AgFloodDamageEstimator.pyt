import arcpy
import os
import numpy as np
import pandas as pd

from random import Random


def parse_months(month_str):
    """Return a set of valid month numbers (1-12) from a comma string."""
    if not month_str:
        return None
    months = set()
    for m in month_str.split(','):
        m = m.strip()
        if not m:
            continue
        val = int(m)
        if not 1 <= val <= 12:
            raise ValueError("Months must be between 1 and 12")
        months.add(val)
    return months

def parse_curve(curve_str):
    try:
        points = [tuple(map(float, pt.split(':'))) for pt in curve_str.split(',')]
    except:
        raise ValueError("Damage curve must be formatted like '0:0,1:0.5,2:1'")
    if len(points) < 2:
        raise ValueError("Damage curve must contain at least two points")
    for d, f in points:
        if not (0 <= f <= 1):
            raise ValueError("Damage curve fractions must be between 0 and 1")
    return sorted(points)

def interp_damage(depth, curve):
    for i in range(1, len(curve)):
        if depth <= curve[i][0]:
            d0, f0 = curve[i-1]
            d1, f1 = curve[i]
            return f0 + (f1 - f0) * (depth - d0) / (d1 - d0)
    return curve[-1][1]

class Toolbox(object):
    def __init__(self):
        self.label = "Flood Damage Toolbox"
        self.alias = "flooddamage"
        self.tools = [AgFloodDamageEstimator]

class AgFloodDamageEstimator(object):
    def __init__(self):
        self.label = "Estimate Agricultural Flood Damage"
        self.description = "Estimate flood damage to crops using depth raster and crop raster"
        self.canRunInBackground = False

    def getParameterInfo(self):
        crop = arcpy.Parameter(displayName="Cropland Raster", name="crop_raster", datatype="Raster Layer",
                               parameterType="Required", direction="Input")
        out = arcpy.Parameter(displayName="Output Folder", name="output_folder", datatype="DEFolder",
                              parameterType="Required", direction="Output")
        val = arcpy.Parameter(displayName="Default Crop Value per Acre", name="value_acre", datatype="Double",
                              parameterType="Required", direction="Input")
        val.value = 1200
        season = arcpy.Parameter(displayName="Default Growing Season (comma separated months; blank = year-round, mismatches warn)",
                                 name="season_months", datatype="String", parameterType="Optional", direction="Input")
        season.value = "6"
        curve = arcpy.Parameter(displayName="Depth-Damage Curve (depth:fraction, comma separated)",
                                name="curve", datatype="String", parameterType="Required", direction="Input")
        curve.value = "0:1,1:1"
        event_info = arcpy.Parameter(displayName="Event Information", name="event_info", datatype="Value Table",
                                     parameterType="Required", direction="Input")
        event_info.columns = [["Raster Layer", "Raster"], ["GPLong", "Month"], ["GPLong", "Return Period"]]
        event_info.value = [["", 6, 100]]

        stddev = arcpy.Parameter(displayName="Uncertainty Std. Dev. (fraction of loss)", name="uncertainty", datatype="Double",
                                 parameterType="Required", direction="Input")
        stddev.value = 0.1
        mc = arcpy.Parameter(displayName="Monte Carlo Simulations", name="mc_runs", datatype="Long",
                             parameterType="Required", direction="Input")
        mc.value = 10
        seed = arcpy.Parameter(displayName="Random Seed", name="random_seed", datatype="Long",
                               parameterType="Required", direction="Input")
        seed.value = 10
        return [crop, out, val, season, curve, event_info, stddev, mc, seed]

    def execute(self, params, messages):
        crop_path = params[0].valueAsText
        out_dir = params[1].valueAsText
        value_acre = float(params[2].value)
        season_str = params[3].value
        curve_str = params[4].value
        event_table = params[5].values
        stddev = float(params[6].value)
        runs = int(params[7].value)
        rand = Random(int(params[8].value))

        os.makedirs(out_dir, exist_ok=True)

        damage_curve_pts = parse_curve(curve_str)
        crop_ras = arcpy.Raster(crop_path)
        crop_arr = arcpy.RasterToNumPyArray(crop_ras)

        # Cell area in acres
        desc = arcpy.Describe(crop_ras)
        meters_per_unit = desc.spatialReference.metersPerUnit
        cell_area_acres = (
            desc.meanCellWidth * desc.meanCellHeight * meters_per_unit ** 2 / 4046.8564224
        )

        # Growing season months as set of ints or None for year-round
        season_months = parse_months(season_str)

        results = []

        for row in event_table:
            depth_path, month, rp = row
            depth_str = depth_path.valueAsText if hasattr(depth_path, 'valueAsText') else str(depth_path)
            if not depth_str:
                continue
            depth_desc = arcpy.Describe(depth_str)
            if crop_ras.extent.disjoint(depth_desc.extent):
                raise ValueError(f"Depth raster {depth_str} does not overlap crop raster extent")
            label = os.path.splitext(os.path.basename(depth_str))[0].replace(" ", "_")
            aligned = os.path.join(out_dir, f"{label}_aligned.tif")
            clipped = os.path.join(out_dir, f"{label}_clipped.tif")

            arcpy.env.snapRaster = crop_ras
            arcpy.env.extent = crop_ras.extent

            arcpy.management.ProjectRaster(
                depth_str,
                aligned,
                crop_ras.spatialReference,
                "NEAREST",
                crop_ras.meanCellWidth,
            )

            # Clip to crop raster template
            arcpy.management.Clip(aligned, "#", clipped, crop_path, "0", "ClippingGeometry")

            depth_arr = arcpy.RasterToNumPyArray(clipped)
            if depth_arr.shape != crop_arr.shape:
                raise ValueError(
                    f"Masked depth raster {label} shape does not match crop raster. "
                    f"Crop: {crop_arr.shape}, Depth: {depth_arr.shape}"
                )

            # Check growing season
            try:
                month = int(month)
            except (TypeError, ValueError):
                month = None
            if season_months and month and month not in season_months:
                messages.addWarningMessage(
                    f"Event month {month} outside growing season; treated as year-round"
                )

            mask = (crop_arr > 0) & (depth_arr > 0)
            damages_run = []
            for _ in range(runs):
                damaged = 0.0
                for i in range(depth_arr.shape[0]):
                    for j in range(depth_arr.shape[1]):
                        if mask[i, j]:
                            d = depth_arr[i, j]
                            f = interp_damage(d, damage_curve_pts)
                            if stddev > 0:
                                f += rand.gauss(0, stddev)
                                f = min(max(f, 0), 1)
                            damaged += f * value_acre * cell_area_acres
                damages_run.append(damaged)
            avg_damage = float(sum(damages_run) / runs)

            results.append({"Label": label, "RP": float(rp), "Damage": avg_damage})

        if not results:
            raise ValueError("No valid events provided")

        # --- Trapezoidal EAD ---
        df_events = pd.DataFrame(results).sort_values("RP", ascending=False).reset_index(drop=True)
        probs = 1 / df_events["RP"].to_numpy()
        damages = df_events["Damage"].to_numpy()
        probs = np.concatenate(([0.0], probs, [1.0]))
        damages = np.concatenate(([damages[0]], damages, [0.0]))
        ead = float(np.trapz(damages, probs))

        with open(os.path.join(out_dir, "ead.csv"), "w") as f:
            f.write(f"EAD,{ead}\n")
        messages.addMessage(f"Expected Annual Damage (Trapezoidal): {ead:,.0f}")
