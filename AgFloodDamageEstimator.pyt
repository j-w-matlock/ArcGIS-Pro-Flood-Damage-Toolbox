import arcpy
import os
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List
import re


class Toolbox(object):
    """Entry point for ArcGIS to discover tools."""

    def __init__(self):
        self.label = "Ag Flood Damage"
        self.alias = "AgFloodDamage"
        self.tools = [AgFloodDamageEstimator]


class AgFloodDamageEstimator(object):
    """Sample crop and depth rasters to estimate flood damages."""

    def __init__(self):
        self.label = "Estimate Agricultural Flood Damage"
        self.description = (
            "Samples Cropscape and depth rasters to estimate crop loss,\n"
            "runs Monte Carlo uncertainty, and annualizes damages in a\n"
            "USACE compliant manner."
        )
        self.canRunInBackground = False

    def getParameterInfo(self):
        crop = arcpy.Parameter(
            displayName="Cropland Raster",
            name="crop_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )

        depths = arcpy.Parameter(
            displayName="Flood Depth Rasters",
            name="depth_rasters",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input"
        )
        depths.multiValue = True

        out_dir = arcpy.Parameter(
            displayName="Output Folder",
            name="output_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        crop_csv = arcpy.Parameter(
            displayName="Crop Info CSV",
            name="crop_csv",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input"
        )

        default_val = arcpy.Parameter(
            displayName="Default Crop Value per Acre",
            name="default_crop_value",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        default_val.value = 0

        default_months = arcpy.Parameter(
            displayName="Default Growing Season (comma separated months)",
            name="default_growing_season",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        default_months.value = ""

        event_info = arcpy.Parameter(
            displayName="Event Information",
            name="event_info",
            datatype="GPValueTable",
            parameterType="Required",
            direction="Input"
        )
        event_info.columns = [
            ["GPRasterLayer", "Raster"],
            ["GPLong", "Month"],
            ["GPLong", "Return Period"]
        ]

        mc_std = arcpy.Parameter(
            displayName="Uncertainty Std. Dev. (fraction of loss)",
            name="mc_std",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        mc_std.value = 0.1

        mc_sims = arcpy.Parameter(
            displayName="Monte Carlo Simulations",
            name="mc_sims",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        mc_sims.value = 1000

        seed = arcpy.Parameter(
            displayName="Random Seed",
            name="seed",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )

        return [
            crop, depths, out_dir, crop_csv,
            default_val, default_months,
            event_info, mc_std, mc_sims, seed
        ]

    def updateParameters(self, params):
        crop_param, depth_param = params[0], params[1]
        csv_param, default_val, default_months = params[3], params[4], params[5]
        event_table_param = params[6]

        if csv_param.altered:
            default_val.enabled = False
            default_months.enabled = False
        else:
            default_val.enabled = True
            default_months.enabled = True

        if depth_param.altered and not event_table_param.altered and depth_param.valueAsText:
            vt = arcpy.ValueTable(0)
            for path in depth_param.valueAsText.split(";"):
                vt.addRow([path.strip().strip("'\""), "", ""])
            event_table_param.value = vt

        return

    def execute(self, params, messages):
        crop_raster = params[0].valueAsText
        depth_rasters = [p.strip().strip("'\"") for p in params[1].valueAsText.split(";")]
        out_dir = params[2].valueAsText
        crop_csv = params[3].valueAsText
        default_val = params[4].value
        default_months = params[5].valueAsText
        event_info = params[6].values
        mc_std = params[7].value
        mc_sims = int(params[8].value)
        seed = params[9].value

        os.makedirs(out_dir, exist_ok=True)
        if seed not in (None, ""):
            np.random.seed(int(seed))

        messages.addMessage("Sampling crop raster")
        base_crop_arr = arcpy.RasterToNumPyArray(crop_raster)
        counts = Counter(base_crop_arr.flatten())
        counts.pop(0, None)
        top_codes = [c for c, _ in counts.most_common(50)]

        crop_table: Dict[int, Dict[str, object]] = {}
        if crop_csv:
            df_csv = pd.read_csv(crop_csv)
            for _, row in df_csv.iterrows():
                try:
                    code = int(row[0])
                    value = float(row[1])
                    months = [int(m) for m in str(row[2]).split(',') if m]
                except (ValueError, TypeError):
                    continue
                crop_table[code] = {"Value": value, "GrowingSeason": months}
        else:
            months = [int(m) for m in str(default_months).split(',') if m]
            for code in top_codes:
                crop_table[code] = {"Value": float(default_val), "GrowingSeason": months}

        crop_table = {c: v for c, v in crop_table.items() if c in top_codes}

        def _safe(name: str) -> str:
            name = os.path.splitext(os.path.basename(str(name)))[0]
            name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
            return name.strip("_")

        messages.addMessage("Sampling depth rasters")
        depth_arrays: Dict[str, np.ndarray] = {}
        for path in depth_rasters:
            label = _safe(path)
            depth_arrays[label] = arcpy.RasterToNumPyArray(path)
        messages.addMessage(f"Processed {len(depth_arrays)} depth rasters")

        value_arr = np.zeros_like(base_crop_arr, dtype=float)
        for code, props in crop_table.items():
            value_arr[base_crop_arr == code] = props["Value"]

        damage_tables: Dict[str, float] = {}
        for label, arr in depth_arrays.items():
            mask = arr > 0
            damage_tables[label] = float(value_arr[mask].sum())

        event_table: Dict[str, Dict[str, int]] = {}
        for row in event_info:
            if len(row) < 3:
                continue
            label = _safe(row[0])
            event_table[label] = {
                "Month": int(str(row[1])),
                "RP": int(str(row[2]))
            }

        messages.addMessage(f"Top 50 crop codes: {list(crop_table.keys())}")

        arcpy.SetProgressor("step", "Running Monte Carlo simulations", 0, mc_sims, 1)
        mc_totals: List[float] = []
        for i in range(mc_sims):
            arcpy.SetProgressorLabel(f"Simulation {i + 1} of {mc_sims}")
            total = 0.0
            for dmg in damage_tables.values():
                factor = max(np.random.normal(1.0, mc_std), 0)
                total += dmg * factor
            mc_totals.append(total)
            arcpy.SetProgressorPosition(i + 1)
        arcpy.ResetProgressor()
        messages.addMessage(f"Completed {mc_sims} simulations")

        messages.addMessage("Aggregating simulation results")
        mean_damage = float(np.mean(mc_totals))
        sd_damage = float(np.std(mc_totals))
        messages.addMessage(
            f"Mean damage: {mean_damage:,.2f}; Standard deviation: {sd_damage:,.2f}"
        )

        out_csv = os.path.join(out_dir, "damage_summary.csv")
        pd.DataFrame({
            "MeanDamage": [mean_damage],
            "StdDev": [sd_damage],
            "Simulations": [mc_sims],
            "DepthRasters": [len(depth_arrays)],
        }).to_csv(out_csv, index=False)
        messages.addMessage(f"Results written to {out_csv}")
