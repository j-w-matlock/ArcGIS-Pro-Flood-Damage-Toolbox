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

        base_crop_arr = arcpy.RasterToNumPyArray(crop_raster)
        counts = Counter(base_crop_arr.flatten())
        counts.pop(0, None)
        top_codes = [c for c, _ in counts.most_common(50)]

        def _parse_months(month_str, context):
            months: List[int] = []
            for m in str(month_str).split(','):
                m = m.strip()
                if not m:
                    continue
                try:
                    mi = int(m)
                except Exception:
                    raise ValueError(f"Invalid month '{m}' in {context}; must be 1-12")
                if mi < 1 or mi > 12:
                    raise ValueError(f"Month {mi} in {context} out of range 1-12")
                months.append(mi)
            return months

        crop_table: Dict[int, Dict[str, object]] = {}
        if crop_csv:
            df_csv = pd.read_csv(crop_csv)
            required = {"CropCode", "ValuePerAcre", "GrowingSeason"}
            missing = required - set(df_csv.columns)
            if missing:
                raise ValueError(
                    f"Crop CSV missing required columns: {', '.join(sorted(missing))}"
                )
            for idx, row in df_csv.iterrows():
                try:
                    code = int(row["CropCode"])
                except Exception:
                    raise ValueError(
                        f"Invalid CropCode at row {idx}: {row['CropCode']}"
                    )
                try:
                    value = float(row["ValuePerAcre"])
                except Exception:
                    raise ValueError(
                        f"Invalid ValuePerAcre for crop code {row['CropCode']}"
                    )
                months = _parse_months(row["GrowingSeason"], f"crop code {code}")
                crop_table[code] = {"Value": value, "GrowingSeason": months}
        else:
            months = _parse_months(default_months, "default growing season")
            for code in top_codes:
                crop_table[code] = {
                    "Value": float(default_val),
                    "GrowingSeason": months,
                }

        crop_table = {c: v for c, v in crop_table.items() if c in top_codes}

        def _safe(name: str) -> str:
            name = os.path.splitext(os.path.basename(str(name)))[0]
            name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
            return name.strip("_")

        event_table: Dict[str, Dict[str, float]] = {}
        for row in event_info:
            if len(row) < 3:
                raise ValueError(
                    "Event information rows must include Raster, Month, and Return Period"
                )
            raster = row[0]
            if not arcpy.Exists(raster):
                raise ValueError(f"Raster path does not exist: {raster}")
            try:
                month = int(str(row[1]))
            except Exception:
                raise ValueError(f"Invalid Month '{row[1]}' for raster {raster}")
            if month < 1 or month > 12:
                raise ValueError(f"Month {month} for raster {raster} out of range 1-12")
            try:
                rp = float(str(row[2]))
            except Exception:
                raise ValueError(f"Invalid Return Period '{row[2]}' for raster {raster}")
            if rp <= 0:
                raise ValueError(
                    f"Return Period must be positive for raster {raster}"
                )
            label = _safe(raster)
            event_table[label] = {"Month": month, "RP": rp}

        messages.addMessage(f"Top 50 crop codes: {list(crop_table.keys())}")
