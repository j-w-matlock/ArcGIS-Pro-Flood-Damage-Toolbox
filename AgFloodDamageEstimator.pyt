import arcpy
import os
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List


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

    # ------------------------------------------------------------------
    # Parameter definitions
    # ------------------------------------------------------------------
    def getParameterInfo(self):
        crop = arcpy.Parameter(
            displayName="Cropland Raster",
            name="crop_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
        )

        depths = arcpy.Parameter(
            displayName="Flood Depth Rasters",
            name="depth_rasters",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
            multiValue=True,
        )

        out_dir = arcpy.Parameter(
            displayName="Output Folder",
            name="output_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input",
        )

        crop_csv = arcpy.Parameter(
            displayName="Crop Info CSV",
            name="crop_csv",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
        )

        default_val = arcpy.Parameter(
            displayName="Default Crop Value per Acre",
            name="default_crop_value",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
        )
        default_val.value = 0

        default_months = arcpy.Parameter(
            displayName="Default Growing Season (comma separated months)",
            name="default_growing_season",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        default_months.value = ""

        event_info = arcpy.Parameter(
            displayName="Event Information",
            name="event_info",
            datatype="GPValueTable",
            parameterType="Required",
            direction="Input",
        )
        event_info.columns = [
            ["GPRasterLayer", "Raster"],
            ["GPLong", "Month"],
            ["GPLong", "Return Period"],
        ]

        mc_std = arcpy.Parameter(
            displayName="Uncertainty Std. Dev. (fraction of loss)",
            name="mc_std",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
        )
        mc_std.value = 0.1

        mc_sims = arcpy.Parameter(
            displayName="Monte Carlo Simulations",
            name="mc_sims",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        mc_sims.value = 1000

        seed = arcpy.Parameter(
            displayName="Random Seed",
            name="seed",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )

        return [
            crop,
            depths,
            out_dir,
            crop_csv,
            default_val,
            default_months,
            event_info,
            mc_std,
            mc_sims,
            seed,
        ]

    # ------------------------------------------------------------------
    def updateParameters(self, params):
        """Autofill tables when possible."""
        crop_param, depth_param = params[0], params[1]
        csv_param, default_val, default_months = params[3], params[4], params[5]
        event_table_param = params[6]

        # If crop CSV supplied, disable defaults
        if csv_param.altered:
            default_val.enabled = False
            default_months.enabled = False
        else:
            default_val.enabled = True
            default_months.enabled = True

        # Populate event table from depth rasters if empty
        if (
            depth_param.altered
            and not event_table_param.altered
            and depth_param.valueAsText
        ):

        if depth_param.altered and not event_table_param.altered and depth_param.values:

            vt = arcpy.ValueTable(0)
            for path in depth_param.valueAsText.split(";"):
                vt.addRow([path, "", ""])
            event_table_param.value = vt

        return

    # ------------------------------------------------------------------
    def execute(self, params, messages):  # noqa: C901 - ArcPy style
        crop_raster = params[0].valueAsText
        depth_rasters = params[1].valueAsText.split(";")
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

        # ------------------------------------------------------------------
        # Determine dominant crop codes
        base_crop_arr = arcpy.RasterToNumPyArray(crop_raster)
        counts = Counter(base_crop_arr.flatten())
        counts.pop(0, None)
        top_codes = [c for c, _ in counts.most_common(20)]

        # ------------------------------------------------------------------
        # Build crop table from CSV or defaults
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

        # Filter to top codes
        crop_table = {c: v for c, v in crop_table.items() if c in top_codes}

        # ------------------------------------------------------------------
        # Event info table (month & return period)
        event_table: Dict[str, Dict[str, int]] = {}
        for row in event_info:
            if len(row) < 3:
                continue
            label = os.path.splitext(os.path.basename(str(row[0])))[0]
            event_table[label] = {
                "Month": int(str(row[1])),
                "RP": int(str(row[2])),
            label = os.path.splitext(os.path.basename(row[0]))[0]
            event_table[label] = {
                "Month": int(row[1]),
                "RP": int(row[2]),
            }

        all_summaries: Dict[str, pd.DataFrame] = {}

        # ------------------------------------------------------------------
        # Process each depth raster
        for depth in depth_rasters:
            label = os.path.splitext(os.path.basename(depth))[0]
            ref_ras = arcpy.Raster(depth)
            arcpy.env.snapRaster = ref_ras
            arcpy.env.extent = ref_ras.extent
            proj_crop = os.path.join(out_dir, f"crop_proj_{label}.tif")
            arcpy.management.ProjectRaster(
                crop_raster,
                proj_crop,
                ref_ras.spatialReference,
                "NEAREST",
                ref_ras.meanCellWidth,
            )
            crop_arr = arcpy.RasterToNumPyArray(proj_crop)
            depth_arr = np.maximum(arcpy.RasterToNumPyArray(ref_ras), 0)
            # Align arrays
            rows, cols = min(crop_arr.shape[0], depth_arr.shape[0]), min(
                crop_arr.shape[1], depth_arr.shape[1]
            )
            crop_arr = crop_arr[:rows, :cols]
            depth_arr = depth_arr[:rows, :cols]

            damage = np.zeros_like(depth_arr, dtype=np.float32)
            event_month = event_table[label]["Month"]
            for code, info in crop_table.items():
                mask = crop_arr == code
                if not np.any(mask):
                    continue
                if event_month not in info["GrowingSeason"]:
                    continue
                # Simple depth-damage curve: piecewise linear
                depth_vals = [0.0, 0.01, 6.0]
                damage_vals = [0.0, 0.9, 1.0]
                damage[mask] = np.interp(depth_arr[mask], depth_vals, damage_vals)

            # Save raster with two bands: crop code and damage
            ll = arcpy.Point(ref_ras.extent.XMin, ref_ras.extent.YMin)
            out_stack = np.stack([crop_arr, damage])
            arcpy.NumPyArrayToRaster(
                out_stack,
                ll,
                ref_ras.meanCellWidth,
                ref_ras.meanCellHeight,
                0,
            ).save(os.path.join(out_dir, f"damage_{label}.tif"))

            pixel_area = ref_ras.meanCellWidth * ref_ras.meanCellHeight
            summary_rows = []
            for code in crop_table:
                mask = crop_arr == code
                if not np.any(mask):
                    continue
                acres = np.sum(mask) * pixel_area * 0.000247105
                avg_damage = float(np.mean(damage[mask]))
                value = crop_table[code]["Value"]
                loss = avg_damage * acres * value
                summary_rows.append(
                    {
                        "Flood": label,
                        "CropCode": code,
                        "Acres": acres,
                        "AvgDamage": avg_damage,
                        "Loss": loss,
                    }
                )
            df_sum = pd.DataFrame(summary_rows)
            df_sum.to_csv(os.path.join(out_dir, f"summary_{label}.csv"), index=False)
            all_summaries[label] = df_sum

        # ------------------------------------------------------------------
        # Monte Carlo uncertainty
        mc_rows = []
        for label, df in all_summaries.items():
            rp = event_table[label]["RP"]
            for _, row in df.iterrows():
                base_loss = row["Loss"]
                sims = np.random.normal(base_loss, base_loss * mc_std, mc_sims)
                sims = np.clip(sims, 0, None)
                for i, loss in enumerate(sims, 1):
                    mc_rows.append(
                        {
                            "Flood": label,
                            "CropCode": int(row["CropCode"]),
                            "RP": rp,
                            "Sim": i,
                            "Loss": float(loss),
                        }
                    )
        mc_df = pd.DataFrame(mc_rows)
        mc_path = os.path.join(out_dir, "monte_carlo_results.csv")
        mc_df.to_csv(mc_path, index=False)

        # ------------------------------------------------------------------
        # Expected Annual Damage (USACE trapezoidal rule)
        ead_rows = []
        g = mc_df.groupby(["Sim", "CropCode"])
        for (sim, code), grp in g:
            grp = grp.sort_values("RP")
            probs = 1.0 / grp["RP"].to_numpy()
            losses = grp["Loss"].to_numpy()
            # Add endpoints: P=1 with 0 damage, P=0 with 0 damage
            probs = np.concatenate([[1.0], probs, [0.0]])
            losses = np.concatenate([[0.0], losses, [0.0]])
            ead = np.sum((probs[:-1] - probs[1:]) * (losses[:-1] + losses[1:]) / 2.0)
            ead_rows.append({"Sim": sim, "CropCode": code, "EAD": ead})
        ead_df = pd.DataFrame(ead_rows)
        ead_summary = (
            ead_df.groupby("CropCode")["EAD"].agg([
                ("Mean", "mean"),
                ("P05", lambda x: np.percentile(x, 5)),
                ("P95", lambda x: np.percentile(x, 95)),
            ]).reset_index()
        )
        ead_path = os.path.join(out_dir, "ead_summary.csv")
        ead_summary.to_csv(ead_path, index=False)

        messages.addMessage(f"Monte Carlo results: {mc_path}")
        messages.addMessage(f"EAD summary: {ead_path}")

