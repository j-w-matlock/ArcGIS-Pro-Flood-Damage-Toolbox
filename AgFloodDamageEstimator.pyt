import arcpy
import os
import pandas as pd
import numpy as np

class Toolbox(object):
    def __init__(self):
        self.label = "Ag Flood Damage"
        self.alias = "AgFloodDamage"
        self.tools = [AgFloodDamageEstimator]

class AgFloodDamageEstimator(object):
    def __init__(self):
        self.label = "Estimate Agricultural Flood Damage"
        self.description = "Estimate flood damages using cropland and flood depth rasters."
        self.canRunInBackground = False

    def getParameterInfo(self):
        crop = arcpy.Parameter("Cropland Raster", "crop_raster", "Raster Layer", "Required", "Input")
        depths = arcpy.Parameter("Flood Depth Rasters", "depth_rasters", "Raster Layer", "Required", "Input")
        depths.multiValue = True
        out_dir = arcpy.Parameter("Output Folder", "output_folder", "DEFolder", "Required", "Input")

        default_val = arcpy.Parameter("Default Crop Value per Acre", "default_crop_value", "GPDouble", "Optional", "Input")
        default_val.value = 1200

        default_months = arcpy.Parameter("Default Growing Season (comma separated months; blank = year-round, mismatches warn)", "default_growing_season", "GPString", "Optional", "Input")
        default_months.value = "6"

        damage_curve = arcpy.Parameter("Depth-Damage Curve (depth:fraction, comma separated)", "damage_curve", "GPString", "Required", "Input")
        damage_curve.value = "0:0,1:1"

        event_info = arcpy.Parameter("Event Information", "event_info", "GPValueTable", "Required", "Input")
        event_info.columns = [["Raster Layer", "Raster"], ["GPLong", "Month"], ["GPLong", "Return Period"]]

        mc_std = arcpy.Parameter("Uncertainty Std. Dev. (fraction of loss)", "mc_std", "GPDouble", "Optional", "Input")
        mc_std.value = 0.1

        mc_sims = arcpy.Parameter("Monte Carlo Simulations", "mc_sims", "GPLong", "Optional", "Input")
        mc_sims.value = 1000

        seed = arcpy.Parameter("Random Seed", "seed", "GPLong", "Optional", "Input")
        seed.value = 12

        return [crop, depths, out_dir, default_val, default_months, damage_curve, event_info, mc_std, mc_sims, seed]

    def execute(self, params, messages):
        crop_raster = params[0].valueAsText
        depth_rasters = [str(p).strip("'\"") for p in params[1].values]
        out_dir = params[2].valueAsText
        default_val = float(params[3].value)
        default_months = str(params[4].value)
        curve_str = str(params[5].value)
        event_info = params[6].values
        mc_std = float(params[7].value)
        mc_sims = int(params[8].value)
        seed = params[9].value

        if seed:
            np.random.seed(int(seed))

        def parse_curve(curve_str):
            pts = [s.split(":") for s in curve_str.split(",")]
            pts = sorted((float(d), float(f)) for d, f in pts)
            return pts

        def interp_curve(depths, curve_pts):
            xs, ys = zip(*curve_pts)
            return np.interp(depths, xs, ys, left=0, right=1)

        def parse_months(s):
            return [int(m.strip()) for m in s.split(",") if m.strip()]

        crop_arr = arcpy.RasterToNumPyArray(crop_raster)
        codes = np.unique(crop_arr).astype(int)
        codes = codes[codes != 0]

        crop_table = {code: {"Value": default_val, "GrowingSeason": parse_months(default_months)} for code in codes}
        val_map = np.zeros(max(codes) + 1)
        for code in codes:
            val_map[code] = crop_table[code]["Value"]

        crop_ras = arcpy.Raster(crop_raster)
        pixel_acres = crop_ras.meanCellWidth * crop_ras.meanCellHeight / 4046.86
        ll = arcpy.Point(crop_ras.extent.XMin, crop_ras.extent.YMin)
        ncols, nrows = crop_ras.width, crop_ras.height

        damage_curve_pts = parse_curve(curve_str)
        results = []

        for row in event_info:
            raster_path = row[0].valueAsText if hasattr(row[0], "valueAsText") else str(row[0])
            label = os.path.splitext(os.path.basename(raster_path))[0]
            month = int(row[1])
            rp = float(row[2])

            depth_arr = arcpy.RasterToNumPyArray(
                raster_path, ll, ncols, nrows, nodata_to_value=0
            )
            if depth_arr.shape != crop_arr.shape:
                if depth_arr.T.shape == crop_arr.shape:
                    depth_arr = depth_arr.T
                else:
                    raise ValueError(
                        f"Raster {raster_path} could not be aligned with crop raster"
                    )
            frac = interp_curve(depth_arr, damage_curve_pts)

            for c in codes:
                season = crop_table[c].get("GrowingSeason", [])
                if season and month not in season:
                    messages.addWarningMessage(
                        f"Event month {month} not in growing season for crop code {c}; assuming year-round."
                    )
            grow_mask = np.isin(crop_arr, codes)
            frac = np.where(grow_mask, frac, 0)
            val_arr = val_map[crop_arr]
            damage_arr = frac * val_arr * pixel_acres

            rows = []
            for code in codes:
                mask = crop_arr == code
                if not mask.any():
                    continue
                area_ac = float(mask.sum() * pixel_acres)
                mean_frac = float(frac[mask].mean())
                dmg = float(damage_arr[mask].sum())
                rows.append({"CropCode": code, "AcresFlooded": area_ac, "MeanFractionalDamage": mean_frac, "TotalDamage": dmg})

            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(out_dir, f"{label}_crop_summary.csv"), index=False)

            sim_damages = []
            for _ in range(mc_sims):
                factor = np.clip(np.random.normal(1.0, mc_std), 0, None)
                sim_damages.append((damage_arr * factor).sum())
            sim_damages = np.array(sim_damages)

            results.append({"Label": label, "Month": month, "RP": rp, "Damage": sim_damages.mean(), "StdDev": sim_damages.std()})
            messages.addMessage(f"{label}: mean damage {sim_damages.mean():,.0f}, std {sim_damages.std():,.0f}")

        df_events = pd.DataFrame(results).sort_values("RP")
        df_events.to_csv(os.path.join(out_dir, "event_damages.csv"), index=False)

        probs = 1 / df_events["RP"]
        damages = df_events["Damage"]
        ead = float(((damages.shift() + damages) / 2 * (probs.shift() - probs)).sum())
        with open(os.path.join(out_dir, "ead.csv"), "w") as f:
            f.write(f"EAD,{ead}\n")
        messages.addMessage(f"Expected Annual Damage: {ead:,.0f}")
