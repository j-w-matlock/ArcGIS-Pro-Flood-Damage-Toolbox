import arcpy
import pandas as pd
import numpy as np
import os
import re
from collections import Counter
from scipy.interpolate import interp1d
from openpyxl import load_workbook
from openpyxl.chart import BarChart, Reference
import random
class Toolbox(object):
    def __init__(self):
        self.label = "Ag Flood Damage"
        self.alias = "AgFloodDamage"
        self.tools = [AgFloodDamageEstimator]

class AgFloodDamageEstimator(object):
    def __init__(self):
        self.label = "Estimate Agricultural Flood Damage"
        self.description = "Estimate crop damage from flood depth rasters using a simple vulnerability function."
        self.canRunInBackground = False

    def getParameterInfo(self):
        crop = arcpy.Parameter(0, "crop_raster", "GPRasterLayer", "Input", "Required")
        depth = arcpy.Parameter(1, "depth_rasters", "GPRasterLayer", "Input", "Required")
        depth.multiValue = True
        out_dir = arcpy.Parameter(2, "output_folder", "DEFolder", "Input", "Required")
        crop_csv = arcpy.Parameter(3, "crop_csv", "DEFile", "Input", "Required")
        event_csv = arcpy.Parameter(4, "event_csv", "DEFile", "Input", "Required")
        return [crop, depth, out_dir, crop_csv, event_csv]

    def isLicensed(self):
        return True

    def updateMessages(self, params):
        return

    def execute(self, params, messages):
        import pandas as pd
        import numpy as np
        import os
        from collections import Counter
        from scipy.interpolate import interp1d
        import random
        crop_raster = params[0].valueAsText
        depth_rasters = params[1].values
        out_dir = params[2].valueAsText
        crop_csv = params[3].valueAsText
        event_csv = params[4].valueAsText

        os.makedirs(out_dir, exist_ok=True)
        crop_arr = arcpy.RasterToNumPyArray(crop_raster)
        counts = Counter(crop_arr.flatten())
        counts.pop(0, None)
        top_crop_codes = [code for code, _ in counts.most_common(10)]

        crop_table = {}
        df_crop = pd.read_csv(crop_csv)
        for _, row in df_crop.iterrows():
            code = int(row["CropCode"])
            if code not in top_crop_codes:
                continue
            months = [int(m) for m in str(row["Months"]).split(',')]
            crop_table[code] = {"Value": float(row["Value"]), "GrowingSeason": months}

        event_table = {}
        df_event = pd.read_csv(event_csv)
        for _, row in df_event.iterrows():
            label = os.path.splitext(os.path.basename(row["Raster"]))[0]
            event_table[label] = {"Month": int(row["Month"]), "RP": int(row["RP"])}

        all_summaries = {}
        for depth in depth_rasters:
            label = os.path.splitext(os.path.basename(depth))[0]
            label = re.sub(r'[^\w\-_.]', '_', label)
            ref_ras = arcpy.Raster(depth)
            proj_crop = os.path.join(out_dir, f"proj_crop_{label}.tif")
            arcpy.env.snapRaster = ref_ras
            arcpy.env.extent = ref_ras.extent
            arcpy.management.ProjectRaster(crop_raster, proj_crop, ref_ras.spatialReference, "NEAREST", ref_ras.meanCellWidth)
            crop_arr = arcpy.RasterToNumPyArray(proj_crop)
            depth_arr = np.maximum(arcpy.RasterToNumPyArray(ref_ras), 0)
            min_rows = min(crop_arr.shape[0], depth_arr.shape[0])
            min_cols = min(crop_arr.shape[1], depth_arr.shape[1])
            crop_arr, depth_arr = crop_arr[:min_rows, :min_cols], depth_arr[:min_rows, :min_cols]
            damage = np.zeros_like(depth_arr, dtype=np.float32)
            for code in top_crop_codes:
                mask = crop_arr == code
                if not np.any(mask):
                    continue
                months = crop_table.get(code, {}).get("GrowingSeason", [])
                if event_table[label]["Month"] not in months:
                    continue
                f = interp1d([0, 0.01, 6], [0, 0.9, 1.0], bounds_error=False, fill_value=(0, 1))
                damage[mask] = f(depth_arr[mask])
            ll = arcpy.Point(ref_ras.extent.XMin, ref_ras.extent.YMin)
            out_ras = os.path.join(out_dir, f"damage_{label}.tif")
            arcpy.NumPyArrayToRaster(damage, ll, ref_ras.meanCellWidth, ref_ras.meanCellHeight, 0).save(out_ras)
            pixel_area = ref_ras.meanCellWidth * ref_ras.meanCellHeight
            summary = []
            for code in top_crop_codes:
                mask = crop_arr == code
                if not np.any(mask):
                    continue
                acres = np.sum(mask) * pixel_area * 0.000247105
                avg = np.mean(damage[mask])
                cv = crop_table.get(code, {}).get("Value", 0)
                loss = avg * acres * cv
                summary.append({"CropCode": code, "Pixels": int(np.sum(mask)), "Acres": acres, "AvgDamage": avg, "DollarsLost": loss})
            df = pd.DataFrame(summary).query("DollarsLost > 0")
            df.to_csv(os.path.join(out_dir, f"summary_{label}.csv"), index=False)
            all_summaries[label] = df
        mc_rows = []
        for label, df in all_summaries.items():
            for _, row in df.iterrows():
                code, acres, base = row["CropCode"], row["Acres"], row["AvgDamage"]
                cv = crop_table[code]["Value"]
                months = crop_table[code]["GrowingSeason"]
                base_month = event_table[label]["Month"]
                rp = event_table[label]["RP"]
                for s in range(500):
                    month = random.choice(months)
                    in_season = month in months
                    perturbed = np.clip(random.gauss(base, 0.1 * base), 0, 1) if in_season else 0
                    mc_rows.append({"Flood": label, "Crop": code, "Month": month, "RP": rp, "Sim": s+1, "Damage": perturbed, "Loss": perturbed * acres * cv})
        mc_df = pd.DataFrame(mc_rows)
        excel_path = os.path.join(out_dir, "ag_damage_summary.xlsx")
        with pd.ExcelWriter(excel_path) as w:
            for lbl, df in all_summaries.items():
                df.to_excel(w, sheet_name=f"Summary_{lbl[:25]}", index=False)
            mc_df.to_excel(w, sheet_name="MonteCarlo", index=False)
            summary_rows = []
            g = mc_df.groupby(["Flood", "Crop"])
            for (flood, code), grp in g:
                loss = grp["Loss"]
                summary_rows.append({"Flood": flood, "Crop": code, "Mean": loss.mean(), "5%": np.percentile(loss, 5), "95%": np.percentile(loss, 95)})
            pd.DataFrame(summary_rows).to_excel(w, sheet_name="Uncertainty", index=False)
            annual_rows = []
            for (flood, code), grp in g:
                rp = event_table[flood]["RP"]
                freq = 1.0 / rp
                annual_rows.append({"Flood": flood, "Crop": code, "RP": rp, "Mean Loss": grp["Loss"].mean(), "Annualized": freq * grp["Loss"].mean()})
            pd.DataFrame(annual_rows).to_excel(w, sheet_name="Annualized", index=False)
        wb = load_workbook(excel_path)
        ws = wb["Annualized"]
        chart = BarChart()
        chart.title = "Annualized Loss"
        chart.y_axis.title = "$"
        chart.x_axis.title = "Flood"
        data = Reference(ws, min_col=5, min_row=2, max_row=ws.max_row)
        cats = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
        chart.add_data(data, titles_from_data=False)
        chart.set_categories(cats)
        ws.add_chart(chart, "H2")
        wb.save(excel_path)
        messages.addMessage(f"Excel exported: {excel_path}")
