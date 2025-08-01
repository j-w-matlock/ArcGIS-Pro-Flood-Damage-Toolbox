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
        crop = arcpy.Parameter(
            displayName="Cropland Raster",
            name="crop_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
        )

        depth = arcpy.Parameter(
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

        crop_info = arcpy.Parameter(
            displayName="Crop Information",
            name="crop_info",
            datatype="GPValueTable",
            parameterType="Required",
            direction="Input",
        )
        crop_info.columns = [
            ["GPLong", "Crop Code"],
            ["GPDouble", "Value Per Acre"],
            ["GPString", "Growing Season Months"],
        ]

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

        return [crop, depth, out_dir, crop_info, event_info]

    def updateParameters(self, params):
        crop_param = params[0]
        depth_param = params[1]
        crop_table_param = params[3]
        event_table_param = params[4]

        if crop_param.altered and not crop_table_param.altered and crop_param.valueAsText:
            arr = arcpy.RasterToNumPyArray(crop_param.valueAsText)
            counts = Counter(arr.flatten())
            counts.pop(0, None)
            top = [c for c, _ in counts.most_common(20)]
            vt = arcpy.ValueTable(0)
            for code in top:
                vt.addRow([code, "", ""])
            crop_table_param.value = vt

        if depth_param.altered and not event_table_param.altered and depth_param.values:
            vt = arcpy.ValueTable(0)
            for v in depth_param.values:
                vt.addRow([v, "", ""])
            event_table_param.value = vt

        return

    def isLicensed(self):
        return True

    def updateMessages(self, params):
        return

    def execute(self, params, messages):
        crop_raster = params[0].valueAsText
        depth_rasters = [v.valueAsText for v in params[1].values]
        out_dir = params[2].valueAsText
        crop_info = params[3].values
        event_info = params[4].values

        os.makedirs(out_dir, exist_ok=True)
        crop_arr = arcpy.RasterToNumPyArray(crop_raster)
        counts = Counter(crop_arr.flatten())
        counts.pop(0, None)
        top_crop_codes = [code for code, _ in counts.most_common(20)]

        crop_table = {}
        for row in crop_info:
            if len(row) < 3:
                continue
            try:
                code = int(row[0])
                value = float(row[1])
                months = [int(m.strip()) for m in str(row[2]).split(',') if m.strip()]
            except (ValueError, TypeError):
                continue
            if code not in top_crop_codes or not months:
                continue
            crop_table[code] = {"Value": value, "GrowingSeason": months}

        event_table = {}
        for row in event_info:
            if len(row) < 3:
                continue
            try:
                label = os.path.splitext(os.path.basename(row[0]))[0]
                month = int(row[1])
                rp = int(row[2])
            except (ValueError, TypeError, AttributeError):
                continue
            event_table[label] = {"Month": month, "RP": rp}

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
