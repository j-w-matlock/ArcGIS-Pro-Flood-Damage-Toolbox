import arcpy
import os
import numpy as np
import pandas as pd
from openpyxl.chart import BarChart, Reference

from random import Random


# Crop code definitions with default value per acre
CROP_DEFINITIONS = {
    1: ("Corn", 1200),
    2: ("Cotton", 1200),
    3: ("Rice", 1200),
    4: ("Sorghum", 1200),
    5: ("Soybeans", 1200),
    6: ("Sunflower", 1200),
    10: ("Peanuts", 1200),
    11: ("Tobacco", 1200),
    12: ("Sweet Corn", 1200),
    13: ("Pop or Orn Corn", 1200),
    14: ("Mint", 1200),
    21: ("Barley", 1200),
    22: ("Durum Wheat", 1200),
    23: ("Spring Wheat", 1200),
    24: ("Winter Wheat", 1200),
    25: ("Other Small Grains", 1200),
    26: ("Dbl Crop WinWht/Soybeans", 1200),
    27: ("Rye", 1200),
    28: ("Oats", 1200),
    29: ("Millet", 1200),
    30: ("Speltz", 1200),
    31: ("Canola", 1200),
    32: ("Flaxseed", 1200),
    33: ("Safflower", 1200),
    34: ("Rape Seed", 1200),
    35: ("Mustard", 1200),
    36: ("Alfalfa", 1200),
    37: ("Other Hay/Non Alfalfa", 1200),
    38: ("Camelina", 1200),
    39: ("Buckwheat", 1200),
    41: ("Sugarbeets", 1200),
    42: ("Dry Beans", 1200),
    43: ("Potatoes", 1200),
    44: ("Other Crops", 1200),
    45: ("Sugarcane", 1200),
    46: ("Sweet Potatoes", 1200),
    47: ("Misc Vegs & Fruits", 1200),
    48: ("Watermelons", 1200),
    49: ("Onions", 1200),
    50: ("Cucumbers", 1200),
    51: ("Chick Peas", 1200),
    52: ("Lentils", 1200),
    53: ("Peas", 1200),
    54: ("Tomatoes", 1200),
    55: ("Caneberries", 1200),
    56: ("Hops", 1200),
    57: ("Herbs", 1200),
    58: ("Clover/Wildflowers", 1200),
    59: ("Sod/Grass Seed", 1200),
    60: ("Switchgrass", 1200),
    61: ("Fallow/Idle Cropland", 0),
    62: ("Pasture/Grass", 1200),
    63: ("Forest", 0),
    64: ("Shrubland", 0),
    65: ("Barren", 0),
    66: ("Cherries", 1200),
    67: ("Peaches", 1200),
    68: ("Apples", 1200),
    69: ("Grapes", 1200),
    70: ("Christmas Trees", 1200),
    71: ("Other Tree Crops", 1200),
    72: ("Citrus", 1200),
    74: ("Pecans", 1200),
    75: ("Almonds", 1200),
    76: ("Walnuts", 1200),
    77: ("Pears", 1200),
    81: ("Clouds/No Data", 0),
    82: ("Developed", 0),
    83: ("Water", 0),
    87: ("Wetlands", 0),
    88: ("Nonag/Undefined", 0),
    92: ("Aquaculture", 1200),
    111: ("Open Water", 0),
    112: ("Perennial Ice/Snow", 0),
    121: ("Developed/Open Space", 0),
    122: ("Developed/Low Intensity", 0),
    123: ("Developed/Med Intensity", 0),
    124: ("Developed/High Intensity", 0),
    131: ("Barren", 0),
    141: ("Deciduous Forest", 0),
    142: ("Evergreen Forest", 0),
    143: ("Mixed Forest", 0),
    152: ("Shrubland", 0),
    176: ("Grassland/Pasture", 0),
    190: ("Woody Wetlands", 0),
    195: ("Herbaceous Wetlands", 0),
    204: ("Pistachios", 1200),
    205: ("Triticale", 1200),
    206: ("Carrots", 1200),
    207: ("Asparagus", 1200),
    208: ("Garlic", 1200),
    209: ("Cantaloupes", 1200),
    210: ("Prunes", 1200),
    211: ("Olives", 1200),
    212: ("Oranges", 1200),
    213: ("Honeydew Melons", 1200),
    214: ("Broccoli", 1200),
    215: ("Avocados", 1200),
    216: ("Peppers", 1200),
    217: ("Pomegranates", 1200),
    218: ("Nectarines", 1200),
    219: ("Greens", 1200),
    220: ("Plums", 1200),
    221: ("Strawberries", 1200),
    222: ("Squash", 1200),
    223: ("Apricots", 1200),
    224: ("Vetch", 1200),
    225: ("Dbl Crop WinWht/Corn", 1200),
    226: ("Dbl Crop Oats/Corn", 1200),
    227: ("Lettuce", 1200),
    228: ("Dbl Crop Triticale/Corn", 1200),
    229: ("Pumpkins", 1200),
    230: ("Dbl Crop Lettuce/Durum Wht", 1200),
    231: ("Dbl Crop Lettuce/Cantaloupe", 1200),
    232: ("Dbl Crop Lettuce/Cotton", 1200),
    233: ("Dbl Crop Lettuce/Barley", 1200),
    234: ("Dbl Crop Durum Wht/Sorghum", 1200),
    235: ("Dbl Crop Barley/Sorghum", 1200),
    236: ("Dbl Crop WinWht/Sorghum", 1200),
    237: ("Dbl Crop Barley/Corn", 1200),
    238: ("Dbl Crop WinWht/Cotton", 1200),
    239: ("Dbl Crop Soybeans/Cotton", 1200),
    240: ("Dbl Crop Soybeans/Oats", 1200),
    241: ("Dbl Crop Corn/Soybeans", 1200),
    242: ("Blueberries", 1200),
    243: ("Cabbage", 1200),
    244: ("Cauliflower", 1200),
    245: ("Celery", 1200),
    246: ("Radishes", 1200),
    247: ("Turnips", 1200),
    248: ("Eggplants", 1200),
    249: ("Gourds", 1200),
    250: ("Cranberries", 1200),
    254: ("Dbl Crop Barley/Soybeans", 1200),
}

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


def intersect_extent(ext1, ext2):
    """Return overlap of two extents or None if disjoint."""
    xmin = max(ext1.XMin, ext2.XMin)
    ymin = max(ext1.YMin, ext2.YMin)
    xmax = min(ext1.XMax, ext2.XMax)
    ymax = min(ext1.YMax, ext2.YMax)
    if xmin >= xmax or ymin >= ymax:
        return None
    return xmin, ymin, xmax, ymax

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
        pts = arcpy.Parameter(displayName="Output Damage Points", name="damage_points", datatype="DEFeatureClass",
                               parameterType="Optional", direction="Output")
        return [crop, out, val, season, curve, event_info, stddev, mc, seed, pts]

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
        out_points = params[9].valueAsText if params[9].value else None

        os.makedirs(out_dir, exist_ok=True)

        damage_curve_pts = parse_curve(curve_str)
        curve_depths = np.array([d for d, _ in damage_curve_pts], dtype=float)
        curve_fracs = np.array([f for _, f in damage_curve_pts], dtype=float)
        crop_ras = arcpy.Raster(crop_path)

        # Cell area in acres
        desc = arcpy.Describe(crop_ras)
        meters_per_unit = desc.spatialReference.metersPerUnit
        cell_area_acres = (
            desc.meanCellWidth * desc.meanCellHeight * meters_per_unit ** 2 / 4046.8564224
        )

        # Growing season months as set of ints or None for year-round
        season_months = parse_months(season_str)

        results = []
        points = []

        for row in event_table:
            depth_path, month, rp = row
            depth_str = (
                depth_path.valueAsText
                if hasattr(depth_path, "valueAsText")
                else depth_path.dataSource
                if hasattr(depth_path, "dataSource")
                else str(depth_path)
            )
            if not depth_str:
                continue
            depth_desc = arcpy.Describe(depth_str)
            if crop_ras.extent.disjoint(depth_desc.extent):
                raise ValueError(f"Depth raster {depth_str} does not overlap crop raster extent")
            label = os.path.splitext(os.path.basename(depth_str))[0].replace(" ", "_")
            aligned = os.path.join(out_dir, f"{label}_aligned.tif")
            clipped = os.path.join(out_dir, f"{label}_clipped.tif")

            arcpy.env.snapRaster = crop_ras
            arcpy.env.cellSize = crop_ras

            arcpy.management.ProjectRaster(
                depth_str,
                aligned,
                crop_ras.spatialReference,
                "NEAREST",
                crop_ras.meanCellWidth,
            )

            aligned_desc = arcpy.Describe(aligned)
            inter = intersect_extent(aligned_desc.extent, crop_ras.extent)
            if not inter:
                raise ValueError(f"Depth raster {depth_str} does not overlap crop raster extent")
            extent_str = f"{inter[0]} {inter[1]} {inter[2]} {inter[3]}"

            crop_clip = os.path.join("in_memory", f"{label}_crop")
            arcpy.management.Clip(crop_path, extent_str, crop_clip, "#", "0", "NONE", "MAINTAIN_EXTENT")
            arcpy.management.Clip(aligned, extent_str, clipped, "#", "0", "NONE", "MAINTAIN_EXTENT")
            crop_arr = arcpy.RasterToNumPyArray(crop_clip)
            depth_arr = arcpy.RasterToNumPyArray(clipped)
            arcpy.management.Delete(crop_clip)
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
            crop_masked = crop_arr[mask].astype(int)
            depth_masked = depth_arr[mask]
            crop_codes = np.unique(crop_masked)

            if crop_codes.size == 0:
                continue

            max_code = int(crop_codes.max())
            val_lookup = np.full(max_code + 1, value_acre, dtype=float)
            for code, (_, val) in CROP_DEFINITIONS.items():
                if code <= max_code:
                    val_lookup[code] = val

            pixel_counts_arr = np.bincount(crop_masked, minlength=max_code + 1)
            pixel_counts = {c: int(pixel_counts_arr[c]) for c in crop_codes}

            base_frac = np.interp(
                depth_masked, curve_depths, curve_fracs, left=curve_fracs[0], right=curve_fracs[-1]
            )

            damages_runs = {c: [] for c in crop_codes}
            damage_accum = np.zeros_like(depth_arr, dtype=float) if out_points else None

            for _ in range(runs):
                if stddev > 0:
                    rng = np.random.default_rng(rand.randint(0, 2**32 - 1))
                    frac = base_frac + rng.normal(0, stddev, size=base_frac.shape)
                    frac = np.clip(frac, 0, 1)
                else:
                    frac = base_frac

                dmg_vals = frac * val_lookup[crop_masked] * cell_area_acres
                dmg_per_crop = np.bincount(
                    crop_masked, weights=dmg_vals, minlength=max_code + 1
                )
                for c in crop_codes:
                    damages_runs[c].append(float(dmg_per_crop[c]))

                if out_points:
                    damage_accum[mask] += dmg_vals

            if out_points:
                damage_avg = damage_accum / runs
                xmin, ymin, xmax, ymax = inter
                cw = crop_ras.meanCellWidth
                ch = crop_ras.meanCellHeight
                x0 = xmin + cw / 2
                y0 = ymax - ch / 2
                for i in range(depth_arr.shape[0]):
                    for j in range(depth_arr.shape[1]):
                        if mask[i, j]:
                            crop_code = int(crop_arr[i, j])
                            landcover = CROP_DEFINITIONS.get(crop_code, ("Unknown", value_acre))[0]
                            points.append((
                                x0 + j * cw,
                                y0 - i * ch,
                                crop_code,
                                landcover,
                                float(damage_avg[i, j]),
                                label,
                                float(rp),
                            ))

            for c in crop_codes:
                arr = np.array(damages_runs[c], dtype=float)
                avg_damage = float(arr.mean())
                std_damage = float(arr.std(ddof=0))
                p05 = float(np.percentile(arr, 5))
                p95 = float(np.percentile(arr, 95))
                name, _ = CROP_DEFINITIONS.get(c, ("Unknown", value_acre))
                results.append({
                    "Label": label,
                    "RP": float(rp),
                    "Crop": int(c),
                    "LandCover": name,
                    "Damage": avg_damage,
                    "StdDev": std_damage,
                    "P05": p05,
                    "P95": p95,
                    "FloodedAcres": pixel_counts[c] * cell_area_acres,
                    "FloodedPixels": pixel_counts[c],
                })

        if not results:
            raise ValueError("No valid events provided")

        # --- Trapezoidal EAD ---
        df_events = pd.DataFrame(results)

        def calc_ead(df):
            df = df.sort_values("RP", ascending=False)
            probs = 1 / df["RP"].to_numpy()
            damages = df["Damage"].to_numpy()
            probs = np.concatenate(([0.0], probs, [1.0]))
            damages = np.concatenate(([damages[0]], damages, [0.0]))
            return float(np.trapz(damages, probs))

        # Total EAD
        df_total = df_events.groupby(["Label", "RP"], as_index=False)["Damage"].sum()
        ead_total = calc_ead(df_total)

        # EAD per crop
        eads_crop = {name: calc_ead(g) for name, g in df_events.groupby("LandCover")}

        # Write CSV for overall EAD
        with open(os.path.join(out_dir, "ead.csv"), "w") as f:
            f.write(f"EAD,{ead_total}\n")
        messages.addMessage(f"Expected Annual Damage (Trapezoidal): {ead_total:,.0f}")

        # Export detailed results to Excel with charts
        excel_path = os.path.join(out_dir, "damage_results.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Raw per-event damages
            df_events.to_excel(writer, sheet_name="EventDamages", index=False)

            # Pivot table for charting
            pivot = df_events.pivot_table(index="Label", columns="LandCover", values="Damage", fill_value=0)
            pivot.to_excel(writer, sheet_name="EventPivot")

            # EAD per crop table
            df_ead = pd.DataFrame([{"LandCover": k, "EAD": v} for k, v in eads_crop.items()])
            df_ead.to_excel(writer, sheet_name="EAD", index=False)

            # Chart for event damages
            worksheet_pivot = writer.sheets["EventPivot"]
            max_row = pivot.shape[0] + 1
            max_col = pivot.shape[1] + 1
            data_ref = Reference(worksheet_pivot, min_col=2, min_row=1, max_col=max_col, max_row=max_row)
            cats_ref = Reference(worksheet_pivot, min_col=1, min_row=2, max_row=max_row)
            chart1 = BarChart()
            chart1.type = "col"
            chart1.grouping = "stacked"
            chart1.title = "Damage by Event and Land Cover"
            chart1.x_axis.title = "Event"
            chart1.y_axis.title = "Damage"
            chart1.add_data(data_ref, titles_from_data=True)
            chart1.set_categories(cats_ref)
            worksheet_pivot.add_chart(chart1, "H2")

            # Chart for EAD per crop
            worksheet_ead = writer.sheets["EAD"]
            data_ref2 = Reference(worksheet_ead, min_col=2, min_row=1, max_row=len(df_ead) + 1)
            cats_ref2 = Reference(worksheet_ead, min_col=1, min_row=2, max_row=len(df_ead) + 1)
            chart2 = BarChart()
            chart2.type = "col"
            chart2.title = "Expected Annual Damage by Land Cover"
            chart2.x_axis.title = "Land Cover"
            chart2.y_axis.title = "EAD"
            chart2.add_data(data_ref2, titles_from_data=True)
            chart2.set_categories(cats_ref2)
            worksheet_ead.add_chart(chart2, "D2")

        if out_points:
            if arcpy.Exists(out_points):
                arcpy.management.Delete(out_points)
            arcpy.management.CreateFeatureclass(os.path.dirname(out_points), os.path.basename(out_points), "POINT",
                                               spatial_reference=crop_ras.spatialReference)
            arcpy.management.AddField(out_points, "Crop", "LONG")
            arcpy.management.AddField(out_points, "LandCover", "TEXT", field_length=50)
            arcpy.management.AddField(out_points, "Damage", "DOUBLE")
            arcpy.management.AddField(out_points, "Event", "TEXT", field_length=50)
            arcpy.management.AddField(out_points, "RP", "DOUBLE")
            with arcpy.da.InsertCursor(out_points, ["SHAPE@XY", "Crop", "LandCover", "Damage", "Event", "RP"]) as cursor:
                for x, y, c, lc, dmg, lbl, rp_val in points:
                    cursor.insertRow([(x, y), c, lc, dmg, lbl, rp_val])
