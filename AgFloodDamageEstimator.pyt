import arcpy
import os
import numpy as np
import pandas as pd
from openpyxl.chart import BarChart, Reference

from random import Random


# Crop code definitions with default value per acre
CROP_DEFINITIONS = {
    1: ("Corn", 779.96),
    2: ("Cotton", 565.14),
    3: ("Rice", 1193.19),
    4: ("Sorghum", 260.89),
    5: ("Soybeans", 512.07),
    6: ("Sunflower", 414.92),
    10: ("Peanuts", 946.34),
    11: ("Tobacco", 4819.62),
    12: ("Sweet Corn", 6516),
    13: ("Pop or Orn Corn", 0),
    14: ("Mint", 2152.7),
    21: ("Barley", 506.22),
    22: ("Durum Wheat", 279.03),
    23: ("Spring Wheat", 299.25),
    24: ("Winter Wheat", 281.77),
    25: ("Other Small Grains", 0),
    26: ("Dbl Crop WinWht/Soybeans", 512.07),
    27: ("Rye", 212.28),
    28: ("Oats", 260.1),
    29: ("Millet", 118.77),
    30: ("Speltz", 0),
    31: ("Canola", 356.8),
    32: ("Flaxseed", 238.38),
    33: ("Safflower", 291.6),
    34: ("Rape Seed", 356.8),
    35: ("Mustard", 259.65),
    36: ("Alfalfa", 606.98),
    37: ("Other Hay/Non Alfalfa", 305.14),
    38: ("Camelina", 0),
    39: ("Buckwheat", 0),
    41: ("Sugarbeets", 2476.5),
    42: ("Dry Beans", 790.78),
    43: ("Potatoes", 5493.4),
    44: ("Other Crops", 0),
    45: ("Sugarcane", 2398.88),
    46: ("Sweet Potatoes", 4162.5),
    47: ("Misc Vegs & Fruits", 0),
    48: ("Watermelons", 6368.4),
    49: ("Onions", 14923.98),
    50: ("Cucumbers", 2690),
    51: ("Chick Peas", 367.22),
    52: ("Lentils", 356.71),
    53: ("Peas", 241.4),
    54: ("Tomatoes", 9149.55),
    55: ("Caneberries", 0),
    56: ("Hops", 9953.28),
    57: ("Herbs", 0),
    58: ("Clover/Wildflowers", 0),
    59: ("Sod/Grass Seed", 0),
    60: ("Switchgrass", 0),
    61: ("Fallow/Idle Cropland", 0),
    62: ("Pasture/Grass", 0),
    63: ("Forest", 0),
    64: ("Shrubland", 0),
    65: ("Barren", 0),
    66: ("Cherries", 8716.8),
    67: ("Peaches", 12781.3),
    68: ("Apples", 9991.2),
    69: ("Grapes", 6727.5),
    70: ("Christmas Trees", 0),
    71: ("Other Tree Crops", 0),
    72: ("Citrus", 7961.01),
    74: ("Pecans", 1060.23),
    75: ("Almonds", 4237.2),
    76: ("Walnuts", 2803.6),
    77: ("Pears", 7387.5),
    81: ("Clouds/No Data", 0),
    82: ("Developed", 0),
    83: ("Water", 0),
    87: ("Wetlands", 0),
    88: ("Nonag/Undefined", 0),
    92: ("Aquaculture", 0),
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
    204: ("Pistachios", 4185),
    205: ("Triticale", 0),
    206: ("Carrots", 27575.52),
    207: ("Asparagus", 3750),
    208: ("Garlic", 8501.35),
    209: ("Cantaloupes", 7400.28),
    210: ("Prunes", 1546.63),
    211: ("Olives", 3375.54),
    212: ("Oranges", 1092.52),
    213: ("Honeydew Melons", 5625),
    214: ("Broccoli", 10611.66),
    215: ("Avocados", 8750),
    216: ("Peppers", 7305.12),
    217: ("Pomegranates", 6825),
    218: ("Nectarines", 13741),
    219: ("Greens", 7400),
    220: ("Plums", 16547.4),
    221: ("Strawberries", 66000),
    222: ("Squash", 5367.96),
    223: ("Apricots", 1200),
    224: ("Vetch", 0),
    225: ("Dbl Crop WinWht/Corn", 530.87),
    226: ("Dbl Crop Oats/Corn", 520.03),
    227: ("Lettuce", 15244),
    228: ("Dbl Crop Triticale/Corn", 779.96),
    229: ("Pumpkins", 3977.86),
    230: ("Dbl Crop Lettuce/Durum Wht", 7761.52),
    231: ("Dbl Crop Lettuce/Cantaloupe", 11322.14),
    232: ("Dbl Crop Lettuce/Cotton", 8011.98),
    233: ("Dbl Crop Lettuce/Barley", 7904.57),
    234: ("Dbl Crop Durum Wht/Sorghum", 271.33),
    235: ("Dbl Crop Barley/Sorghum", 383.555),
    236: ("Dbl Crop WinWht/Sorghum", 271.33),
    237: ("Dbl Crop Barley/Corn", 643.09),
    238: ("Dbl Crop WinWht/Cotton", 423.455),
    239: ("Dbl Crop Soybeans/Cotton", 538.605),
    240: ("Dbl Crop Soybeans/Oats", 386.085),
    241: ("Dbl Crop Corn/Soybeans", 646.015),
    242: ("Blueberries", 9600),
    243: ("Cabbage", 10329.5),
    244: ("Cauliflower", 14515.72),
    245: ("Celery", 13513.44),
    246: ("Radishes", 3080),
    247: ("Turnips", 3000),
    248: ("Eggplants", 13520),
    249: ("Gourds", 0),
    250: ("Cranberries", 8360),
    254: ("Dbl Crop Barley/Soybeans", 509.145),
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
            # Sanitize the label derived from the depth raster so it can be used
            # safely as part of an in-memory dataset name.  The in-memory
            # workspace does not allow dataset names with extensions, and any
            # periods in the base filename are interpreted as extensions.  This
            # replaces spaces and periods with underscores to ensure a valid
            # name such as "event_1_crop".
            label = (
                os.path.splitext(os.path.basename(depth_str))[0]
                .replace(" ", "_")
                .replace(".", "_")
            )
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
            # Check growing season
            try:
                month = int(month)
            except (TypeError, ValueError):
                month = None
            if season_months and month and month not in season_months:
                messages.addWarningMessage(
                    f"Event month {month} outside growing season; treated as year-round"
                )
            label = _safe(raster)
            event_table[label] = {"Path": raster, "Month": month, "RP": rp}

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
