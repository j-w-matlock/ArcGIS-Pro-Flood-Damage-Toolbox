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
        crop.description = (
            "Raster of cropland classification codes. "
            "Switching rasters changes which crop types are analyzed and therefore"
            " which per-acre values feed into the damage model."
        )
        out = arcpy.Parameter(displayName="Output Folder", name="output_folder", datatype="DEFolder",
                              parameterType="Required", direction="Output")
        out.description = (
            "Folder where result tables, Excel summaries, and optional damage points will be written. "
            "Location only affects where outputs are stored, not the damage calculations themselves."
        )
        val = arcpy.Parameter(displayName="Default Crop Value per Acre (USD/acre)", name="value_acre", datatype="Double",
                              parameterType="Required", direction="Input")
        val.value = 1200
        val.description = (
            "Dollar value applied per acre for crops not found in the predefined list. "
            "Raising this value increases damage estimates for unknown crops, while lowering it reduces them."
        )
        winter = arcpy.Parameter(
            displayName="Winter Growing Months",
            name="winter_months",
            datatype="String",
            parameterType="Optional",
            direction="Input",
        )
        winter.value = "12,1,2"
        winter.description = (
            "Comma separated months between the winter and spring equinox. "
            "Months listed here are considered part of the winter growing season."
        )
        spring = arcpy.Parameter(
            displayName="Spring Growing Months",
            name="spring_months",
            datatype="String",
            parameterType="Optional",
            direction="Input",
        )
        spring.value = "3,4,5"
        spring.description = (
            "Comma separated months between the spring and summer equinox. "
            "Months listed here are considered part of the spring growing season."
        )
        summer = arcpy.Parameter(
            displayName="Summer Growing Months",
            name="summer_months",
            datatype="String",
            parameterType="Optional",
            direction="Input",
        )
        summer.value = "6,7,8"
        summer.description = (
            "Comma separated months between the summer and fall equinox. "
            "Months listed here are considered part of the summer growing season."
        )
        fall = arcpy.Parameter(
            displayName="Fall Growing Months",
            name="fall_months",
            datatype="String",
            parameterType="Optional",
            direction="Input",
        )
        fall.value = "9,10,11"
        fall.description = (
            "Comma separated months between the fall and winter equinox. "
            "Months listed here are considered part of the fall growing season."
        )
        curve = arcpy.Parameter(displayName="Depth-Damage Curve (depth in ft:fraction, comma separated)",
                                name="curve", datatype="String", parameterType="Required", direction="Input")
        curve.value = "0:1,1:1"
        curve.description = (
            "Pairs of flood depth and damage fraction (e.g., '0:0,1:0.5,2:1'). "
            "Editing these points changes how quickly losses climb with depth, directly affecting damage totals."
        )
        specific_curve = arcpy.Parameter(
            displayName="Specific Crop Depth Damage Curve",
            name="specific_curves",
            datatype="Value Table",
            parameterType="Optional",
            direction="Input",
        )
        specific_curve.columns = [
            ["GPLong", "Crop Code"],
            ["GPString", "Depth-Damage Curve"],
            ["GPString", "Grow Months"],
        ]
        specific_curve.description = (
            "Optional crop-specific depth-damage curves. "
            "Provide a crop code, its depth-damage curve, and an optional list of growing season months "
            "formatted like '0:0,1:0.5,2:1' and '6,7,8'. Listed codes override the default curve and months."
        )
        event_info = arcpy.Parameter(displayName="Event Information", name="event_info", datatype="Value Table",
                                     parameterType="Required", direction="Input")
        event_info.columns = [["Raster Layer", "Raster"], ["GPLong", "Month (1-12)"], ["GPLong", "Return Period (years)"]]
        event_info.value = [["", 6, 100]]
        event_info.description = (
            "Table of flood events with depth rasters, flood month, and return period. "
            "Adding or modifying rows changes which scenarios are modeled and their frequency, influencing total expected damage."
        )

        stddev = arcpy.Parameter(
            displayName="Damage Fraction Std. Dev. (0-1)",
            name="uncertainty",
            datatype="Double",
            parameterType="Required",
            direction="Input",
        )
        stddev.value = 0.1
        stddev.description = (
            "Standard deviation applied to the damage fractions during Monte Carlo simulations. "
            "Higher values introduce more variability in outcomes, modeling greater uncertainty in the curve."
        )

        mc = arcpy.Parameter(
            displayName="Monte Carlo Simulations (count)",
            name="mc_runs",
            datatype="Long",
            parameterType="Required",
            direction="Input",
        )
        mc.value = 10
        mc.description = (
            "Number of Monte Carlo iterations for each event per year. "
            "Increasing the count stabilizes averages but lengthens processing time."
        )

        seed = arcpy.Parameter(
            displayName="Random Seed",
            name="random_seed",
            datatype="Long",
            parameterType="Required",
            direction="Input",
        )
        seed.value = 10
        seed.description = (
            "Seed value for the random number generator to ensure reproducible simulations. "
            "Changing it yields different random sequences and hence different simulated damages."
        )

        rand_month = arcpy.Parameter(
            displayName="Randomize Flood Month",
            name="random_month",
            datatype="Boolean",
            parameterType="Optional",
            direction="Input",
        )
        rand_month.value = False
        rand_month.description = (
            "If checked, randomly selects the flood month in simulations instead of using the month provided for each event. "
            "Randomizing months can move events into or out of the growing season, altering damage totals."
        )
        rand_season = arcpy.Parameter(
            displayName="Randomize Flood Season",
            name="random_season",
            datatype="Boolean",
            parameterType="Optional",
            direction="Input",
        )
        rand_season.value = False
        rand_season.description = (
            "If checked, selects a flood season based on provided probabilities and then picks a random month within that season."
        )
        season_prob = arcpy.Parameter(
            displayName="Season Probabilities",
            name="season_probs",
            datatype="Value Table",
            parameterType="Optional",
            direction="Input",
        )
        season_prob.columns = [["GPString", "Season"], ["GPDouble", "Probability"]]
        season_prob.value = [
            ["Winter", 0.25],
            ["Spring", 0.25],
            ["Summer", 0.25],
            ["Fall", 0.25],
        ]
        season_prob.description = (
            "Probability weights for each season when randomizing flood season. Weights will be normalized." 
        )

        depth_sd = arcpy.Parameter(
            displayName="Flood Depth Std. Dev. (ft)",
            name="depth_stddev",
            datatype="Double",
            parameterType="Optional",
            direction="Input",
        )
        depth_sd.value = 0.0
        depth_sd.description = (
            "Standard deviation for adding a normally distributed event-wide offset to flood depths. "
            "Higher values produce greater depth variation, which affects interpolated damage fractions."
        )

        value_sd = arcpy.Parameter(
            displayName="Crop Value Std. Dev. (USD/acre)",
            name="value_stddev",
            datatype="Double",
            parameterType="Optional",
            direction="Input",
        )
        value_sd.value = 0.0
        value_sd.description = (
            "Standard deviation for crop values per acre (additive, USD/acre). "
            "Increasing the deviation widens the range of possible crop values, changing overall damage estimates."
        )

        analysis = arcpy.Parameter(
            displayName="Analysis Period (years)",
            name="analysis_years",
            datatype="Long",
            parameterType="Optional",
            direction="Input",
        )
        analysis.value = 1
        analysis.description = (
            "Number of years to simulate for each event. "
            "Extending the period scales damages across more years of exposure."
        )

        pts = arcpy.Parameter(
            displayName="Output Damage Points",
            name="damage_points",
            datatype="DEFeatureClass",
            parameterType="Optional",
            direction="Output",
        )
        pts.description = (
            "Optional feature class storing per-pixel average damage for visualization. "
            "Creating this output enables spatial analysis but increases processing time; leaving it blank skips this step."
        )

        return [
            crop,
            out,
            val,
            winter,
            spring,
            summer,
            fall,
            curve,
            specific_curve,
            event_info,
            stddev,
            mc,
            seed,
            rand_month,
            rand_season,
            season_prob,
            depth_sd,
            value_sd,
            analysis,
            pts,
        ]

    def execute(self, params, messages):
        crop_path = params[0].valueAsText
        out_dir = params[1].valueAsText
        value_acre = float(params[2].value)
        winter_str = params[3].value
        spring_str = params[4].value
        summer_str = params[5].value
        fall_str = params[6].value
        curve_str = params[7].value
        specific_table = params[8].values if params[8].values else []
        event_table = params[9].values
        frac_std = float(params[10].value)
        runs = int(params[11].value)
        rand = Random(int(params[12].value))
        random_month = bool(params[13].value) if params[13].value is not None else False
        random_season = bool(params[14].value) if params[14].value is not None else False
        season_prob_table = params[15].values if params[15].values else []
        depth_std = float(params[16].value) if params[16].value is not None else 0.0
        value_std = float(params[17].value) if params[17].value is not None else 0.0
        analysis_years = int(params[18].value) if params[18].value is not None else 1
        out_points = params[19].valueAsText if params[19].value else None

        os.makedirs(out_dir, exist_ok=True)

        damage_curve_pts = parse_curve(curve_str)
        curve_depths = np.array([d for d, _ in damage_curve_pts], dtype=float)
        curve_fracs = np.array([f for _, f in damage_curve_pts], dtype=float)
        specific_curves = {}
        for row in specific_table:
            if len(row) < 2:
                continue
            code = row[0]
            curve_txt = row[1]
            grow_txt = row[2] if len(row) > 2 else None
            if curve_txt:
                pts = parse_curve(curve_txt)
                months = parse_months(grow_txt)
                specific_curves[int(code)] = (
                    np.array([d for d, _ in pts], dtype=float),
                    np.array([f for _, f in pts], dtype=float),
                    months,
                )
        crop_desc = arcpy.Describe(crop_path)
        crop_sr = getattr(crop_desc, "spatialReference", None)
        if not crop_sr or crop_sr.name in (None, ""):
            raise ValueError("Crop raster must have a defined spatial reference.")

        # Growing season months for each season
        winter_months = parse_months(winter_str)
        spring_months = parse_months(spring_str)
        summer_months = parse_months(summer_str)
        fall_months = parse_months(fall_str)
        season_lookup = {
            "Winter": winter_months,
            "Spring": spring_months,
            "Summer": summer_months,
            "Fall": fall_months,
        }
        season_months = set()
        for s in season_lookup.values():
            if s:
                season_months.update(s)
        if not season_months:
            season_months = None
        season_prob_dict = {}
        for s, p in season_prob_table:
            season_prob_dict[str(s)] = float(p)
        if not season_prob_dict:
            season_prob_dict = {"Winter": 0.25, "Spring": 0.25, "Summer": 0.25, "Fall": 0.25}
        season_names = list(season_lookup.keys())
        weights = [season_prob_dict.get(s, 1.0) for s in season_names]
        total_w = sum(weights)
        if total_w <= 0:
            weights = [1.0] * len(season_names)
        else:
            weights = [w / total_w for w in weights]

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
            depth_sr = getattr(depth_desc, "spatialReference", None)
            if not depth_sr or getattr(depth_sr, "type", "") != "Projected":
                raise ValueError(
                    f"Depth raster {depth_str} must use a projected coordinate system with linear units"
                    " in meters or feet so acreage can be computed reliably."
                )
            meters_per_unit = depth_sr.metersPerUnit
            if meters_per_unit in (None, 0):
                raise ValueError(
                    f"Depth raster {depth_str} spatial reference does not provide a valid metersPerUnit conversion."
                )
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
            crop_proj = os.path.join("in_memory", f"{label}_crop_proj")
            crop_clip = os.path.join("in_memory", f"{label}_crop")
            depth_clip = os.path.join("in_memory", f"{label}_depth")

            crop_arr = None
            depth_arr = None
            cell_area_acres = None
            clip_width = None
            clip_height = None
            inter = None

            with arcpy.EnvManager(snapRaster=depth_str, cellSize=depth_str):
                arcpy.management.ProjectRaster(
                    crop_path,
                    crop_proj,
                    depth_sr,
                    "NEAREST",
                    depth_desc.meanCellWidth,
                )

                crop_proj_desc = arcpy.Describe(crop_proj)
                inter = intersect_extent(crop_proj_desc.extent, depth_desc.extent)
            if not inter:
                raise ValueError(f"Depth raster {depth_str} does not overlap crop raster extent")
            extent_str = f"{inter[0]} {inter[1]} {inter[2]} {inter[3]}"

            with arcpy.EnvManager(snapRaster=depth_str, cellSize=depth_str):
                arcpy.management.Clip(crop_proj, extent_str, crop_clip, "#", "0", "NONE", "MAINTAIN_EXTENT")
                arcpy.management.Clip(depth_str, extent_str, depth_clip, "#", "0", "NONE", "MAINTAIN_EXTENT")
                crop_arr = arcpy.RasterToNumPyArray(crop_clip)
                depth_arr = arcpy.RasterToNumPyArray(depth_clip)
                depth_clip_desc = arcpy.Describe(depth_clip)
                clip_width = abs(depth_clip_desc.meanCellWidth)
                clip_height = abs(depth_clip_desc.meanCellHeight)
                cell_area_acres = (
                    clip_width * clip_height * meters_per_unit ** 2 / 4046.8564224
                )

            for ds in (crop_proj, crop_clip, depth_clip):
                if arcpy.Exists(ds):
                    arcpy.management.Delete(ds)
            if cell_area_acres is None:
                raise RuntimeError(
                    f"Failed to compute cell area for depth raster {depth_str}."
                )
            if depth_arr.shape != crop_arr.shape:
                raise ValueError(
                    f"Masked depth raster {label} shape does not match crop raster. "
                    f"Crop: {crop_arr.shape}, Depth: {depth_arr.shape}"
                )

            mask = (crop_arr > 0) & (depth_arr > 0)
            crop_masked = crop_arr[mask].astype(int)
            depth_masked = depth_arr[mask]
            crop_codes = np.unique(crop_masked)

            if crop_codes.size == 0:
                continue

            spec_masks = {}
            for code, (d_arr, f_arr, _) in specific_curves.items():
                m = crop_masked == code
                if np.any(m):
                    spec_masks[code] = m

            max_code = int(crop_codes.max())
            val_lookup = np.full(max_code + 1, value_acre, dtype=float)
            for code, (_, val) in CROP_DEFINITIONS.items():
                if code <= max_code:
                    val_lookup[code] = val

            pixel_counts_arr = np.bincount(crop_masked, minlength=max_code + 1)
            pixel_counts = {c: int(pixel_counts_arr[c]) for c in crop_codes}

            damages_runs = {c: [] for c in crop_codes}
            damage_accum = np.zeros_like(depth_arr, dtype=float) if out_points else None

            # Pre-compute boolean masks for each crop code once so we do not
            # repeatedly allocate large temporary arrays inside the Monte Carlo
            # loop.  The previous approach recreated "crop_masked == c" for
            # every crop on every iteration which becomes extremely slow for
            # large rasters.  Caching the masks ensures that each simulation
            # run only toggles already computed masks which dramatically
            # shortens processing time.
            crop_code_masks = {c: (crop_masked == c) for c in crop_codes}

            total_runs = runs * analysis_years
            for _ in range(total_runs):
                if random_season:
                    sel_season = rand.choices(season_names, weights=weights)[0]
                    months_sel = season_lookup.get(sel_season)
                    if months_sel:
                        sim_month = rand.choice(list(months_sel))
                    else:
                        sim_month = rand.randint(1, 12)
                elif random_month:
                    sim_month = rand.randint(1, 12)
                else:
                    sim_month = month

                active_mask = np.zeros_like(crop_masked, dtype=bool)
                for c, base_mask in crop_code_masks.items():
                    _, _, gm = specific_curves.get(c, (None, None, season_months))
                    gm = gm if gm is not None else season_months
                    if gm is None or sim_month in gm:
                        active_mask |= base_mask
                if not active_mask.any():
                    for c in crop_codes:
                        damages_runs[c].append(0.0)
                    # When no crops are active in the simulated month the
                    # remaining Monte Carlo work would only propagate zeros.
                    # Skip immediately to the next iteration so runs that fall
                    # completely outside the growing season do not waste time
                    # on interpolation or random draws.  This restores the
                    # short-circuit behaviour that prevents apparent hangs when
                    # large event tables contain many out-of-season months.
                    continue

                rng = np.random.default_rng(rand.randint(0, 2**32 - 1))
                depth_sim = depth_masked
                if depth_std > 0:
                    depth_offset = rng.normal(0, depth_std)
                    depth_sim = np.clip(depth_masked + depth_offset, 0, None)
                    depth_sim = np.clip(depth_sim, 0, None)

                frac = np.interp(
                    depth_sim,
                    curve_depths,
                    curve_fracs,
                    left=curve_fracs[0],
                    right=curve_fracs[-1],
                )
                for code, m in spec_masks.items():
                    d_arr, f_arr, _ = specific_curves[code]
                    frac[m] = np.interp(
                        depth_sim[m],
                        d_arr,
                        f_arr,
                        left=f_arr[0],
                        right=f_arr[-1],
                    )
                if frac_std > 0:
                    frac = frac + rng.normal(0, frac_std, size=frac.shape)
                    frac = np.clip(frac, 0, 1)

                values = val_lookup[crop_masked]
                if value_std > 0:
                    values = np.clip(values + rng.normal(0, value_std, size=values.shape), 0, None)
                values[~active_mask] = 0

                dmg_vals = frac * values * cell_area_acres
                dmg_per_crop = np.bincount(crop_masked, weights=dmg_vals, minlength=max_code + 1)
                for c in crop_codes:
                    damages_runs[c].append(float(dmg_per_crop[c]))

                if out_points:
                    damage_accum[mask] += dmg_vals

            if out_points:
                damage_avg = damage_accum / total_runs
                xmin, ymin, xmax, ymax = inter
                cw = clip_width
                ch = clip_height
                x0 = xmin + cw / 2
                y0 = ymax - ch / 2

                # Iterate only over cells that were actually part of the mask
                # rather than the entire raster extent.  The nested loops over
                # the full grid previously caused the tool to appear to hang on
                # larger rasters because it still inspected every pixel even
                # when most were empty.  Restricting iteration to the masked
                # cells produces the same output while drastically reducing the
                # amount of work required.
                rows, cols = np.nonzero(mask)
                masked_crops = crop_arr[mask]
                masked_damages = damage_avg[mask]
                for row, col, crop_code, dmg in zip(rows, cols, masked_crops, masked_damages):
                    crop_code = int(crop_code)
                    landcover = CROP_DEFINITIONS.get(crop_code, ("Unknown", value_acre))[0]
                    # Clip text fields to the feature class length to avoid
                    # "insertRow returned NULL" failures when names exceed the
                    # field size.  Long raster names are common when rasters
                    # include full scenario descriptions or timestamps.
                    landcover = str(landcover)[:255]
                    event_label = str(label)[:255]
                    points.append(
                        (
                            x0 + col * cw,
                            y0 - row * ch,
                            crop_code,
                            landcover,
                            float(dmg),
                            event_label,
                            float(rp),
                        )
                    )

            for c in crop_codes:
                arr = np.array(damages_runs[c], dtype=float)
                avg_damage = float(arr.mean())
                std_damage = float(arr.std(ddof=0))
                p05 = float(np.percentile(arr, 5))
                p95 = float(np.percentile(arr, 95))
                name, _ = CROP_DEFINITIONS.get(c, ("Unknown", value_acre))
                results.append(
                    {
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
                        "ValuePerAcre": float(val_lookup[c]),
                    }
                )

        if not results:
            raise ValueError("No valid events provided")

        # --- EAD calculations ---
        df_events = pd.DataFrame(results)

        def calc_ead_discrete(df):
            return float((df["Damage"] / df["RP"]).sum())

        # Total EAD (discrete method)
        df_total = df_events.groupby(["Label", "RP"], as_index=False)["Damage"].sum()
        ead_total = calc_ead_discrete(df_total)

        # EAD per crop (discrete method)
        eads_crop = {name: calc_ead_discrete(g) for name, g in df_events.groupby("LandCover")}

        # Optional trapezoidal EAD when multiple return periods are provided
        ead_total_trap = None
        eads_crop_trap = {}
        if df_total["RP"].nunique() > 1:
            def calc_ead_trap(df):
                df = df.sort_values("RP", ascending=False)
                probs = 1 / df["RP"].to_numpy()
                damages = df["Damage"].to_numpy()
                probs = np.concatenate(([0.0], probs, [1.0]))
                damages = np.concatenate(([damages[0]], damages, [0.0]))
                return float(np.trapz(damages, probs))

            ead_total_trap = calc_ead_trap(df_total)
            eads_crop_trap = {name: calc_ead_trap(g) for name, g in df_events.groupby("LandCover")}

        # Write CSV for overall EAD (discrete)
        with open(os.path.join(out_dir, "ead.csv"), "w") as f:
            f.write(f"EAD,{ead_total}\n")
        messages.addMessage(f"Expected Annual Damage (Discrete): {ead_total:,.0f}")

        # Trapezoidal CSV and message if applicable
        if ead_total_trap is not None:
            with open(os.path.join(out_dir, "ead_trapezoidal.csv"), "w") as f:
                f.write(f"EAD,{ead_total_trap}\n")
            messages.addMessage(f"Expected Annual Damage (Trapezoidal): {ead_total_trap:,.0f}")

        # Export detailed results to Excel with charts
        excel_path = os.path.join(out_dir, "damage_results.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Raw per-event damages
            df_events.to_excel(writer, sheet_name="EventDamages", index=False)

            # Pivot table for charting
            pivot = df_events.pivot_table(index="Label", columns="LandCover", values="Damage", fill_value=0)
            pivot.to_excel(writer, sheet_name="EventPivot")

            # EAD per crop table (discrete)
            df_ead = pd.DataFrame([{"LandCover": k, "EAD": v} for k, v in eads_crop.items()])
            df_ead.to_excel(writer, sheet_name="EAD", index=False)

            # Optional trapezoidal EAD per crop table
            if ead_total_trap is not None:
                df_ead_trap = pd.DataFrame([{"LandCover": k, "EAD": v} for k, v in eads_crop_trap.items()])
                df_ead_trap.to_excel(writer, sheet_name="EAD_Trapezoidal", index=False)

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

            # Chart for EAD per crop (discrete)
            worksheet_ead = writer.sheets["EAD"]
            data_ref2 = Reference(worksheet_ead, min_col=2, min_row=1, max_row=len(df_ead) + 1)
            cats_ref2 = Reference(worksheet_ead, min_col=1, min_row=2, max_row=len(df_ead) + 1)
            chart2 = BarChart()
            chart2.type = "col"
            chart2.title = "Expected Annual Damage by Land Cover (Discrete)"
            chart2.x_axis.title = "Land Cover"
            chart2.y_axis.title = "EAD"
            chart2.add_data(data_ref2, titles_from_data=True)
            chart2.set_categories(cats_ref2)
            worksheet_ead.add_chart(chart2, "D2")

            # Chart for EAD per crop (trapezoidal) if present
            if ead_total_trap is not None:
                worksheet_ead_trap = writer.sheets["EAD_Trapezoidal"]
                data_ref3 = Reference(worksheet_ead_trap, min_col=2, min_row=1, max_row=len(df_ead_trap) + 1)
                cats_ref3 = Reference(worksheet_ead_trap, min_col=1, min_row=2, max_row=len(df_ead_trap) + 1)
                chart3 = BarChart()
                chart3.type = "col"
                chart3.title = "Expected Annual Damage by Land Cover (Trapezoidal)"
                chart3.x_axis.title = "Land Cover"
                chart3.y_axis.title = "EAD"
                chart3.add_data(data_ref3, titles_from_data=True)
                chart3.set_categories(cats_ref3)
                worksheet_ead_trap.add_chart(chart3, "D2")

        if out_points:
            if arcpy.Exists(out_points):
                arcpy.management.Delete(out_points)
            arcpy.management.CreateFeatureclass(
                os.path.dirname(out_points),
                os.path.basename(out_points),
                "POINT",
                spatial_reference=crop_sr,
            )
            arcpy.management.AddField(out_points, "Crop", "LONG")
            # Allow event and landcover labels to store long filenames without
            # silently truncating values which can lead to insert cursor
            # failures when the provided string exceeds the field length.
            # Geodatabases support up to 255 characters for text fields which
            # is sufficient for typical raster names that include descriptive
            # prefixes or timestamps.
            arcpy.management.AddField(out_points, "LandCover", "TEXT", field_length=255)
            arcpy.management.AddField(out_points, "Damage", "DOUBLE")
            arcpy.management.AddField(out_points, "Event", "TEXT", field_length=255)
            arcpy.management.AddField(out_points, "RP", "DOUBLE")
            with arcpy.da.InsertCursor(out_points, ["SHAPE@XY", "Crop", "LandCover", "Damage", "Event", "RP"]) as cursor:
                for x, y, c, lc, dmg, lbl, rp_val in points:
                    cursor.insertRow([(x, y), c, lc, dmg, lbl, rp_val])
