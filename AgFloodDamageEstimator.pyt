import arcpy
import os
import shutil
import tempfile
from dataclasses import dataclass
from random import Random
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from openpyxl.chart import BarChart, Reference


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


@dataclass(frozen=True)
class DamageCurve:
    depths: np.ndarray
    fractions: np.ndarray
    months: Optional[FrozenSet[int]] = None


@dataclass(frozen=True)
class EventSpec:
    path: str
    month: int
    return_period: float


@dataclass
class SimulationConfig:
    crop_path: str
    crop_sr: object
    out_dir: str
    default_value: float
    base_curve: DamageCurve
    specific_curves: Dict[int, DamageCurve]
    events: List[EventSpec]
    deterministic: bool
    runs: int
    analysis_years: int
    frac_std: float
    depth_std: float
    value_std: float
    random: Random
    random_month: bool
    random_season: bool
    season_lookup: Dict[str, Optional[FrozenSet[int]]]
    season_names: Sequence[str]
    season_weights: Sequence[float]
    season_months: Optional[FrozenSet[int]]
    out_points: Optional[str]

    @property
    def total_runs(self) -> int:
        return self.runs * self.analysis_years

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

        mode = arcpy.Parameter(
            displayName="Simulation Mode",
            name="simulation_mode",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        mode.filter.type = "ValueList"
        mode.filter.list = ["Single Run", "Monte Carlo"]
        mode.value = "Monte Carlo"
        mode.description = (
            "Choose between a single deterministic evaluation of each event or a Monte Carlo simulation that "
            "samples from the uncertainty parameters."
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
            mode,
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
        config = self._collect_config(params)
        results: List[Dict[str, object]] = []
        points: List[Tuple[float, float, int, str, float, str, float]] = []

        for event in config.events:
            event_results, event_points = self._simulate_event(config, event)
            results.extend(event_results)
            points.extend(event_points)

        if not results:
            raise ValueError("No valid events provided")

        self._write_outputs(config, results, points, messages)

    def _collect_config(self, params) -> SimulationConfig:
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
        simulation_mode = params[10].valueAsText or "Monte Carlo"
        frac_std = float(params[11].value)
        runs = int(params[12].value)
        random = Random(int(params[13].value))
        random_month = bool(params[14].value) if params[14].value is not None else False
        random_season = bool(params[15].value) if params[15].value is not None else False
        season_prob_table = params[16].values if params[16].values else []
        depth_std = float(params[17].value) if params[17].value is not None else 0.0
        value_std = float(params[18].value) if params[18].value is not None else 0.0
        analysis_years = int(params[19].value) if params[19].value is not None else 1
        out_points = params[20].valueAsText if params[20].value else None

        os.makedirs(out_dir, exist_ok=True)

        if value_acre < 0:
            raise ValueError("Default crop value per acre must be non-negative.")

        if not 0 <= frac_std <= 1:
            raise ValueError("Damage fraction standard deviation must be between 0 and 1.")

        if depth_std < 0:
            raise ValueError("Flood depth standard deviation must be zero or positive.")

        if value_std < 0:
            raise ValueError("Crop value standard deviation must be zero or positive.")

        if analysis_years < 1:
            raise ValueError("Analysis period must be at least one year.")

        deterministic = simulation_mode == "Single Run"
        if deterministic:
            runs = 1
            frac_std = 0.0
            depth_std = 0.0
            value_std = 0.0
            random_month = False
            random_season = False
        elif runs < 1:
            raise ValueError("Monte Carlo simulations must be at least 1.")

        base_curve_pts = parse_curve(curve_str)
        base_curve = DamageCurve(
            depths=np.array([d for d, _ in base_curve_pts], dtype=float),
            fractions=np.array([f for _, f in base_curve_pts], dtype=float),
        )

        specific_curves = self._parse_specific_curves(specific_table)

        crop_desc = arcpy.Describe(crop_path)
        crop_sr = getattr(crop_desc, "spatialReference", None)
        if not crop_sr or crop_sr.name in (None, ""):
            raise ValueError("Crop raster must have a defined spatial reference.")

        (
            season_lookup,
            season_months,
            season_names,
            season_weights,
        ) = self._build_season_config(
            winter_str,
            spring_str,
            summer_str,
            fall_str,
            season_prob_table,
        )

        events = self._parse_events(event_table)

        return SimulationConfig(
            crop_path=crop_path,
            crop_sr=crop_sr,
            out_dir=out_dir,
            default_value=value_acre,
            base_curve=base_curve,
            specific_curves=specific_curves,
            events=events,
            deterministic=deterministic,
            runs=runs,
            analysis_years=analysis_years,
            frac_std=frac_std,
            depth_std=depth_std,
            value_std=value_std,
            random=random,
            random_month=random_month,
            random_season=random_season,
            season_lookup=season_lookup,
            season_names=season_names,
            season_weights=season_weights,
            season_months=season_months,
            out_points=out_points,
        )

    def _parse_specific_curves(self, specific_table: Iterable[Sequence]) -> Dict[int, DamageCurve]:
        curves: Dict[int, DamageCurve] = {}
        for row in specific_table:
            if len(row) < 2:
                continue
            code = row[0]
            curve_txt = row[1]
            grow_txt = row[2] if len(row) > 2 else None
            if not curve_txt:
                continue
            pts = parse_curve(curve_txt)
            months = parse_months(grow_txt)
            curves[int(code)] = DamageCurve(
                depths=np.array([d for d, _ in pts], dtype=float),
                fractions=np.array([f for _, f in pts], dtype=float),
                months=frozenset(months) if months else None,
            )
        return curves

    def _build_season_config(
        self,
        winter_str: Optional[str],
        spring_str: Optional[str],
        summer_str: Optional[str],
        fall_str: Optional[str],
        season_prob_table: Iterable[Sequence],
    ) -> Tuple[Dict[str, Optional[FrozenSet[int]]], Optional[FrozenSet[int]], Sequence[str], Sequence[float]]:
        def to_frozen(months: Optional[Iterable[int]]) -> Optional[FrozenSet[int]]:
            if not months:
                return None
            return frozenset(int(m) for m in months)

        winter_months = to_frozen(parse_months(winter_str))
        spring_months = to_frozen(parse_months(spring_str))
        summer_months = to_frozen(parse_months(summer_str))
        fall_months = to_frozen(parse_months(fall_str))

        season_lookup: Dict[str, Optional[FrozenSet[int]]] = {
            "Winter": winter_months,
            "Spring": spring_months,
            "Summer": summer_months,
            "Fall": fall_months,
        }

        month_union = set()
        for months in season_lookup.values():
            if months:
                month_union.update(months)
        season_months = frozenset(month_union) if month_union else None

        season_prob_dict: Dict[str, float] = {}
        for season, probability in season_prob_table:
            prob = float(probability)
            if prob < 0:
                raise ValueError("Season probabilities must be zero or positive.")
            season_prob_dict[str(season)] = prob
        if not season_prob_dict:
            season_prob_dict = {
                "Winter": 0.25,
                "Spring": 0.25,
                "Summer": 0.25,
                "Fall": 0.25,
            }

        season_names: List[str] = list(season_lookup.keys())
        weights = [season_prob_dict.get(name, 1.0) for name in season_names]
        total = sum(weights)
        if total <= 0:
            weights = [1.0] * len(season_names)
        else:
            weights = [w / total for w in weights]

        return season_lookup, season_months, season_names, weights

    def _parse_events(self, event_table: Iterable[Sequence]) -> List[EventSpec]:
        events: List[EventSpec] = []
        for row in event_table:
            if len(row) < 3:
                continue
            depth_path, month, rp = row
            if month is None:
                raise ValueError("Flood month is required for each event.")
            month = int(month)
            if not 1 <= month <= 12:
                raise ValueError(f"Flood month {month} must be between 1 and 12.")
            if rp is None:
                raise ValueError("Return period is required for each event.")
            rp = float(rp)
            if rp <= 0:
                raise ValueError(f"Return period {rp} must be greater than zero.")
            depth_str = self._normalize_raster_path(depth_path)
            if not depth_str:
                continue
            events.append(EventSpec(depth_str, month, rp))
        return events

    @staticmethod
    def _normalize_raster_path(path) -> str:
        if hasattr(path, "valueAsText"):
            return path.valueAsText
        if hasattr(path, "dataSource"):
            return path.dataSource
        return str(path)

    @staticmethod
    def _sanitize_label(path: str) -> str:
        base = os.path.splitext(os.path.basename(path))[0]
        return base.replace(" ", "_").replace(".", "_")

    def _select_month(self, config: SimulationConfig, default_month: int) -> int:
        if config.random_season:
            season = config.random.choices(config.season_names, weights=config.season_weights)[0]
            months = config.season_lookup.get(season)
            if months:
                return config.random.choice(list(months))
            return config.random.randint(1, 12)
        if config.random_month:
            return config.random.randint(1, 12)
        return default_month

    @staticmethod
    def _compute_active_indices(
        code_indices: Dict[int, np.ndarray],
        code_months: Dict[int, Optional[FrozenSet[int]]],
    ) -> Optional[Dict[int, np.ndarray]]:
        if not code_indices:
            return None
        if not any(months is not None for months in code_months.values()):
            return None
        lists: Dict[int, List[np.ndarray]] = {m: [] for m in range(1, 13)}
        for code, indices in code_indices.items():
            months = code_months.get(code)
            if months is None:
                for m in range(1, 13):
                    lists[m].append(indices)
            else:
                for m in months:
                    lists[int(m)].append(indices)
        active: Dict[int, np.ndarray] = {}
        for m in range(1, 13):
            if lists[m]:
                active[m] = np.concatenate(lists[m])
            else:
                active[m] = np.array([], dtype=np.int64)
        return active

    @staticmethod
    def _temporary_raster_workspace(config: SimulationConfig) -> Tuple[str, Optional[str]]:
        """Select a disk-backed workspace for intermediate rasters.

        Using the in-memory workspace can trigger out-of-memory errors when
        large rasters are projected. Prefer any scratch workspace configured in
        the current ArcPy environment and fall back to the output directory when
        necessary so the intermediate rasters are written to disk instead of
        memory.
        """

        candidates = [
            getattr(arcpy.env, "scratchWorkspace", None),
            getattr(arcpy.env, "scratchGDB", None),
            getattr(arcpy.env, "scratchFolder", None),
            config.out_dir,
        ]
        for workspace in candidates:
            if not workspace:
                continue
            workspace_str = os.fspath(workspace)
            if workspace_str.lower() == "in_memory":
                continue
            workspace_str = os.path.abspath(workspace_str)
            try:
                if arcpy.Exists(workspace_str):
                    if AgFloodDamageEstimator._workspace_has_path_capacity(workspace_str):
                        return workspace_str, None
            except Exception:
                # Fall back to returning the path even if Exists fails so the
                # caller still has a usable location.
                if AgFloodDamageEstimator._workspace_has_path_capacity(workspace_str):
                    return workspace_str, None

        temp_dir = tempfile.mkdtemp(prefix="agfd_")
        return temp_dir, temp_dir

    @staticmethod
    def _workspace_has_path_capacity(workspace: str) -> bool:
        """Return True if the workspace leaves room for raster dataset names.

        Raster outputs written to folders on Windows can fail with ERROR 160155
        when the fully-qualified path exceeds the 260 character limit. Guard
        against using extremely deep output folders by ensuring there is enough
        space for typical raster names before selecting the workspace.
        """

        # Reserve space for a reasonably long raster name plus extension to
        # avoid triggering Windows path length issues when writing temporary
        # rasters. The +10 buffer covers the unique suffix generated by ArcPy.
        max_path = os.path.join(workspace, "x" * 80 + ".tif")
        return len(max_path) < 240

    @staticmethod
    def _make_temp_raster_path(workspace: str, label: str, suffix: str) -> str:
        base_name = f"{label}_{suffix}"
        use_gdb = workspace.lower().endswith(".gdb")
        name = base_name if use_gdb else f"{base_name}.tif"
        return arcpy.CreateUniqueName(name, workspace)

    def _simulate_event(
        self,
        config: SimulationConfig,
        event: EventSpec,
    ) -> Tuple[List[Dict[str, object]], List[Tuple[float, float, int, str, float, str, float]]]:
        depth_desc = arcpy.Describe(event.path)
        depth_sr = getattr(depth_desc, "spatialReference", None)
        if not depth_sr or getattr(depth_sr, "type", "") != "Projected":
            raise ValueError(
                f"Depth raster {event.path} must use a projected coordinate system with linear units"
                " in meters or feet so acreage can be computed reliably."
            )
        meters_per_unit = depth_sr.metersPerUnit
        if meters_per_unit in (None, 0):
            raise ValueError(
                f"Depth raster {event.path} spatial reference does not provide a valid metersPerUnit conversion."
            )

        label = self._sanitize_label(event.path)
        temp_workspace, cleanup_dir = self._temporary_raster_workspace(config)
        crop_proj = self._make_temp_raster_path(temp_workspace, label, "crop_proj")
        crop_clip = self._make_temp_raster_path(temp_workspace, label, "crop")
        depth_clip = self._make_temp_raster_path(temp_workspace, label, "depth")
        temp_datasets = [crop_proj, crop_clip, depth_clip]

        crop_arr = None
        depth_arr = None
        clip_width = None
        clip_height = None
        inter = None
        cell_area_acres = None

        try:
            with arcpy.EnvManager(snapRaster=event.path, cellSize=event.path):
                arcpy.management.ProjectRaster(
                    config.crop_path,
                    crop_proj,
                    depth_sr,
                    "NEAREST",
                    depth_desc.meanCellWidth,
                )
                crop_proj_desc = arcpy.Describe(crop_proj)
                inter = intersect_extent(crop_proj_desc.extent, depth_desc.extent)
            if not inter:
                raise ValueError(f"Depth raster {event.path} does not overlap crop raster extent")

            extent_str = f"{inter[0]} {inter[1]} {inter[2]} {inter[3]}"

            with arcpy.EnvManager(snapRaster=event.path, cellSize=event.path):
                arcpy.management.Clip(crop_proj, extent_str, crop_clip, "#", "0", "NONE", "MAINTAIN_EXTENT")
                arcpy.management.Clip(event.path, extent_str, depth_clip, "#", "0", "NONE", "MAINTAIN_EXTENT")
                crop_arr = arcpy.RasterToNumPyArray(crop_clip)
                depth_arr = arcpy.RasterToNumPyArray(depth_clip)
                depth_clip_desc = arcpy.Describe(depth_clip)
                clip_width = abs(depth_clip_desc.meanCellWidth)
                clip_height = abs(depth_clip_desc.meanCellHeight)
                cell_area_acres = clip_width * clip_height * meters_per_unit ** 2 / 4046.8564224
        finally:
            for ds in temp_datasets:
                if arcpy.Exists(ds):
                    arcpy.management.Delete(ds)
            if cleanup_dir:
                try:
                    shutil.rmtree(cleanup_dir, ignore_errors=True)
                except Exception:
                    pass

        if cell_area_acres is None:
            raise RuntimeError(f"Failed to compute cell area for depth raster {event.path}.")
        if depth_arr is None or crop_arr is None:
            raise RuntimeError(f"Failed to read raster data for event {event.path}.")
        if depth_arr.shape != crop_arr.shape:
            raise ValueError(
                f"Masked depth raster {label} shape does not match crop raster. "
                f"Crop: {crop_arr.shape}, Depth: {depth_arr.shape}"
            )

        mask = (crop_arr > 0) & (depth_arr > 0)
        crop_masked = crop_arr[mask].astype(int)
        depth_masked = depth_arr[mask]
        if crop_masked.size == 0:
            return [], []

        crop_codes = np.unique(crop_masked)
        max_code = int(crop_codes.max())
        val_lookup = np.full(max_code + 1, config.default_value, dtype=float)
        for code, (_, val) in CROP_DEFINITIONS.items():
            if code <= max_code:
                val_lookup[code] = val

        pixel_counts_arr = np.bincount(crop_masked, minlength=max_code + 1)
        pixel_counts = {int(code): int(pixel_counts_arr[int(code)]) for code in crop_codes}

        code_indices: Dict[int, np.ndarray] = {int(code): np.where(crop_masked == code)[0] for code in crop_codes}
        spec_curves = {
            code: config.specific_curves[code]
            for code in config.specific_curves
            if code in code_indices
        }

        code_months: Dict[int, Optional[FrozenSet[int]]] = {}
        for code in code_indices:
            curve = spec_curves.get(code)
            if curve and curve.months is not None:
                code_months[code] = curve.months
            else:
                code_months[code] = config.season_months

        active_indices_by_month = self._compute_active_indices(code_indices, code_months)

        damages_runs: Dict[int, List[float]] = {int(code): [] for code in crop_codes}
        damage_accum = np.zeros_like(depth_arr, dtype=float) if config.out_points else None

        curve_depths = config.base_curve.depths
        curve_fracs = config.base_curve.fractions
        spec_indices = {code: code_indices[code] for code in spec_curves}

        crop_codes_list = [int(code) for code in crop_codes]

        for _ in range(config.total_runs):
            sim_month = self._select_month(config, event.month)

            active_indices = None
            if active_indices_by_month is not None:
                active_indices = active_indices_by_month.get(sim_month, np.array([], dtype=np.int64))
                if active_indices.size == 0:
                    for code in crop_codes_list:
                        damages_runs[code].append(0.0)
                    continue

            rng = None
            if not config.deterministic and (
                config.depth_std > 0 or config.frac_std > 0 or config.value_std > 0
            ):
                rng = np.random.default_rng(config.random.randint(0, 2**32 - 1))

            depth_sim = depth_masked
            if rng is not None and config.depth_std > 0:
                depth_offset = rng.normal(0, config.depth_std)
                depth_sim = np.clip(depth_masked + depth_offset, 0, None)

            frac = np.interp(
                depth_sim,
                curve_depths,
                curve_fracs,
                left=curve_fracs[0],
                right=curve_fracs[-1],
            )

            for code, indices in spec_indices.items():
                spec_curve = spec_curves[code]
                frac[indices] = np.interp(
                    depth_sim[indices],
                    spec_curve.depths,
                    spec_curve.fractions,
                    left=spec_curve.fractions[0],
                    right=spec_curve.fractions[-1],
                )

            if rng is not None and config.frac_std > 0:
                frac = np.clip(frac + rng.normal(0, config.frac_std, size=frac.shape), 0, 1)

            values = val_lookup[crop_masked]
            if rng is not None and config.value_std > 0:
                values = np.clip(values + rng.normal(0, config.value_std, size=values.shape), 0, None)

            if active_indices is not None:
                active_values = np.zeros_like(values)
                if active_indices.size > 0:
                    active_values[active_indices] = values[active_indices]
                values = active_values

            dmg_vals = frac * values * cell_area_acres
            dmg_per_crop = np.bincount(crop_masked, weights=dmg_vals, minlength=max_code + 1)
            for code in crop_codes_list:
                damages_runs[code].append(float(dmg_per_crop[code]))

            if damage_accum is not None:
                damage_accum[mask] += dmg_vals

        points: List[Tuple[float, float, int, str, float, str, float]] = []
        if damage_accum is not None:
            damage_avg = damage_accum / config.total_runs
            xmin, ymin, xmax, ymax = inter
            cw = clip_width
            ch = clip_height
            x0 = xmin + cw / 2
            y0 = ymax - ch / 2
            rows, cols = np.nonzero(mask)
            masked_crops = crop_arr[mask]
            masked_damages = damage_avg[mask]
            for row, col, crop_code, dmg in zip(rows, cols, masked_crops, masked_damages):
                crop_code = int(crop_code)
                landcover = CROP_DEFINITIONS.get(crop_code, ("Unknown", config.default_value))[0]
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
                        float(event.return_period),
                    )
                )

        results: List[Dict[str, object]] = []
        for code in crop_codes_list:
            arr = np.array(damages_runs[code], dtype=float)
            avg_damage = float(arr.mean())
            std_damage = float(arr.std(ddof=0))
            p05 = float(np.percentile(arr, 5))
            p95 = float(np.percentile(arr, 95))
            name, _ = CROP_DEFINITIONS.get(code, ("Unknown", config.default_value))
            results.append(
                {
                    "Label": label,
                    "RP": float(event.return_period),
                    "Crop": int(code),
                    "LandCover": name,
                    "Damage": avg_damage,
                    "StdDev": std_damage,
                    "P05": p05,
                    "P95": p95,
                    "FloodedAcres": pixel_counts[code] * cell_area_acres,
                    "FloodedPixels": pixel_counts[code],
                    "ValuePerAcre": float(val_lookup[code]),
                }
            )

        return results, points

    def _write_outputs(
        self,
        config: SimulationConfig,
        results: List[Dict[str, object]],
        points: List[Tuple[float, float, int, str, float, str, float]],
        messages,
    ) -> None:
        df_events = pd.DataFrame(results)

        def calc_ead_discrete(df: pd.DataFrame) -> float:
            return float((df["Damage"] / df["RP"]).sum())

        df_total = df_events.groupby(["Label", "RP"], as_index=False)["Damage"].sum()
        ead_total = calc_ead_discrete(df_total)

        eads_crop = {name: calc_ead_discrete(g) for name, g in df_events.groupby("LandCover")}

        ead_total_trap: Optional[float] = None
        eads_crop_trap: Dict[str, float] = {}
        if df_total["RP"].nunique() > 1:

            def calc_ead_trap(df: pd.DataFrame) -> float:
                df = df.sort_values("RP", ascending=False)
                probs = 1 / df["RP"].to_numpy()
                damages = df["Damage"].to_numpy()
                probs = np.concatenate(([0.0], probs, [1.0]))
                damages = np.concatenate(([damages[0]], damages, [0.0]))
                return float(np.trapz(damages, probs))

            ead_total_trap = calc_ead_trap(df_total)
            eads_crop_trap = {name: calc_ead_trap(g) for name, g in df_events.groupby("LandCover")}

        with open(os.path.join(config.out_dir, "ead.csv"), "w") as f:
            f.write(f"EAD,{ead_total}\\n")
        messages.addMessage(f"Expected Annual Damage (Discrete): {ead_total:,.0f}")

        if ead_total_trap is not None:
            with open(os.path.join(config.out_dir, "ead_trapezoidal.csv"), "w") as f:
                f.write(f"EAD,{ead_total_trap}\\n")
            messages.addMessage(f"Expected Annual Damage (Trapezoidal): {ead_total_trap:,.0f}")

        excel_path = os.path.join(config.out_dir, "damage_results.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_events.to_excel(writer, sheet_name="EventDamages", index=False)

            pivot = df_events.pivot_table(index="Label", columns="LandCover", values="Damage", fill_value=0)
            pivot.to_excel(writer, sheet_name="EventPivot")

            df_ead = pd.DataFrame([{"LandCover": k, "EAD": v} for k, v in eads_crop.items()])
            df_ead.to_excel(writer, sheet_name="EAD", index=False)

            if ead_total_trap is not None:
                df_ead_trap = pd.DataFrame([{"LandCover": k, "EAD": v} for k, v in eads_crop_trap.items()])
                df_ead_trap.to_excel(writer, sheet_name="EAD_Trapezoidal", index=False)

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

        if config.out_points:
            out_points = config.out_points
            if arcpy.Exists(out_points):
                arcpy.management.Delete(out_points)
            arcpy.management.CreateFeatureclass(
                os.path.dirname(out_points),
                os.path.basename(out_points),
                "POINT",
                spatial_reference=config.crop_sr,
            )
            arcpy.management.AddField(out_points, "Crop", "LONG")
            arcpy.management.AddField(out_points, "LandCover", "TEXT", field_length=255)
            arcpy.management.AddField(out_points, "Damage", "DOUBLE")
            arcpy.management.AddField(out_points, "Event", "TEXT", field_length=255)
            arcpy.management.AddField(out_points, "RP", "DOUBLE")
            with arcpy.da.InsertCursor(out_points, ["SHAPE@XY", "Crop", "LandCover", "Damage", "Event", "RP"]) as cursor:
                for x, y, crop_code, landcover, damage, label, rp in points:
                    cursor.insertRow([(x, y), crop_code, landcover, damage, label, rp])
