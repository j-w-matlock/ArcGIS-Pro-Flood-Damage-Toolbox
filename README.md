# ArcGIS-Pro-Flood-Damage-Toolbox

This toolbox estimates agricultural flood damages in ArcGIS Pro by
sampling a Cropscape raster and one or more flood depth rasters. Crop
code definitions, land-cover names and default per-acre values are
hard-coded within the tool. The *Default Crop Value per Acre* parameter
acts as a fallback for any crop codes that are not in the built-in
lookup.

Optional *Winter*, *Spring*, *Summer* and *Fall Growing Months*
parameters define season boundaries. Crops without specified growing
season are treated as year-round. When an event month falls outside a
crop's listed growing season, the tool assumes year-round susceptibility
and emits a warning. Season selection can be randomized with
user-defined probabilities, or flood months can be randomized directly to
explore seasonal uncertainty.

The *Specific Crop Depth Damage Curve* parameter allows custom
depth-damage relationships and optional growing months for particular
crop codes. Provide a crop code, its depth-damage curve and an optional
list of growing season months (for example: `42`, `0:0,1:0.5,2:1`,
`6,7,8`). Listed codes override the default *Depth-Damage Curve* and
growing months, while other crops continue to use the defaults.

For each flood depth raster the toolbox produces a two–band raster
containing crop type and damage fraction, a CSV summary table and
performs a Monte Carlo analysis with user-defined uncertainty and number
of simulations. The Monte Carlo engine now allows optional uncertainty in
flood month, flood depth and crop value, and the analysis period can be
specified in years to align with USACE CAFRE workflows. Results are
calculated for each impacted crop and annualized using a discrete
1/return-period method. When multiple return periods are supplied, the
toolbox also computes a trapezoidal expected annual damage estimate. The
Excel workbook includes separate tabs for each approach with per-event
damages, per-crop expected annual damages and illustrative charts. The
exported tables include both crop codes and human-readable land-cover
names. Per-event damage exports now also include the standard deviation
and 5th/95th percentile damages across the Monte Carlo simulations to
convey uncertainty, along with the number of flooded pixels and acres.
Optionally, the tool can export a feature class of per-pixel damage
values for mapping and further analysis.

The tool is designed to handle very large rasters efficiently while
producing outputs that can withstand economic review.

