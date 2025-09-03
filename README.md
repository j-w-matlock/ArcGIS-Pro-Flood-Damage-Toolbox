# ArcGIS-Pro-Flood-Damage-Toolbox

This toolbox estimates agricultural flood damages in ArcGIS Pro by
sampling a Cropscape raster and one or more flood depth rasters. Crop
code definitions, land-cover names and default per-acre values are
hard-coded within the tool. The *Default Crop Value per Acre* parameter
acts as a fallback for any crop codes that are not in the built-in
lookup.

If the *Default Growing Season* parameter is left blank, crops without a
specified season are treated as year-round. When an event month falls
outside a crop's listed growing season, the tool assumes year-round
susceptibility and emits a warning.

The *Specific Crop Depth Damage Curve* parameter allows custom
depth-damage relationships for particular crop codes. Provide the crop
code on one line and its depth-damage curve on the following line (for
example: `42` on one line and `0:0,1:0.5,2:1` on the next). Listed codes
override the default *Depth-Damage Curve* while other crops continue to
use the default relationship.

For each flood depth raster the toolbox produces a two–band raster
containing crop type and damage fraction, a CSV summary table and
performs a Monte Carlo analysis with user-defined uncertainty and number
of simulations. The Monte Carlo engine now allows optional uncertainty in
flood month, flood depth and crop value, and the analysis period can be
specified in years to align with USACE CAFRE workflows. Results are calculated for each impacted crop and
annualized using the U.S. Army Corps of Engineers trapezoidal expected
annual damage method. A detailed Excel workbook with per-event damages,
per-crop expected annual damages and illustrative charts is created for
full transparency. The exported tables include both crop codes and
human-readable land-cover names. Per-event damage exports now also
include the standard deviation and 5th/95th percentile damages across
the Monte Carlo simulations to convey uncertainty, along with the number
of flooded pixels and acres.

The tool is designed to handle very large rasters efficiently while
producing outputs that can withstand economic review.
