# ArcGIS-Pro-Flood-Damage-Toolbox

This toolbox estimates agricultural flood damages in ArcGIS Pro by
sampling a Cropscape raster and one or more flood depth rasters.  The
tool supports two ways to supply crop values and growing seasons:

* provide a CSV file containing **CropCode**, **ValuePerAcre** and
  **GrowingSeason** columns, or
* specify a single value and growing season to apply to all sampled crop
  codes.

For each flood depth raster the toolbox produces a two–band raster
containing crop type and damage fraction, a CSV summary table and
performs a Monte Carlo analysis with user‑defined uncertainty and number
of simulations.  Results are annualized using the U.S. Army Corps of
Engineers trapezoidal expected annual damage method and written to CSV
files for full transparency.

The tool is designed to handle very large rasters efficiently while
producing outputs that can withstand economic review.
