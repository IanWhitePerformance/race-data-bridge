# race-data-bridge

Fetches open 3D data for endurance race courses (triathlon and marathon) so they can be modelled in 3D: terrain, LiDAR surface and terrain models, aerial or satellite imagery, and OpenStreetMap. Runs on GitHub Actions: push a course file to `courses/`, or run the `race-data` workflow from the Actions tab with a race id. Output lands in `data/<race id>/` with a `report.json` of what each source returned.

Inputs here are public: course lines from the organisers' published course maps, and points of interest along them. Nothing personal or athlete-related lives in this repository.

## Sources and licences
- Terrain tiles: AWS Terrain Tiles (Mapzen/Tilezen; SRTM and national datasets), open data.
- OpenStreetMap via Geofabrik extracts or Overpass: © OpenStreetMap contributors, ODbL.
- England: Environment Agency LIDAR composite DTM/DSM, Open Government Licence v3.
- Wales: Welsh Government LiDAR, Open Government Licence v3.
- United States: USGS 3DEP (bare earth and point clouds) and USDA NAIP imagery, public domain.
- New Zealand: LINZ aerial imagery and LiDAR, CC BY 4.0.
- Bavaria and Hesse: geodata of the state survey offices, CC BY 4.0 / DL-DE BY 2.0.
- France: IGN open data, Etalab Open Licence.
- Sentinel-2: Copernicus, free and open.
- Elevation profile check: Open-Meteo (CC BY 4.0).

Derived files keep the licence of their source; credit the sources above when using them.

## Layout
- `courses/`: course specs the workflow fetches. A push that adds or changes a course file fetches that course; a manual run takes a race id or `all`.
- `courses-done/`: specs already fetched (the triathlons); move one back into `courses/` to refresh it.
- `data/<id>/`: the fetched files and `report.json`.
- One job per course, four at a time, each committing its own folder.
