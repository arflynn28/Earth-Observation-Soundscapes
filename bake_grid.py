#!/usr/bin/env python3
"""bake_grid.py - one time Earth Engine download of all the layers into an offline uint8 grid.

    python bake_grid.py world --bbox -125 35 -120 40        # TEST on a small region first
    python bake_grid.py world                                # whole globe at ~1 km
    python bake_grid.py site --lat 37.87 --lon -122.5 --radius-km 100 --scale 500
    python bake_grid.py world --bands depth pres             # add or redo only some layers later

Each layer is its own raw file: data/<kind>.<band>.u8 (H x W, uint8). There is one shared
data/<kind>.json, and a resume map data/<kind>.<band>.done.npy for each layer. Run the same command
again to resume. Finished tiles are skipped for each layer, so a new layer can be added to an
existing grid without redoing the others. The kiosk then only needs numpy and these files.

Encoding: 0 = no data, 1-255 spans BOUNDS (earth_soundscapes.py), land cover keeps its class numbers.
World bakes use MODIS cloud state (cheap), site bakes use Sentinel-2 (10 m, slow but detailed).
Open ocean tiles (land cover almost all water) skip the expensive land and radar layers.
Time averaged layers use the same months as the live mode (MONTHS), --full-year uses all 12.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from earth_soundscapes import BANDS, BOUNDS, MONTHS, build_layers, half_extent, init_ee


def quantize(image, band):
    """One float band -> one uint8 band. 0 = no data, 1-255 spans BOUNDS, land cover keeps its class numbers."""
    if band != "lc":
        low, high = BOUNDS[band]
        image = image.subtract(low).divide(high - low).clamp(0, 1).multiply(254).round().add(1)
    return image.unmask(0).toUint8()


def is_open_ocean(lc):
    """lc is a uint8 land cover tile (0 = no data).
    True when the tile is almost all water and has no land in it."""
    if lc.size == 0:
        return False
    has_land = ((lc >= 1) & (lc <= 16)).any()
    mostly_water = (lc == 17).mean() >= 0.9
    return bool(not has_land and mostly_water)


def fetch_tile(west, north, cols, rows, res, hires, bands=BANDS, months=None, lc=None, retries=2):
    """Download one tile from Earth Engine as {band: uint8 array (rows, cols)}.
    The tile's north west corner is (west, north).

    Every layer is its own request, so a slow layer is retried by itself. `lc` is the tile's land
    cover if it is already baked. If not, it is fetched first (it's cheap) to spot open ocean tiles,
    where the land and radar layers are skipped (left as 0 = no data)."""
    import ee

    rectangle = ee.Geometry.Rectangle([west, north - rows * res, west + cols * res, north],
                                      geodesic=False)
    grid = {
        "dimensions": {"width": cols, "height": rows},
        "affineTransform": {"scaleX": res, "shearX": 0, "translateX": west,
                            "shearY": 0, "scaleY": -res, "translateY": north},
        "crsCode": "EPSG:4326",
    }

    def pull(layers, band):
        for attempt in range(retries):
            try:
                array = ee.data.computePixels({"expression": quantize(layers[band], band),
                                               "fileFormat": "NUMPY_NDARRAY", "grid": grid})
                field_name = band if band in array.dtype.names else array.dtype.names[0]
                return np.asarray(array[field_name], np.uint8)
            except Exception as exc:
                if attempt == retries - 1:
                    raise RuntimeError(f"layer {band}: {exc}") from exc
                time.sleep(5 * (attempt + 1))

    result = {}
    if lc is None:
        # Get the land cover first so we know if this tile is open ocean
        lc = pull(build_layers(rectangle, hires, months), "lc")
        result["lc"] = lc

    layers = build_layers(rectangle, hires, months, ocean=is_open_ocean(lc))
    for band in bands:
        if band in result:
            continue
        if band in layers:
            result[band] = pull(layers, band)
        else:
            result[band] = np.zeros((rows, cols), np.uint8)    # skipped layer = no data
    return {band: result[band] for band in bands}


def _paths(base, band):
    """The data file and the resume file for one layer"""
    return f"{base}.{band}.u8", f"{base}.{band}.done.npy"


def bake(path, bbox, res, tile_px=256, hires=False, workers=6, tile_fn=None, retries=2,
         months=MONTHS, bands=BANDS):
    """Bake `bands` over `bbox` (west, south, east, north in degrees) at `res` degrees per pixel.
    The layers are saved as <path without extension>.<band>.u8 files.

    tile_fn(west, north, cols, rows, res, hires, bands, lc) -> {band: array} can be swapped out
    for testing. By default it downloads from Earth Engine."""
    bands = [band for band in BANDS if band in set(bands)]
    west, south, east, north = bbox
    width = math.ceil((east - west) / res)
    height = math.ceil((north - south) / res)
    base = os.path.splitext(path)[0]
    geometry = {"bbox": [west, north - height * res, west + width * res, north],
                "res": res, "height": height, "width": width}
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)

    # If this grid was started before, make sure this run matches it
    meta_path = base + ".json"
    already_baked = []
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        if old.get("layout") != "bands":
            sys.exit(f"{meta_path} is from the older single-file format. Delete {base}.* "
                     f"(or use another --out) and run again.")
        for key in geometry:
            if old[key] != geometry[key]:
                sys.exit(f"{meta_path} has different bounds or resolution than this command. "
                         f"Use the same arguments as the original bake, or another --out.")
        already_baked = old["bands"]

    meta = {**geometry, "layout": "bands", "hires": hires,
            "months": list(months) if months else None,
            "bands": [band for band in BANDS if band in set(already_baked) | set(bands)]}
    json.dump(meta, open(meta_path, "w"), indent=1)

    # Open the data file for each layer, and the map of which tiles are finished
    tiles_down = math.ceil(height / tile_px)
    tiles_across = math.ceil(width / tile_px)
    memmaps, done = {}, {}
    for band in bands:
        data_file, done_file = _paths(base, band)
        mode = "r+" if os.path.exists(data_file) else "w+"
        memmaps[band] = np.memmap(data_file, np.uint8, mode, shape=(height, width))
        if os.path.exists(done_file):
            done[band] = np.load(done_file)
        else:
            done[band] = np.zeros((tiles_down, tiles_across), bool)
        if done[band].shape != (tiles_down, tiles_across):
            sys.exit(f"{done_file} doesn't match this tiling; delete it (and {data_file}) to start over")

    # If land cover isn't being baked now but is already on disk, use it to spot open ocean tiles
    saved_lc = None
    if "lc" not in bands:
        data_file, done_file = _paths(base, "lc")
        if os.path.exists(data_file) and os.path.exists(done_file):
            saved_lc = (np.memmap(data_file, np.uint8, "r", shape=(height, width)), np.load(done_file))

    if tile_fn is None:
        def tile_fn(west, north, cols, rows, res, hires, need, lc):
            return fetch_tile(west, north, cols, rows, res, hires, need, months=months, lc=lc)

    # Every tile that still has at least one unfinished layer
    todo = []
    for i in range(tiles_down):
        for j in range(tiles_across):
            if any(not done[band][i, j] for band in bands):
                todo.append((i, j))
    print(f"{width}x{height} px, {tiles_down * tiles_across} tiles, {len(todo)} to do, layers: {', '.join(bands)}")

    def work(i, j):
        row0, col0 = i * tile_px, j * tile_px
        rows, cols = min(tile_px, height - row0), min(tile_px, width - col0)
        need = [band for band in bands if not done[band][i, j]]

        # Land cover for this tile, if we already have it
        lc = None
        if "lc" not in need:
            if "lc" in memmaps:
                lc = np.array(memmaps["lc"][row0:row0 + rows, col0:col0 + cols])
            elif saved_lc is not None and saved_lc[1][i, j]:
                lc = np.array(saved_lc[0][row0:row0 + rows, col0:col0 + cols])

        for attempt in range(retries):
            try:
                got = tile_fn(geometry["bbox"][0] + col0 * res, geometry["bbox"][3] - row0 * res,
                              cols, rows, res, hires, need, lc)
                break
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                print(f"  tile {i},{j} retry ({exc})")
                time.sleep(5 * (attempt + 1))

        for band in need:
            memmaps[band][row0:row0 + rows, col0:col0 + cols] = got[band]
        return i, j, need

    def save():
        for band in bands:
            np.save(_paths(base, band)[1], done[band])
            memmaps[band].flush()

    start = time.time()
    failed = []
    pool = ThreadPoolExecutor(workers)
    futures = {pool.submit(work, i, j): (i, j) for i, j in todo}
    try:
        for count, future in enumerate(as_completed(futures), 1):
            try:
                i, j, need = future.result()
                for band in need:
                    done[band][i, j] = True
            except Exception as exc:
                failed.append(futures[future])
                print(f"  tile {futures[future]} FAILED: {exc}")

            # Save every 10 tiles and print how long is left
            if count % 10 == 0 or count == len(todo):
                save()
                minutes_left = (time.time() - start) / count * (len(todo) - count) / 60
                print(f"  {count}/{len(todo)} tiles, ~{minutes_left:.0f} min left")
    except KeyboardInterrupt:
        # Ctrl+C: stop starting new tiles
        print("\nStopping: cancelling queued tiles; tiles already in flight finish first (up to a minute)...")
        pool.shutdown(wait=False, cancel_futures=True)
        save()
        print("Progress saved. Re-run the same command to resume.")
        sys.exit(130)

    pool.shutdown()
    save()
    print(f"done. {len(failed)} failed tiles" + (" - re-run to retry them" if failed else ""))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("kind", choices=["world", "site"])
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--radius-km", type=float, default=10)
    parser.add_argument("--scale", type=float, help="metres per pixel (default 1000 world, 30 site)")
    parser.add_argument("--tile-px", type=int, default=256)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bands", nargs="+", choices=BANDS, default=list(BANDS),
                        help="layers to bake (default all), use this to add or redo some later")
    parser.add_argument("--out", default="data")
    parser.add_argument("--full-year", action="store_true",
                        help="all 12 months (slower, default samples %s)" % (MONTHS,))
    args = parser.parse_args()

    world = args.kind == "world"
    scale = args.scale or (1000 if world else 30)
    res = scale / 111320    # degrees per pixel

    # Work out the area to bake
    if args.bbox:
        bbox = tuple(args.bbox)
    elif world:
        bbox = (-180, -85, 180, 85)
    elif args.lat is not None and args.lon is not None:
        half_lat, half_lon = half_extent(args.lat, 2 * args.radius_km)
        bbox = (args.lon - half_lon, args.lat - half_lat, args.lon + half_lon, args.lat + half_lat)
    else:
        sys.exit("site needs --lat and --lon (or --bbox)")

    init_ee()
    bake(os.path.join(args.out, f"{args.kind}.u8"), bbox, res, args.tile_px, hires=not world,
         workers=args.workers, months=None if args.full_year else MONTHS, bands=args.bands)


if __name__ == "__main__":
    main()
