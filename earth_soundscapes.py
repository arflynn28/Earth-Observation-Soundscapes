#!/usr/bin/env python3
"""Earth Observation Soundscapes

Slide a square across the globe and the song it plays changes with the ground under it.

    satellite layers -> small square of pixels -> musical features for each step -> MIDI

How the pixels turn into music (one square = 32 steps left to right = one 4 bar loop)
    pitch        height above / depth below sea level + temperature, snapped to a scale
                 (mountains and ocean trenches both pull the pitch down)
    rhythm       cloud cover sets the notes per step (1-4) and the overall tempo
    instruments  land cover class -> General MIDI instrument (one MIDI channel per class)
    volume       forest and other nature -> EVI, crops and cities -> night lights,
                 water -> sea level pressure (lower pressure = louder)
    percussion   Sentinel-1 VV radar -> extra drum, snare backbeat (no drums over open water)

Usage
    python earth_soundscapes.py --demo                    # fake data, no Earth Engine needed
    python earth_soundscapes.py --lat 37.87 --lon -122.5  # one square -> earth_song.mid
    python earth_soundscapes.py --grid data/world.u8 --km 64 --lat 48 --lon 11   # offline
    python earth_soundscapes.py --ports                   # list MIDI output ports
    python earth_soundscapes.py --live --port "IAC Driver Bus 1"   # move with n/s/e/w

The kiosk (kiosk.py) imports this file and adds the map page and the sound.
Needs numpy. Earth Engine mode also needs earthengine-api. Live MIDI needs mido + python-rtmidi.
"""
from __future__ import annotations

__version__ = "2026-11-24.1"

import argparse, base64, io, logging, math, os, struct, sys, threading, time, wave, warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# Earth Engine logs a harmless warning on every request about per-request timeouts,
# so turn that one logger down (real errors still show up)
logging.getLogger("google_auth_httplib2").setLevel(logging.ERROR)


# Settings
EE_PROJECT = os.environ.get("EE_PROJECT", "ee-arflynn28")

YEAR = 2024            # year used for the time averaged layers (radar, clouds, temperature)
ERA_YEAR = 2019        # sea level pressure year (the ERA5 daily data stops in 2020)
FETCH_THREADS = 7      # Earth Engine requests at once (use 2-3 if layers fail)
MONTHS = (1, 4, 7, 10) # only use these months, about 3x faster and still seasonal (None = all)

PATCH_KM = 10          # width of the square on the ground
PATCH_PX = 64          # pixels fetched along each side of the square

STEPS = 32             # 32 eighth notes = 4 bars of 4/4
STEP_BEATS = 0.5
GATE = 0.9             # how much of each step a note is held
BPM_RANGE = (80, 140)  # clear sky = slow, cloudy = fast

# Melody range and scale (C3 to C6, major pentatonic)
LO, HI, ROOT = 48, 84, 0
SCALE = (0, 2, 4, 7, 9)
SCALE_NOTES = np.array([n for n in range(LO, HI + 1) if (n - ROOT) % 12 in SCALE])

W_ELEV, W_TEMP = 0.5, 0.5   # how much elevation and temperature count toward pitch
CONTOUR = 0.3               # how far the tune can wander from the place's register (0-1)
MAX_VOICES = 3              # instruments playing at once in a step
MIN_SHARE = 0.05            # land cover needs at least this share of a step to play
SNARE_EVERY = 2             # snare on every 2nd step
OCEAN_DRUMLESS = 0.8        # no drums where at least this share of the ground is water
VEL_MIN = 50                # quietest note (1-127), raise it if quiet places disappear

# Fixed global ranges, so a place sounds the same no matter where the square started
# (depth is metres below sea level, pressure is hPa)
BOUNDS = {
    "elev": (0, 4500),
    "lst": (-20, 50),
    "vv": (-25, -5),
    "cloud": (0, 100),
    "night": (0, 60),
    "evi": (0, 1),
    "depth": (0, 6000),
    "pres": (985, 1030),
}
NODATA = -9999.0
BANDS = ("elev", "lst", "vv", "cloud", "night", "evi", "lc", "depth", "pres")
LEGACY_BANDS = BANDS[:7]    # grids baked before depth and pressure were added

# Land cover class -> (instrument, General MIDI program, octave shift)
INSTRUMENTS = {
    1: ("Cello", 42, 0),
    2: ("Viola", 41, 0),
    3: ("Contrabassoon", 70, -24),
    4: ("Oboe", 68, 0),
    5: ("Bassoon", 70, -12),
    6: ("French Horn", 60, 0),
    7: ("Clarinet", 71, 0),
    8: ("Flute", 73, 12),
    9: ("Violin", 40, 12),
    10: ("Saxophone", 66, 0),
    11: ("Harp", 46, 0),
    12: ("Piano", 0, 0),
    13: ("Synthesizer", 80, 0),
    14: ("Trumpet", 56, 0),
    15: ("Tuba", 58, -24),
    16: ("Trombone", 57, -12),
    17: ("String Bass", 43, -24),
}

# Which layer sets the loudness of each class: "evi", "night", or a fixed number from 0 to 1
DRIVER = {}
for land_class in range(1, 12):
    DRIVER[land_class] = "evi"
DRIVER[12] = "night"
DRIVER[13] = "night"
for land_class in (14, 15, 16):
    DRIVER[land_class] = "evi"
DRIVER[17] = "pres"

# Radar brightness bins -> extra drum (General MIDI percussion note)
SAR_EDGES = [0.21, 0.41, 0.61, 0.71, 0.81, 0.91, 0.99]
SAR_PERC = [None, ("Bass Drum", 35), ("Cymbals", 49), ("Hi-hat", 42),
            ("Tom Drum", 45), ("Bongo", 60), ("Claps", 39), ("Cowbell", 56)]
SNARE = 38

LC_NAMES = {
    1: "Evergreen needleleaf forest", 2: "Evergreen broadleaf forest",
    3: "Deciduous needleleaf forest", 4: "Deciduous broadleaf forest", 5: "Mixed forest",
    6: "Closed shrubland", 7: "Open shrubland", 8: "Woody savanna", 9: "Savanna",
    10: "Grassland", 11: "Permanent wetland", 12: "Cropland", 13: "Urban",
    14: "Cropland/natural mosaic", 15: "Snow and ice", 16: "Barren", 17: "Water",
}


# Earth Engine
def init_ee(project=EE_PROJECT):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def half_extent(lat, km=PATCH_KM):
    """Half the width of the square in degrees (lat, lon) at this latitude"""
    half_lat = km / 2 / 111.32
    half_lon = half_lat / max(math.cos(math.radians(lat)), 0.05)
    return half_lat, half_lon


def _band(image, name):
    """Return an image with exactly one band called `name`.
    The mean of an empty collection (open ocean, or no Sentinel pass that year) has no bands,
    which breaks later math ("Image.multiply: ... Got 0 and 1") or drops the layer (KeyError 'vv').
    In that case use a fully masked band instead."""
    import ee
    image = ee.Image(image)
    blank = ee.Image.constant(0).updateMask(ee.Image.constant(0)).rename(name)
    has_bands = image.bandNames().size().gt(0)
    return ee.Image(ee.Algorithms.If(has_bands, image.select([0]).rename(name), blank))


def build_layers(region, hires=True, months=MONTHS, ocean=False):
    """Build every layer as a single band ee.Image (masked where there is no data):
    elev, lst (degC), vv (dB), cloud (%), night, evi, lc, depth (m below sea level), pres (hPa).

    region : any ee.Geometry
    hires  : True takes cloud cover from Sentinel-2 (10 m, slow),
             False takes it from MODIS daily cloud state (1 km, fast)
    ocean  : True for tiles that are open water, only returns the layers that matter there
             (sea surface temperature, cloud, depth, pressure, land cover)"""
    import ee

    # Only keep the chosen months
    if months:
        month_filters = [ee.Filter.calendarRange(m, m, "month") for m in months]
        month_filter = ee.Filter.Or(*month_filters)
    else:
        month_filter = None

    def seasonal(date_filter):
        if month_filter is None:
            return date_filter
        return ee.Filter.And(date_filter, month_filter)

    year_filter = seasonal(ee.Filter.date(f"{YEAR}-01-01", f"{YEAR + 1}-01-01"))

    def latest(collection, band):
        # Annual products: take the newest valid pixel from the last 3 years so one empty
        # newest image can't blank the whole layer
        images = ee.ImageCollection(collection).select(band)
        images = images.sort("system:time_start", False).limit(3)
        return images.sort("system:time_start").mosaic()

    # Sea surface temperature (0.25 degrees, ocean only)
    sst = (ee.ImageCollection("NOAA/CDR/OISST/V2_1").filter(year_filter).select("sst")
           .mean().multiply(0.01).rename("lst"))

    # Cloud cover
    if hires:
        # Sentinel-2 scene classes 3, 8, 9, 10, 11 are shadow, cloud, cirrus and snow/ice.
        # Mapping them to 1 and everything else to 0 makes the mean the cloud fraction.
        def classify_clouds(image):
            return image.select("SCL").remap([3, 8, 9, 10, 11], [1, 1, 1, 1, 1], 0)

        cloud_mean = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
                      .filter(year_filter)
                      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
                      .map(classify_clouds).mean())
    else:
        # MODIS state_1km bits 0-1: 1 = cloudy, 2 = mixed
        def modis_cloud(image):
            state = image.select("state_1km").bitwiseAnd(3)
            return state.eq(1).Or(state.eq(2))

        cloud_mean = (ee.ImageCollection("MODIS/061/MOD09GA").filterBounds(region)
                      .filter(year_filter).map(modis_cloud).mean())
    cloud = _band(cloud_mean, "cloud").multiply(100)

    # Ocean depth from ETOPO1 (1.9 km): bedrock height is negative under the sea, so flip it,
    # and use 0 on land
    depth = (ee.Image("NOAA/NGDC/ETOPO1").select("bedrock").multiply(-1)
             .max(ee.Image.constant(0)).rename("depth"))

    # Sea level pressure from ERA5 (28 km pixels): changes between ocean regions, not within one
    pressure_dates = seasonal(ee.Filter.date(f"{ERA_YEAR}-01-01", f"{ERA_YEAR + 1}-01-01"))
    pres = (ee.ImageCollection("ECMWF/ERA5/DAILY").filter(pressure_dates)
            .select("mean_sea_level_pressure").mean().divide(100).rename("pres"))

    # Land cover
    lc = latest("MODIS/061/MCD12Q1", "LC_Type1").rename("lc")

    if ocean:
        return {"lst": sst, "cloud": cloud, "depth": depth, "pres": pres, "lc": lc}

    # Elevation
    alos = ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1").select("DSM")
    elev = alos.mosaic().setDefaultProjection(alos.first().projection()).rename("elev")

    # Land surface temperature, with sea surface temperature filling in the ocean
    lst_mean = (ee.ImageCollection("MODIS/061/MYD11A1").filterBounds(region).filter(year_filter)
                .select("LST_Day_1km").mean())
    lst_land = _band(lst_mean, "lst").multiply(0.02).subtract(273.15)
    lst = lst_land.unmask(sst, False)

    # Radar (Sentinel-1 VV), with the noisy edge pixels dropped and a light despeckle
    def mask_edge(image):
        return image.updateMask(image.gt(-30))

    vv_mean = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region).filter(year_filter)
               .filter(ee.Filter.eq("instrumentMode", "IW"))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
               .select("VV").map(mask_edge).mean())
    vv = _band(vv_mean, "vv").focalMean(1)

    # Night lights and vegetation
    night = latest("NOAA/VIIRS/DNB/ANNUAL_V22", "average").rename("night")
    evi = latest("LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL_EVI", "EVI").rename("evi")

    return {"elev": elev, "lst": lst, "vv": vv, "cloud": cloud, "night": night, "evi": evi,
            "lc": lc, "depth": depth, "pres": pres}


def build_stack(region, hires=True, months=MONTHS):
    """Every layer stacked into one float ee.Image"""
    import ee
    layers = build_layers(region, hires, months)
    return ee.Image.cat([layers[name] for name in BANDS]).toFloat()


LAST_FETCH = {}   # timings and errors from the most recent Earth Engine fetch


def fetch_patch(lat, lon, km=PATCH_KM, hires=None):
    """Fetch a PATCH_PX x PATCH_PX square as {band: 2D float array} (NaN = no data).
    Each layer is its own request and they run in parallel. A layer that fails becomes
    NaN instead of stopping the song. hires defaults to True for squares of 20 km or less."""
    import ee
    from concurrent.futures import ThreadPoolExecutor

    if hires is None:
        hires = km <= 20
    lat = float(np.clip(lat, -85, 85))
    half_lat, half_lon = half_extent(lat, km)
    grid = {
        "dimensions": {"width": PATCH_PX, "height": PATCH_PX},
        "affineTransform": {"scaleX": 2 * half_lon / PATCH_PX, "shearX": 0,
                            "translateX": lon - half_lon,
                            "shearY": 0, "scaleY": -2 * half_lat / PATCH_PX,
                            "translateY": lat + half_lat},
        "crsCode": "EPSG:4326",
    }
    layers = build_layers(ee.Geometry.Point([lon, lat]), hires)
    seconds, errors = {}, {}

    def fetch_one(name):
        start = time.perf_counter()
        try:
            array = ee.data.computePixels({"expression": layers[name].unmask(NODATA).toFloat(),
                                           "fileFormat": "NUMPY_NDARRAY", "grid": grid})
            field_name = name if name in array.dtype.names else array.dtype.names[0]
            values = np.array(array[field_name], dtype=np.float32)
            values[values <= NODATA + 1] = np.nan
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            values = np.full((PATCH_PX, PATCH_PX), np.nan, np.float32)
        seconds[name] = time.perf_counter() - start
        return name, values

    start = time.perf_counter()
    with ThreadPoolExecutor(min(len(BANDS), FETCH_THREADS)) as pool:
        patch = dict(pool.map(fetch_one, BANDS))
    LAST_FETCH.clear()
    LAST_FETCH.update(seconds=time.perf_counter() - start, layers=seconds, errors=errors)

    if len(errors) == len(BANDS):
        # Nothing worked, so it's a login, quota or network problem
        raise RuntimeError(next(iter(errors.values())))
    if errors:
        # Tells Source not to cache a square with missing layers
        patch["_failed"] = sorted(errors)
    return patch


def synthetic_patch(lat, lon, px=PATCH_PX):
    """Made up data for each location, the same every time. For testing without Earth Engine."""
    rng = np.random.default_rng(abs(int(lat * 100) * 100003 + int(lon * 100)))

    def blob(k=6):
        grid = rng.random((k, k))
        x = np.linspace(0, k - 1, px)
        grid = np.array([np.interp(x, np.arange(k), row) for row in grid])
        return np.array([np.interp(x, np.arange(k), col) for col in grid.T]).T

    elev_raw, cloud_raw = blob(), blob()
    lst = 30 - abs(lat) * 0.5 - elev_raw * 20 + blob() * 5
    vv = -22 + blob() * 16
    night = blob() ** 3 * 60
    evi = blob() * 0.8
    lc = np.clip(1 + (blob() * 17).astype(int), 1, 17).astype(float)

    sea = elev_raw < 0.22    # the lowest ground is sea
    return {
        "elev": np.where(sea, 0, elev_raw * 3000),
        "lst": lst,
        "vv": vv,
        "cloud": cloud_raw * 100,
        "night": night,
        "evi": evi,
        "lc": np.where(sea, 17.0, lc),
        "depth": np.where(sea, (0.22 - elev_raw) / 0.22 * 5500, 0.0),
        "pres": 1013 + (blob() - 0.5) * 40,
    }


# Where the squares come from
@dataclass
class Source:
    """Where squares of data come from. `km` is the width of the square this source plays."""
    name: str
    km: float
    fetch: object            # function (lat, lon) -> {band: 2D float array}
    snap: float = 0.0        # snap centres to this fraction of a square (for slow sources)
    _cache: dict = field(default_factory=dict, repr=False)

    def center(self, lat, lon):
        """The centre that is really sampled. Draw this square, not the raw cursor position."""
        if not self.snap:
            return lat, lon
        step = self.km * self.snap / 111.32
        return round(lat / step) * step, round(lon / step) * step

    def patch(self, lat, lon):
        lat, lon = self.center(lat, lon)
        key = (round(lat, 6), round(lon, 6))
        if self.snap and key in self._cache:
            return self._cache[key]
        patch = self.fetch(lat, lon)
        failed = patch.pop("_failed", None)   # a layer failed: use the square but try again next time
        if self.snap and not failed:
            self._cache[key] = patch
        return patch


def ee_source(km=PATCH_KM, hires=None):
    return Source("earth-engine", km, lambda lat, lon: fetch_patch(lat, lon, km, hires), snap=0.25)


def demo_source(km=PATCH_KM):
    return Source("demo", km, synthetic_patch)


def _dq(band, plane):
    """uint8 plane -> float. 0 = no data, 1-255 spans BOUNDS, land cover keeps its class numbers."""
    values = plane.astype(np.float32)
    if band != "lc":
        low, high = BOUNDS[band]
        values = low + (values - 1) / 254 * (high - low)
    values[plane == 0] = np.nan
    return values


def dequantize(stack, bands=BANDS):
    return {band: _dq(band, plane) for band, plane in zip(bands, stack)}


def open_grid(path):
    """Open a grid made by bake_grid.py. Returns (meta, {band: 2D uint8 memmap}).
    `path` is 'data/world.u8' or 'data/world', the json file next to it says where the layers are.
    Works with the current one-file-per-layer grids and the older single file grids.
    Layers that weren't baked are just missing."""
    import json
    base = os.path.splitext(path)[0]
    meta = json.load(open(base + ".json"))
    height, width = meta["height"], meta["width"]

    planes = {}
    if meta.get("layout") == "bands":
        for band in meta["bands"]:
            filename = f"{base}.{band}.u8"
            if os.path.exists(filename):
                planes[band] = np.memmap(filename, np.uint8, "r", shape=(height, width))
    else:
        # Older grids kept every layer in one file
        names = meta.get("bands", list(LEGACY_BANDS))
        stack = np.memmap(base + ".u8", np.uint8, "r", shape=(len(names), height, width))
        for i, band in enumerate(names):
            planes[band] = stack[i]
    return meta, planes


def grid_source(path, km, name=None):
    """A source that reads a baked grid. Only needs numpy, no Earth Engine and no internet."""
    meta, planes = open_grid(path)
    height, width, res = meta["height"], meta["width"], meta["res"]
    west, north = meta["bbox"][0], meta["bbox"][3]

    def fetch(lat, lon):
        half_lat, half_lon = half_extent(lat, km)
        fraction = (np.arange(PATCH_PX) + 0.5) / PATCH_PX
        rows = np.floor((north - (lat + half_lat - fraction * 2 * half_lat)) / res).astype(int)
        cols = np.floor((lon - half_lon + fraction * 2 * half_lon - west) / res).astype(int)

        # Anything outside the grid is "no data"
        inside = ((rows >= 0) & (rows < height))[:, None] & ((cols >= 0) & (cols < width))[None, :]
        rows = np.clip(rows, 0, height - 1)[:, None]
        cols = np.clip(cols, 0, width - 1)[None, :]

        patch = {}
        for band, plane in planes.items():
            values = np.where(inside, plane[rows, cols], 0).astype(np.uint8)
            patch[band] = _dq(band, values)
        return patch

    return Source(name or os.path.basename(path), km, fetch)


def diagnose(patch):
    """Print what each layer holds and which land cover classes (so which instruments) are in a square"""
    for band in BANDS:
        values = patch.get(band)
        if values is None:
            print(f"{band:6} (not in this grid)")
            continue
        valid = ~np.isnan(values)
        if valid.any():
            value_range = f"  range {values[valid].min():.3g} .. {values[valid].max():.3g}"
        else:
            value_range = ""
        print(f"{band:6} {100 * valid.mean():5.1f}% valid{value_range}")

    lc = patch.get("lc")
    if lc is not None and (~np.isnan(lc)).any():
        classes, counts = np.unique(lc[~np.isnan(lc)].astype(int), return_counts=True)
        for land_class, count in sorted(zip(classes, counts), key=lambda pair: -pair[1]):
            instrument = INSTRUMENTS.get(int(land_class), ("(not played)",))[0]
            print(f"  class {land_class:2d} {LC_NAMES.get(int(land_class), '?'):28} "
                  f"{100 * count / counts.sum():5.1f}%  -> {instrument}")


# Turning pixels into music
def _norm(values, band):
    """Scale to 0-1 using the fixed range for this layer"""
    low, high = BOUNDS[band]
    return np.clip((np.asarray(values, np.float32) - low) / (high - low), 0, 1)


def _cols(values_2d, steps=STEPS, default=0.5):
    """Average each step's columns (ignoring NaN). Gaps are filled in from their neighbours,
    and if everything is NaN every step gets `default`."""
    averages = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for columns in np.array_split(np.arange(values_2d.shape[1]), steps):
            averages.append(np.nanmean(values_2d[:, columns]))
    averages = np.array(averages)
    valid = ~np.isnan(averages)
    if not valid.any():
        return np.full(steps, default)
    return np.interp(np.arange(steps), np.flatnonzero(valid), averages[valid])


@dataclass
class Song:
    events: list     # (beat, 1 = note on / 0 = note off, channel, note, velocity), in order
    voices: dict     # channel -> (name, General MIDI program), channel 9 = drums
    bpm: float
    beats: float
    info: str = ""
    bars: dict = field(init=False, repr=False)

    def __post_init__(self):
        # Group the events by bar so the player can fetch one bar at a time
        self.bars = defaultdict(list)
        for event in self.events:
            self.bars[int(event[0] // 4)].append(event)


def sonify(patch):
    """One square of data -> a Song"""
    # A layer that is missing completely becomes all NaN
    reference = next(iter(patch.values()))
    patch = {band: patch.get(band, np.full(reference.shape, np.nan, np.float32)) for band in BANDS}

    gaps = []
    for label, band in (("temperature", "lst"), ("radar", "vv"), ("cloud", "cloud")):
        if np.isnan(patch[band]).all():
            gaps.append(label)
    if np.isnan(patch["lc"]).all():
        gaps.append("land cover (all-piano fallback)")

    # Pitch
    # Mountains and ocean trenches both pull the pitch down. Sea pixels use ocean depth and
    # land pixels use elevation (a pixel only counts as sea if land cover agrees, so ice sheets
    # and lakes stay land).
    land_cover, depth = patch["lc"], np.nan_to_num(patch["depth"])
    sea = (depth > 1) & (np.isnan(land_cover) | (land_cover == 17))
    height = np.where(sea, _norm(depth, "depth"), _norm(np.nan_to_num(patch["elev"]), "elev"))
    low_ground = 1 - _cols(height)
    warmth = _cols(_norm(patch["lst"], "lst"))
    pitch = ((W_ELEV * low_ground * low_ground + W_TEMP * warmth * warmth)
             / (W_ELEV * low_ground + W_TEMP * warmth + 1e-9))

    # The place sets the register and the square sets the shape of the tune
    middle = pitch.mean()
    spread = max(np.ptp(pitch), 0.15)
    pitch = np.clip(middle + (pitch - middle) / spread * CONTOUR, 0, 1)
    note_index = np.round(pitch * (len(SCALE_NOTES) - 1)).astype(int)

    # Rhythm: more cloud = more notes per step and a faster tempo
    cloudiness = _cols(_norm(patch["cloud"], "cloud"), default=0.35)
    notes_per_step = 1 + (cloudiness * 3.999).astype(int)
    bpm = round(BPM_RANGE[0] + cloudiness.mean() * (BPM_RANGE[1] - BPM_RANGE[0]))

    # Percussion: radar brightness picks the extra drum
    if "radar" in gaps:
        radar_bin = np.zeros(STEPS, int)    # no radar: just the snare, no made up drum
    else:
        radar_bin = np.digitize(_cols(_norm(patch["vv"], "vv")), SAR_EDGES)

    # Instruments and volume: how much of each step is each land cover class,
    # and the layer that sets how loud it is
    land_cover = np.nan_to_num(patch["lc"]).astype(int)
    loudness = {
        "evi": np.nan_to_num(_norm(patch["evi"], "evi"), nan=0.3),
        "night": np.nan_to_num(_norm(patch["night"], "night")),
        # low pressure = louder, but never faint
        "pres": np.nan_to_num(0.5 + 0.4 * (1 - _norm(patch["pres"], "pres")), nan=0.65),
    }
    share = np.zeros((18, STEPS))
    volume = np.zeros((18, STEPS))
    for step, columns in enumerate(np.array_split(np.arange(land_cover.shape[1]), STEPS)):
        step_classes = land_cover[:, columns]
        for land_class in np.unique(step_classes[(step_classes >= 1) & (step_classes <= 17)]):
            mask = step_classes == land_class
            share[land_class, step] = mask.mean()
            driver = DRIVER[land_class]
            if isinstance(driver, float):
                volume[land_class, step] = driver
            else:
                volume[land_class, step] = loudness[driver][:, columns][mask].mean()

    no_land_cover = share.sum() == 0
    if no_land_cover:
        # No land cover data at all: play piano
        share[12], volume[12] = 1.0, 0.5

    # The 8 most common classes get a channel each (channel 9 is the drums)
    ranked = [c for c in np.argsort(-share.sum(1)) if share[c].sum() > 0][:8]
    channel_of = {int(land_class): channel for channel, land_class in enumerate(ranked)}

    # Write the notes
    events, drums_played = [], set()
    for step in range(STEPS):
        start = step * STEP_BEATS

        # Classes playing in this step, biggest share first
        here = [c for c in channel_of if share[c, step] >= MIN_SHARE]
        here.sort(key=lambda c: -share[c, step])

        for rank, land_class in enumerate(here[:MAX_VOICES]):
            octave = INSTRUMENTS[land_class][2]
            # Each extra voice sits a third below the lead
            note = int(np.clip(SCALE_NOTES[max(note_index[step] - 2 * rank, 0)] + octave, 24, 108))
            velocity = int(np.clip(VEL_MIN + (125 - VEL_MIN) * volume[land_class, step]
                                   * (0.6 + 0.4 * share[land_class, step]), 1, 127))
            # The lead repeats its note, the other voices hold
            repeats = int(notes_per_step[step]) if rank == 0 else 1
            for i in range(repeats):
                note_start = start + i * STEP_BEATS / repeats
                note_end = note_start + STEP_BEATS / repeats * GATE
                channel = channel_of[land_class]
                events.append((note_start, 1, channel, note, velocity))
                events.append((note_end, 0, channel, note, 0))

        # Drums, but not over open water (just the deep strings there)
        drums = []
        if share[17, step] < OCEAN_DRUMLESS:
            if step % SNARE_EVERY == SNARE_EVERY - 1:
                drums.append(SNARE)
                drums_played.add("Snare")
            if SAR_PERC[radar_bin[step]]:
                drums.append(SAR_PERC[radar_bin[step]][1])
                drums_played.add(SAR_PERC[radar_bin[step]][0])
        for drum in drums:
            events.append((start, 1, 9, drum, 100))
            events.append((start + 0.25, 0, 9, drum, 0))

    # Note offs go before note ons that land on the same beat
    events.sort(key=lambda e: (e[0], e[1]))

    # The text shown on screen
    totals = share.sum(1) / STEPS
    if no_land_cover:
        mix = "NO DATA"
    else:
        top = [f"{LC_NAMES[c]} {100 * totals[c]:.0f}%" for c in np.argsort(-totals)[:3]
               if totals[c] >= 0.02]
        mix = ", ".join(top)

    voices = {channel: (INSTRUMENTS[c][0], INSTRUMENTS[c][1]) for c, channel in channel_of.items()}
    voices[9] = ("Percussion", 0)

    instrument_names = ", ".join(v[0] for channel, v in voices.items() if channel != 9)
    drum_names = ", ".join(sorted(drums_played, key=lambda name: name != "Snare")) or "none"
    info = f"{bpm} BPM | ground: {mix or 'unknown'} | {instrument_names} | drums: {drum_names}"
    if gaps:
        info += f" | no data: {', '.join(gaps)}"
    return Song(events, voices, bpm, STEPS * STEP_BEATS, info)


# Writing and playing MIDI
def _vlq(number):
    """MIDI variable length number"""
    out = [number & 0x7F]
    while number >> 7:
        number >>= 7
        out.append((number & 0x7F) | 0x80)
    return bytes(reversed(out))


def to_midi_bytes(song, ppq=480):
    """A Standard MIDI File (format 1): a tempo track plus one named track per instrument"""

    def chunk(tag, body):
        return tag + struct.pack(">I", len(body)) + body

    end_of_track = _vlq(0) + b"\xFF\x2F\x00"
    tempo = int(60e6 / song.bpm).to_bytes(3, "big")
    tracks = [chunk(b"MTrk", _vlq(0) + b"\xFF\x51\x03" + tempo
                    + _vlq(0) + b"\xFF\x58\x04\x04\x02\x18\x08" + end_of_track)]

    by_channel = defaultdict(list)
    for beat, on, channel, note, velocity in song.events:
        by_channel[channel].append((round(beat * ppq), on, note, velocity))

    for channel, events in sorted(by_channel.items()):
        name, program = song.voices[channel]
        body = _vlq(0) + b"\xFF\x03" + _vlq(len(name)) + name.encode()
        if channel != 9:
            body += _vlq(0) + bytes([0xC0 | channel, program])
        now = 0
        for tick, on, note, velocity in sorted(events, key=lambda e: (e[0], e[1])):
            status = (0x90 if on else 0x80) | channel
            body += _vlq(tick - now) + bytes([status, note, velocity])
            now = tick
        # Pad the track to the exact loop length
        body += _vlq(max(0, round(song.beats * ppq) - now)) + b"\xFF\x2F\x00"
        tracks.append(chunk(b"MTrk", body))

    return chunk(b"MThd", struct.pack(">HHH", 1, len(tracks), ppq)) + b"".join(tracks)


class LoopPlayer:
    """Loops a Song on a MIDI output. A new song swaps in at the next bar line, at the same
    place in the loop, so moving the square changes the music instead of restarting it.

    Pass `out` to use your own synth (anything with send_msg(kind, **fields), like kiosk.py does),
    or `port` to use a MIDI port through mido."""

    def __init__(self, port=None, virtual=False, out=None):
        if out is not None:
            self.mido, self.out = None, out
        else:
            import mido
            self.mido = mido
            if port:
                self.out = mido.open_output(port, virtual=virtual)
            else:
                self.out = mido.open_output()
        self._next = None
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def set_song(self, song):
        self._next = song

    def close(self):
        self._stop.set()

    __call__ = set_song

    def _msg(self, kind, **fields):
        if self.mido is None:
            self.out.send_msg(kind, **fields)
        else:
            self.out.send(self.mido.Message(kind, **fields))

    def _load(self, song):
        # Silence everything, then set the instrument on each channel
        for channel in range(16):
            self._msg("control_change", channel=channel, control=123, value=0)
        for channel, (_, program) in song.voices.items():
            if channel != 9:
                self._msg("program_change", channel=channel, program=program)

    def _sleep_until(self, when):
        while not self._stop.is_set():
            remaining = when - time.perf_counter()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.005))

    def _run(self):
        song, bar, bar_start = None, 0, time.perf_counter()
        while not self._stop.is_set():
            if self._next is not None:
                song, self._next = self._next, None
                self._load(song)
            if song is None:
                time.sleep(0.05)
                bar_start = time.perf_counter()
                continue

            seconds_per_beat = 60.0 / song.bpm
            bar_in_loop = bar % max(1, math.ceil(song.beats / 4))
            for beat, on, channel, note, velocity in song.bars.get(bar_in_loop, []):
                self._sleep_until(bar_start + (beat - 4 * bar_in_loop) * seconds_per_beat)
                self._msg("note_on" if on else "note_off", channel=channel, note=note,
                          velocity=velocity)
            bar_start += 4 * seconds_per_beat
            self._sleep_until(bar_start)
            bar += 1


# Notebook (Colab) players
# The kiosk doesn't use these. They are here so the old notebook workflow still works.
_PLAYER_HTML = """<script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.5.0"></script>
<div style="font:13px sans-serif;margin:4px 0">{info}</div>
<midi-player src="data:audio/midi;base64,{b64}" sound-font loop></midi-player>"""


class NotebookSink:
    """Shows a MIDI player in the notebook page (press play after each move).
    For smooth live changes run on your own computer with LoopPlayer instead."""

    def __init__(self):
        import ipywidgets
        from IPython.display import display
        self.out = ipywidgets.Output()
        display(self.out)

    def __call__(self, song):
        from IPython.display import HTML, clear_output, display
        b64 = base64.b64encode(to_midi_bytes(song)).decode()
        with self.out:
            clear_output(wait=True)
            display(HTML(_PLAYER_HTML.format(info=song.info, b64=b64)))


def _repeat(song, times):
    """The same song played `times` in a row"""
    events = []
    for i in range(times):
        for beat, *rest in song.events:
            events.append((beat + i * song.beats, *rest))
    return Song(events, song.voices, song.bpm, song.beats * times, song.info)


def beep_test():
    """If you can't hear this, the problem is the browser or Colab audio, not the soundscape code"""
    from IPython.display import Audio, display
    t = np.linspace(0, 1, 22050, endpoint=False)
    display(Audio(0.3 * np.sin(2 * np.pi * 440 * t), rate=22050))


def install_audio():
    """Colab: install FluidSynth and a General MIDI soundfont. Run once per runtime."""
    import subprocess
    subprocess.run("apt-get -qq update && apt-get -qq install -y fluidsynth fluid-soundfont-gm",
                   shell=True, check=True)


class AudioSink:
    """Notebook sink: renders each song to a WAV with FluidSynth (real General MIDI sounds)
    and updates one audio player on the page. The caption shows the length and loudness, so
    a silent render is easy to spot. Run install_audio() once first."""
    SOUNDFONT = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

    def __init__(self, loops=3, rate=22050):
        import shutil
        import ipywidgets
        from IPython.display import display
        if not shutil.which("fluidsynth"):
            raise RuntimeError("FluidSynth not found: run install_audio() first")
        if not os.path.exists(self.SOUNDFONT):
            raise RuntimeError(f"General-MIDI soundfont missing: {self.SOUNDFONT}")
        self.loops, self.rate = loops, rate
        self.html = ipywidgets.HTML("<i>move the square to start</i>")
        display(self.html)

    def __call__(self, song):
        import subprocess
        with open("song.mid", "wb") as f:
            f.write(to_midi_bytes(_repeat(song, self.loops)))
        result = subprocess.run(["fluidsynth", "-ni", "-F", "song.wav", "-r", str(self.rate),
                                 self.SOUNDFONT, "song.mid"],
                                capture_output=True, text=True, timeout=60,
                                stdin=subprocess.DEVNULL)   # don't wait on the notebook's input
        if result.returncode != 0:
            raise RuntimeError(f"fluidsynth failed: {result.stderr[-200:]}")

        # Stereo -> mono to make the file half the size
        with wave.open("song.wav") as w:
            channels, rate, width = w.getnchannels(), w.getframerate(), w.getsampwidth()
            frames = w.readframes(w.getnframes())
        if width != 2:
            raise RuntimeError(f"unexpected WAV sample width {width}")
        mono = np.frombuffer(frames, np.int16).reshape(-1, channels).mean(axis=1).astype(np.int16)
        peak = float(np.abs(mono).max()) / 32768 if len(mono) else 0.0

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(mono.tobytes())
        b64 = base64.b64encode(buffer.getvalue()).decode()

        warning = " - <b>SILENT RENDER: soundfont problem</b>" if peak < 0.01 else ""
        self.html.value = (f'<div style="font:13px sans-serif">{song.info}<br>'
                           f'audio {len(mono) / rate:.0f}s, loudness peak {peak:.2f}{warning}</div>'
                           f'<audio controls autoplay src="data:audio/wav;base64,{b64}"></audio>')


# Moving the square
class GlobePlayer:
    """Every move(lat, lon) becomes a song that is sent to `sink(song)`.

    sync=False (default): move() returns right away and a worker thread does the work.
        If you keep moving, only the newest position is used (good for a local MIDI player).
    sync=True: move() does the work before it returns. Colab needs this because it throws away
        page updates made after the click that caused them has finished.

    Use player.set_source(...) to switch data, like the world grid and the convention site grid."""

    def __init__(self, sink, source=None, say=print, fetch=None, sync=False):
        if fetch is not None:
            # Old style: GlobePlayer(sink, fetch=synthetic_patch)
            source = Source("custom", PATCH_KM, fetch)
        self.sink, self.say, self.sync = sink, say, sync
        self.source = source or ee_source()
        say(f"earth_soundscapes {__version__} ({os.path.abspath(__file__)})")
        self._target = None
        self._wake = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def move(self, lat, lon):
        self._target = (lat, lon)
        if self.sync:
            self._process()
        else:
            self._wake.set()

    def set_source(self, source):
        self.source = source
        if self._target:
            self.move(*self._target)

    def _run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            self._process()

    def _process(self):
        lat, lon = self._target
        try:
            self.say(f"({lat:.3f}, {lon:.3f}) loading...")
            start = time.perf_counter()
            patch = self.source.patch(lat, lon)
            took = time.perf_counter() - start

            # If this was a slow Earth Engine fetch, say which layer was slowest and any errors
            extra = ""
            if took > 0.5 and LAST_FETCH.get("layers"):
                slowest = max(LAST_FETCH["layers"], key=LAST_FETCH["layers"].get)
                extra = f" (slowest: {slowest} {LAST_FETCH['layers'][slowest]:.1f}s)"
                for band, message in LAST_FETCH["errors"].items():
                    extra += f" | LAYER ERROR {band}: {message[:90]}"

            song = sonify(patch)
            self.say("  sending to sound output...")
            self.sink(song)
            self.say(f"({lat:.3f}, {lon:.3f}) [{self.source.name}] {song.info} | data {took:.1f}s{extra}")
        except Exception as exc:
            self.say(f"error at ({lat:.3f}, {lon:.3f}): {exc}")


def launch_map(player, center=(37.87, -122.5), zoom=8, basemap=None, draggable=False):
    """A map for notebooks. Click anywhere, or use the arrow buttons (half a square per press),
    to move the square. The default basemap is Esri imagery because OpenStreetMap blocks
    notebook widgets.

    Two notebook quirks this works around:
    1. Colab drops page updates made after the click that caused them has finished, so the
       player runs in sync mode (the page is busy while a song renders).
    2. Dragging fires dozens of events a second and each would start a full render, so the
       marker can't be dragged unless you pass draggable=True."""
    import html
    import ipywidgets
    from ipyleaflet import Map, Marker, Rectangle, basemaps

    player.sync = True
    notebook_map = Map(center=center, zoom=zoom, scroll_wheel_zoom=True,
                       basemap=basemap or basemaps.Esri.WorldImagery)
    marker = Marker(location=center, draggable=draggable)
    square = []    # the red square on the map (one item)

    # A small status box showing the last few messages
    status = ipywidgets.HTML()
    lines = []

    def say(message):
        lines.append(message)
        del lines[:-6]
        status.value = ("<pre style='margin:0;font-size:12px;white-space:pre-wrap'>"
                        + html.escape("\n".join(lines)) + "</pre>")

    player.say = say
    say(f"earth_soundscapes {__version__}")

    def moved(_=None):
        lat, lon = marker.location
        # Draw the square that is really playing, not the raw marker position
        sample_lat, sample_lon = player.source.center(lat, lon)
        half_lat, half_lon = half_extent(sample_lat, player.source.km)
        new_square = Rectangle(bounds=((sample_lat - half_lat, sample_lon - half_lon),
                                       (sample_lat + half_lat, sample_lon + half_lon)),
                               color="red", weight=2, fill_opacity=0.1)
        # A new layer each move redraws reliably
        for old_square in square:
            notebook_map.remove_layer(old_square)
        square[:] = [new_square]
        notebook_map.add_layer(new_square)
        player.move(lat, lon)

    def nudge(up, right):
        def go(_):
            lat, lon = marker.location
            half_lat, half_lon = half_extent(lat, player.source.km)
            new_lat = float(np.clip(lat + up * half_lat, -85, 85))
            new_lon = (lon + right * half_lon + 180) % 360 - 180
            marker.location = (new_lat, new_lon)
        return go

    buttons = []
    for label, up, right in (("◀", 0, -1), ("▲", 1, 0), ("▼", -1, 0), ("▶", 0, 1)):
        button = ipywidgets.Button(description=label, layout=ipywidgets.Layout(width="64px"))
        button.on_click(nudge(up, right))
        buttons.append(button)

    def clicked(**event):
        if event.get("type") == "click":
            marker.location = event["coordinates"]

    marker.observe(moved, names="location")
    notebook_map.on_interaction(clicked)
    notebook_map.add_layer(marker)
    moved()

    page = ipywidgets.VBox([status, ipywidgets.HBox(buttons), notebook_map])
    page.redraw = moved    # call this after player.set_source(...)
    return page


def live_cli(player, lat, lon):
    """Move the square by typing in the terminal"""
    print("n/s/e/w = move one square | 'lat lon' = jump | q = quit")
    player.move(lat, lon)
    while True:
        command = input("> ").strip().lower()
        if command in ("q", "quit"):
            return
        half_lat, half_lon = half_extent(lat, player.source.km)
        try:
            if command in ("n", "s", "e", "w"):
                if command == "n":
                    lat += 2 * half_lat
                elif command == "s":
                    lat -= 2 * half_lat
                elif command == "e":
                    lon += 2 * half_lon
                else:
                    lon -= 2 * half_lon
            else:
                lat, lon = map(float, command.replace(",", " ").split())
        except ValueError:
            continue
        lat = float(np.clip(lat, -85, 85))
        lon = (lon + 180) % 360 - 180
        player.move(lat, lon)


def main():
    parser = argparse.ArgumentParser(description="Earth Observation Soundscapes")
    parser.add_argument("--lat", type=float, default=37.87)
    parser.add_argument("--lon", type=float, default=-122.5)
    parser.add_argument("--out", default="earth_song.mid")
    parser.add_argument("--demo", action="store_true", help="fake data, no Earth Engine")
    parser.add_argument("--grid", help="baked grid (.u8 from bake_grid.py), offline, no Earth Engine")
    parser.add_argument("--km", type=float, default=PATCH_KM, help="width of the square in km")
    parser.add_argument("--live", action="store_true", help="play on a MIDI port, move from the terminal")
    parser.add_argument("--port", help="MIDI output port name")
    parser.add_argument("--virtual", action="store_true", help="make a virtual port (macOS/Linux)")
    parser.add_argument("--ports", action="store_true", help="list MIDI output ports and quit")
    args, _ = parser.parse_known_args()    # ignore extra arguments (Jupyter adds -f kernel.json)

    if args.ports:
        import mido
        print("\n".join(mido.get_output_names()) or "no MIDI output ports found")
        return

    if args.grid:
        source = grid_source(args.grid, args.km)
    elif args.demo:
        source = demo_source(args.km)
    else:
        init_ee()
        source = ee_source(args.km)

    if args.live:
        live_cli(GlobePlayer(LoopPlayer(args.port, args.virtual), source), args.lat, args.lon)
    else:
        song = sonify(source.patch(args.lat, args.lon))
        with open(args.out, "wb") as f:
            f.write(to_midi_bytes(song))
        print(f"{os.path.abspath(args.out)}\n{song.info}")


# Only run from the command line, a notebook just imports the file
if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()
