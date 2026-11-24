# Earth Observation Soundscapes

This project transforms remote sensing datasets into music. Move a small square around a map of the
world and hear the satellite data translated: elevation, ocean depth,
temperature, vegetation, cloud, night lights, and radar become pitch, rhythm, instruments, and volume
with the song changing as you move.

It now runs independently with no Colab or live Earth Engine connection. The satellite data needs to be
downloaded once into a local grid, and a small web app plays the music through the
FluidSynth synthesizer.

## Project goals

- Convert geospatial raster datasets into music that responds instantly.
- Map environmental variables to musical elements (elevation to pitch, radar backscatter to percussion).
- Explore the artistic intersection of Earth science and sound design.

## Musical mapping

| Music element | Controlled by | Logic |
|---|---|---|
| Instruments | Land cover (MODIS IGBP, 17 classes) | Each class is one instrument; the most common classes in the square play together |
| Pitch | Elevation, ocean depth, land temperature | Higher ground gives lower notes, deeper sea gives lower notes, warmer gives higher; snapped to a pentatonic scale |
| Tempo and rhythm | Cloud cover | Clear is slow with sparse notes (80 BPM); cloudy is fast with dense notes (140 BPM) |
| Volume | Vegetation (EVI), night lights, sea-level pressure | Green areas boost instruments assigned to natural classes, bright night lights boost Piano and Synthesizer, low pressure boosts the ocean's String Bass |
| Percussion | Sentinel-1 radar (VV) | Rougher returns pick fuller drums; a snare backbeat plays; no data is only snare, and open ocean has no drums |

## Datasets

| Dataset | Used for |
|---|---|
| JAXA ALOS AW3D30 (elevation) | Pitch |
| NOAA ETOPO1 (bathymetry) | Pitch over water |
| MODIS land surface temperature, NOAA OISST (sea surface temperature) | Pitch |
| MODIS cloud state (world), Sentinel-2 cloud (site) | Tempo and rhythm |
| Sentinel-1 SAR (VV) | Percussion |
| MODIS land cover (IGBP) | Instrument assignment |
| Landsat annual EVI | Volume for natural classes |
| VIIRS night lights | Volume for Piano and Synthesizer |
| ERA5 sea-level pressure | Volume for ocean instruments |

## What's here

| File | Purpose |
|---|---|
| `earth_soundscapes.py` | Core: reads the grid, turns pixels into a song, plays it |
| `bake_grid.py` | One-time bake of the satellite layers into an offline grid (needs Earth Engine) |
| `kiosk.py` | The app: local web page with the map, layer switcher and sound |
| `start_soundscape.bat`, `stop_soundscape.bat` | Windows double-click launcher and stopper |

The original Colab notebook is kept in `legacy/`.

## Setup (Windows)

1. Install Python 3. In this folder, run `python -m venv .venv`, then `.venv\Scripts\activate`
   (if PowerShell blocks it: `Set-ExecutionPolicy -Scope Process RemoteSigned`), then
   `pip install -r requirements.txt`.
2. Download [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases) (Windows build) and a
   General MIDI soundfont such as GeneralUser GS, and unzip both into this folder. The kiosk
   searches this folder for them. Check the soundfont's license if you redistribute it.
3. Test the sound: `python kiosk.py --test-sound`. You should hear a chord, a bass note and a drum.

## Make the data (once)

Needs a Google Earth Engine account and project (`earthengine authenticate`, and set your project
with the `EE_PROJECT` environment variable).

```
python bake_grid.py world --bbox -125 35 -120 40 --out data   # small test region first
python bake_grid.py world --out worlddata                     # whole globe at ~1 km, many hours
python bake_grid.py site --lat LAT --lon LON --radius-km 100 --scale 500 --out sitedata
```

Re-running the same command resumes, and Ctrl+C saves progress. If you see rate-limit errors, add
`--workers 3`. The world grid is several GB.

## Run

```
python kiosk.py --world worlddata\world.u8 --open
python kiosk.py --world worlddata\world.u8 --site sitedata\site.u8 --open
```

Or edit and double-click `start_soundscape.bat`. Click or drag the square, use the arrow keys or WASD,
zoom with the wheel or +/-. The layer buttons show each satellite layer and explain what it does to
the music. Useful flags: `--volume`, `--idle SECONDS` (return home when idle), `--driver wasapi|dsound`.

## Next steps

[] Multi-seat table version: several squares, each with its own headphones
[] Touch support improvements (pinch-zoom)
[] Auto-start and recovery setup for unattended exhibition use

## Data credits

MODIS and VIIRS (NASA), Landsat (USGS), Copernicus Sentinel-1/2 (ESA), JAXA ALOS AW3D30,
NOAA OISST and ETOPO1, and ERA5 (Copernicus Climate Change Service, contains modified information).
Check each dataset's terms before reuse.

## License

- Code is licensed under the MIT License. See LICENSE.
- Music and audio output are licensed under [CC BY 4.0](https://github.com/arflynn28/Earth-Observation-Soundscapes/blob/main/music-license.txt).
