#!/usr/bin/env python3
"""kiosk.py - the standalone Earth Soundscapes app. No Colab, no Earth Engine, no internet.

    python kiosk.py --world worlddata/world.u8 --site sitedata/site.u8 --open
    python kiosk.py --test-sound                  # play a short chord to check FluidSynth works
    python kiosk.py --world data/world.u8 --no-sound   # silent preview (map and square only)

It serves a full screen map in your browser (drawn from the baked data itself, so no map
downloads) and plays the music through FluidSynth. Tap or click the map to move the square, drag
the square to slide it, drag anywhere else to pan, +/- or the mouse wheel to zoom, arrow keys or WASD
to move half a square. A new song swaps in at the next bar line, so the music changes smoothly
instead of restarting. After --idle seconds with no input it goes back to the start position.

Needs numpy, the FluidSynth program and a General MIDI soundfont (.sf2). Both are found
automatically if they are in this folder or up to three folders below it. To run the browser
in kiosk mode:
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --kiosk http://localhost:8765
"""
from __future__ import annotations

import argparse, atexit, glob, json, math, os, shutil, socket, struct, subprocess, sys, threading, time, webbrowser, zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import earth_soundscapes as es

CREDITS = ("Data: MODIS and VIIRS (NASA), Landsat (USGS), Copernicus Sentinel-1/2 (ESA), JAXA ALOS AW3D30, "
           "NOAA OISST and ETOPO1 (NCEI). Contains modified Copernicus Climate Change Service information "
           "(ERA5). Processed with Google Earth Engine.")


# Sound
class NullOut:
    """Silent output (--no-sound)"""

    def send_msg(self, kind, **fields):
        pass

    def set_volume(self, volume):
        pass

    def close(self):
        pass


def _under(*patterns):
    """Files that match the patterns in this folder and up to three levels of folders below it"""
    found = []
    for depth in range(4):
        prefix = "*/" * depth
        for pattern in patterns:
            for match in sorted(glob.glob(prefix + pattern)):
                # Skip anything inside the virtual environment
                if ".venv" not in match.replace("\\", "/").split("/"):
                    found.append(match)
    return found


def find_fluidsynth(explicit=None):
    exe = "fluidsynth.exe" if os.name == "nt" else "fluidsynth"
    candidates = [explicit, os.environ.get("FLUIDSYNTH"), shutil.which("fluidsynth")]
    candidates += [r"C:\Program Files\FluidSynth\bin\fluidsynth.exe", r"C:\tools\fluidsynth\bin\fluidsynth.exe",
                   "/usr/bin/fluidsynth", "/usr/local/bin/fluidsynth", "/opt/homebrew/bin/fluidsynth"]
    # Also look in this folder, like fluidsynth-v2.6.2-win10-x64-cpp11\bin\fluidsynth.exe
    candidates += _under(f"bin/{exe}", exe)
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


def find_soundfont(explicit=None):
    candidates = [explicit, os.environ.get("SOUNDFONT")]
    candidates += _under("*.sf2")    # like GeneralUser-GS\GeneralUser-GS.sf2
    candidates += ["/usr/share/sounds/sf2/FluidR3_GM.sf2", "/usr/share/soundfonts/FluidR3_GM.sf2"]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


_JOB = None


def _kill_with_me(process):
    """Windows only: put the process in a job object that ends when this program ends.
    That way FluidSynth is stopped even if the PowerShell window is closed or Python is killed
    (atexit doesn't run in those cases)."""
    global _JOB
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes as wt

        class _Basic(ctypes.Structure):
            _fields_ = [("a", ctypes.c_int64), ("b", ctypes.c_int64), ("LimitFlags", wt.DWORD),
                        ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("e", wt.DWORD),
                        ("f", ctypes.c_size_t), ("g", wt.DWORD), ("h", wt.DWORD)]

        class _Ext(ctypes.Structure):
            _fields_ = [("Basic", _Basic), ("io", ctypes.c_uint64 * 6), ("m1", ctypes.c_size_t),
                        ("m2", ctypes.c_size_t), ("m3", ctypes.c_size_t), ("m4", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.restype = ctypes.c_void_p
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
        kernel.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        job = kernel.CreateJobObjectW(None, None)
        info = _Ext()
        info.Basic.LimitFlags = 0x2000    # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        set_ok = kernel.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
        if job and set_ok and kernel.AssignProcessToJobObject(job, int(process._handle)):
            _JOB = job    # keep the handle open until we exit
    except Exception:
        pass


class FluidSynthOut:
    """Real time General MIDI synth. Runs the `fluidsynth` program and sends it shell commands
    (noteon / noteoff / prog / cc / gain) through its standard input. No MIDI ports needed."""
    MAX_GAIN = 1.2

    def __init__(self, soundfont, exe, driver=None, volume=0.65, log="fluidsynth.log"):
        command = [exe, "-n", "-g", f"{self.MAX_GAIN * volume:.2f}", "-r", "44100"]    # -n = no MIDI input
        if driver:
            command += ["-a", driver]
        command.append(soundfont)

        self._lock = threading.Lock()
        self.dead = False
        self._log = open(log, "w")
        self.p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=self._log, stderr=self._log,
                                  text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        _kill_with_me(self.p)
        time.sleep(1.0)
        if self.p.poll() is not None:
            self._log.flush()
            tail = open(log).read()[-600:]
            raise RuntimeError(f"FluidSynth exited immediately ({exe}). Its log:\n{tail}")
        atexit.register(self.close)

    def _write(self, line):
        if self.dead:
            return
        try:
            with self._lock:
                self.p.stdin.write(line + "\n")
                self.p.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self.dead = True

    def send_msg(self, kind, **fields):
        if kind == "note_on":
            self._write(f"noteon {fields['channel']} {fields['note']} {fields['velocity']}")
        elif kind == "note_off":
            self._write(f"noteoff {fields['channel']} {fields['note']}")
        elif kind == "program_change":
            self._write(f"prog {fields['channel']} {fields['program']}")
        elif kind == "control_change":
            self._write(f"cc {fields['channel']} {fields['control']} {fields['value']}")

    def set_volume(self, volume):
        self._write(f"gain {self.MAX_GAIN * volume:.2f}")

    def close(self):
        if self.p.poll() is None:
            self._write("quit")
            try:
                self.p.stdin.close()
                self.p.wait(timeout=3)
            except Exception:
                self.p.kill()


class PyFluidOut:
    """Backup synth that uses the pyfluidsynth package (pip install pyfluidsynth). It needs the
    FluidSynth library (DLL) to be findable. Used with --synth py."""
    MAX_GAIN = 1.2

    def __init__(self, soundfont, driver=None, volume=0.65):
        import fluidsynth
        self.fs = fluidsynth.Synth(gain=self.MAX_GAIN * volume)
        self.fs.start(driver=driver)
        sfid = self.fs.sfload(soundfont)
        for channel in range(16):
            bank = 128 if channel == 9 else 0
            self.fs.program_select(channel, sfid, bank, 0)

    def send_msg(self, kind, **fields):
        if kind == "note_on":
            self.fs.noteon(fields["channel"], fields["note"], fields["velocity"])
        elif kind == "note_off":
            self.fs.noteoff(fields["channel"], fields["note"])
        elif kind == "program_change":
            if fields["channel"] != 9:
                self.fs.program_change(fields["channel"], fields["program"])
        elif kind == "control_change":
            self.fs.cc(fields["channel"], fields["control"], fields["value"])

    def set_volume(self, volume):
        try:
            self.fs.setting("synth.gain", self.MAX_GAIN * volume)
        except Exception:
            pass

    def close(self):
        try:
            self.fs.delete()
        except Exception:
            pass


def make_backend(args):
    """Pick the sound output: silent, the FluidSynth program, or pyfluidsynth"""
    if args.no_sound:
        return NullOut()

    soundfont = find_soundfont(args.soundfont)
    if not soundfont:
        sys.exit("No soundfont found. Put a General-MIDI .sf2 file (for example GeneralUser.sf2) in this "
                 "folder, or pass --soundfont path\\to\\file.sf2")

    exe = find_fluidsynth(args.fluidsynth)
    if args.synth == "py" or (args.synth == "auto" and not exe):
        try:
            return PyFluidOut(soundfont, args.driver, args.volume)
        except Exception as exc:
            sys.exit(f"No FluidSynth program found and pyfluidsynth failed ({exc}). Install FluidSynth "
                     f"(unzip it next to kiosk.py, or pass --fluidsynth path\\to\\fluidsynth.exe).")
    if not exe:
        sys.exit("FluidSynth not found. Unzip it next to kiosk.py, or pass --fluidsynth path\\to\\fluidsynth.exe")

    print(f"synth: {exe}\nsoundfont: {soundfont}")
    try:
        return FluidSynthOut(soundfont, exe, args.driver, args.volume)
    except RuntimeError as exc:
        sys.exit(str(exc))


def test_sound(out):
    """Plays a C major chord, a bass note and a drum hit. You should hear all of it."""
    for channel, program in ((0, 0), (1, 43)):
        out.send_msg("program_change", channel=channel, program=program)
    out.send_msg("note_on", channel=1, note=36, velocity=110)
    for note in (60, 64, 67, 72):
        out.send_msg("note_on", channel=0, note=note, velocity=100)
        time.sleep(0.35)
    out.send_msg("note_on", channel=9, note=38, velocity=110)
    time.sleep(1.5)
    for note in (60, 64, 67, 72):
        out.send_msg("note_off", channel=0, note=note)
    out.send_msg("note_off", channel=1, note=36)
    time.sleep(0.8)


# Map tiles (drawn from the baked grid)
# Colour for each land cover class
LC_RGB = {1: (34, 89, 52), 2: (22, 110, 56), 3: (72, 110, 60), 4: (86, 140, 70), 5: (60, 120, 65),
          6: (140, 130, 80), 7: (176, 160, 110), 8: (150, 160, 80), 9: (190, 180, 90), 10: (170, 200, 110),
          11: (80, 160, 150), 12: (225, 205, 110), 13: (200, 70, 70), 14: (190, 190, 100),
          15: (240, 245, 250), 16: (190, 160, 130), 17: (30, 90, 150)}
LUT = np.zeros((256, 3), np.float32)
for land_class, colour in LC_RGB.items():
    LUT[land_class] = colour

# Shallow and deep water colours
SHALLOW = np.array([70, 140, 200], np.float32)
DEEP = np.array([8, 28, 80], np.float32)


def png_bytes(rgba):
    """Turn an (height, width, 4) uint8 array into PNG file bytes"""
    height, width, _ = rgba.shape
    raw = np.zeros((height, 1 + width * 4), np.uint8)
    raw[:, 1:] = rgba.reshape(height, width * 4)

    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 1)) + chunk(b"IEND", b""))


BLANK_TILE = png_bytes(np.zeros((256, 256, 4), np.uint8))

# The layers the map can show. "map" is the shaded land cover picture, the rest each paint one baked layer.
# id: (label, unit, colours from low to high)
LAYER_INFO = {
    "map": ("Map", "", None),
    "lc": ("Land cover", "", None),
    "elev": ("Elevation", "m", [(30, 90, 50), (190, 180, 90), (140, 100, 70), (250, 250, 250)]),
    "depth": ("Ocean depth", "m", [(120, 190, 230), (30, 80, 170), (8, 20, 70)]),
    "lst": ("Land temp", "\u00b0C", [(40, 60, 160), (90, 200, 220), (250, 240, 120), (230, 60, 40)]),
    "pres": ("Sea pressure", "hPa", [(70, 40, 140), (100, 150, 210), (240, 240, 240), (230, 150, 70)]),
    "cloud": ("Cloud cover", "%", [(20, 40, 70), (150, 170, 200), (250, 250, 255)]),
    "evi": ("Vegetation", "", [(120, 90, 60), (200, 200, 100), (40, 150, 60), (0, 80, 30)]),
    "night": ("Night lights", "nW", [(10, 10, 30), (150, 80, 40), (255, 210, 90), (255, 255, 235)]),
    "vv": ("Radar", "dB", [(10, 10, 20), (80, 100, 150), (240, 240, 250)]),
}

# What each layer does to the music (this matches sonify in earth_soundscapes.py)
LAYER_SOUND = {
    "map": ("Not a sound itself", "Just the picture: land cover shaded by elevation, ocean shaded by depth. "
            "Pick another layer to see what drives the music."),
    "lc": ("Instruments", "Each land-cover class is one instrument. The (up to) 8 most common classes inside "
           "the square play together."),
    "elev": ("Pitch", "Higher ground makes lower notes. Blended 50/50 with temperature. Over open water, "
             "ocean depth takes its place."),
    "depth": ("Pitch over water", "Deeper sea makes lower notes (String Bass range), so trenches sound deeper "
              "than shallow coasts."),
    "lst": ("Pitch", "Warmer ground makes higher notes. Blended 50/50 with elevation."),
    "cloud": ("Tempo and rhythm", "Clear sky: 80 BPM, one note per step. Overcast: 140 BPM, up to four notes "
              "per step."),
    "pres": ("Volume of water instruments", "Lower sea-level pressure makes the String Bass louder; high "
             "pressure makes it softer (never silent)."),
    "evi": ("Volume of nature instruments", "Greener vegetation makes forest, shrub, grass, wetland, snow and "
            "barren instruments louder."),
    "night": ("Volume of Piano and Synthesizer", "Brighter night lights make cropland (Piano) and urban "
              "(Synthesizer) louder."),
    "vv": ("Extra drums", "Radar roughness picks the drum: smooth surfaces silent, then bass drum, cymbals, "
           "hi-hat, tom, bongo, claps, cowbell as it gets brighter. A snare backbeat always plays, but "
           "there are no drums over open water."),
}


def _ramp(stops):
    """A few colours -> a 256 x 3 table that blends between them"""
    positions = np.linspace(0, 255, len(stops))
    colours = np.array(stops, np.float32)
    channels = [np.interp(np.arange(256), positions, colours[:, c]) for c in range(3)]
    return np.stack(channels, axis=1)


RAMPS = {name: _ramp(info[2]) for name, info in LAYER_INFO.items() if info[2]}


class Grid:
    """One baked grid: the source the music is played from, and the map tiles drawn from it"""

    def __init__(self, mode, path, km, label):
        self.mode, self.km, self.label = mode, km, label
        meta, self.planes = es.open_grid(path)
        self.H, self.W, self.res = meta["height"], meta["width"], meta["res"]
        self.bbox = meta["bbox"]    # west, south, east, north
        self.west, self.north = self.bbox[0], self.bbox[3]
        self.source = es.grid_source(path, km, label)
        self.max_zoom = int(min(10, math.ceil(math.log2(180 / (256 * self.res))) + 1))
        self.center = ((self.bbox[1] + self.bbox[3]) / 2, (self.bbox[0] + self.bbox[2]) / 2)
        self._cache = {}

        # "map" is always there, the other buttons only appear for layers that were baked
        self.layers = ["map"]
        for name in LAYER_INFO:
            if name != "map" and name in self.planes:
                self.layers.append(name)

    def layer_infos(self):
        """What the page needs to know about each layer"""
        infos = []
        for name in self.layers:
            label, unit, stops = LAYER_INFO[name]
            low, high = es.BOUNDS.get(name, (None, None))
            what, how = LAYER_SOUND[name]
            if stops:
                legend = stops
            else:
                legend = [] if name == "lc" else None
            info = {"id": name, "label": label, "unit": unit, "lo": low, "hi": high,
                    "what": what, "how": how, "stops": legend}
            if name == "lc":
                info["classes"] = []
                for land_class in sorted(es.LC_NAMES):
                    if land_class in LC_RGB and land_class in es.INSTRUMENTS:
                        info["classes"].append({"name": es.LC_NAMES[land_class],
                                                "rgb": list(LC_RGB[land_class]),
                                                "instr": es.INSTRUMENTS[land_class][0]})
            infos.append(info)
        return infos

    def tile(self, z, tx, ty, layer="map"):
        if layer not in self.layers:
            layer = "map"
        key = (layer, z, tx, ty)
        if key not in self._cache:
            if len(self._cache) > 3000:
                self._cache.clear()
            self._cache[key] = self._render(z, tx, ty, layer)
        return self._cache[key]

    def _render(self, z, tx, ty, layer="map"):
        size = 256
        degrees_per_pixel = 180 / (size * 2 ** z)
        lon = -180 + (tx * size + np.arange(size) + 0.5) * degrees_per_pixel
        lat = 90 - (ty * size + np.arange(size) + 0.5) * degrees_per_pixel

        # Which grid pixel each tile pixel falls on
        cols = np.floor((lon - self.west) / self.res).astype(int)
        rows = np.floor((self.north - lat) / self.res).astype(int)
        if not (((rows >= 0) & (rows < self.H)).any() and ((cols >= 0) & (cols < self.W)).any()):
            return BLANK_TILE
        inside = ((rows >= 0) & (rows < self.H))[:, None] & ((cols >= 0) & (cols < self.W))[None, :]
        rows = np.clip(rows, 0, self.H - 1)[:, None]
        cols = np.clip(cols, 0, self.W - 1)[None, :]

        def get(name):
            if name not in self.planes:
                return None
            return np.where(inside, self.planes[name][rows, cols], 0)

        lc = get("lc")
        if lc is None:
            return BLANK_TILE

        # The normal map: land cover colours, with deeper water darker and higher land lighter
        rgb = LUT[lc].copy()
        elev, depth = get("elev"), get("depth")
        water = lc == 17
        if depth is not None:
            t = np.clip((depth.astype(np.float32) - 1) / 254, 0, 1) ** 0.6
            rgb[water] = (SHALLOW + (DEEP - SHALLOW) * t[..., None])[water]
        if elev is not None:
            shade = 0.72 + 0.4 * np.clip((elev.astype(np.float32) - 1) / 254, 0, 1)
            rgb[~water] *= shade[..., None][~water]

        # Or paint one layer instead
        if layer == "lc":
            rgb = LUT[lc].copy()
        elif layer != "map":
            values = get(layer)
            if values is not None:
                rgb *= 0.3    # dim the map underneath so gaps in the layer are still readable
                has_data = values > 0
                rgb[has_data] = RAMPS[layer][np.clip(values[has_data].astype(int) - 1, 0, 255)]

        tile = np.zeros((size, size, 4), np.uint8)
        tile[..., :3] = np.clip(rgb, 0, 255)
        tile[..., 3] = np.where((lc > 0) & inside, 255, 0)
        return png_bytes(tile)


# App state
class App:
    def __init__(self, grids, player, home=None, idle=300, volume=0.65):
        self.grids, self.player, self.idle, self.volume = grids, player, idle, volume
        self.mode0 = next(iter(grids))    # the first grid is the start screen
        self.mode = self.mode0
        self.home = tuple(home) if home else None    # (lat, lon) on the first grid, otherwise the grid's centre
        self.lock = threading.RLock()
        self.last_move = time.time()
        self.state = {}
        self.move(None, None, self.mode0, home=True, user=False)
        threading.Thread(target=self._idle_loop, daemon=True).start()

    def home_of(self, mode):
        if self.home and mode == self.mode0:
            return self.home
        return self.grids[mode].center

    def move(self, lat, lon, mode=None, home=False, user=True):
        """Move the square, work out the new song and start playing it"""
        with self.lock:
            if mode not in self.grids:
                mode = self.mode
            grid = self.grids[mode]
            if home or lat is None or lon is None:
                lat, lon = self.home_of(mode)
            lat = float(np.clip(lat, -85, 85))
            lon = (float(lon) + 180) % 360 - 180

            # The square that is really sampled (snapped to the grid)
            sample_lat, sample_lon = grid.source.center(lat, lon)
            patch = grid.source.patch(sample_lat, sample_lon)
            half_lat, half_lon = es.half_extent(sample_lat, grid.km)
            state = {"mode": mode, "lat": sample_lat, "lon": sample_lon, "km": grid.km,
                     "bounds": [sample_lat - half_lat, sample_lon - half_lon,
                                sample_lat + half_lat, sample_lon + half_lon]}

            if np.isnan(patch["lc"]).all():
                # Off the baked area: keep playing the last song
                state["info"] = "No data here (outside the baked area)"
                state["nodata"] = True
            else:
                song = es.sonify(patch)
                self.player.set_song(song)
                state["info"] = song.info

            self.mode = mode
            state["t"] = time.time()
            self.state = state
            if user:
                self.last_move = time.time()
            return state

    def _idle_loop(self):
        """Go back to the start position when nobody has touched it for a while"""
        while True:
            if self.idle > 0:
                time.sleep(max(0.5, min(5, self.idle / 2)))
            else:
                time.sleep(5)
            if self.idle <= 0 or time.time() - self.last_move < self.idle:
                continue
            with self.lock:
                home_lat, home_lon = self.home_of(self.mode0)
                state = self.state
                away = (state.get("mode") != self.mode0
                        or abs(state.get("lat", 0) - home_lat) > 1e-6
                        or abs(state.get("lon", 0) - home_lon) > 1e-6)
                if away:
                    self.move(None, None, self.mode0, home=True, user=False)
                self.last_move = time.time()

    def set_volume(self, volume):
        self.volume = float(np.clip(volume, 0, 1))
        self.player.out.set_volume(self.volume)
        return self.volume

    def info(self):
        """Everything the page needs when it first loads"""
        modes = []
        for mode, grid in self.grids.items():
            modes.append({"id": mode, "label": grid.label, "km": grid.km, "bbox": grid.bbox,
                          "maxZoom": grid.max_zoom, "home": list(self.home_of(mode)),
                          "layers": grid.layer_infos()})
        return {"modes": modes, "credits": CREDITS, "volume": self.volume,
                "silent": isinstance(self.player.out, NullOut)}


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        # Send the headers and the body together in one write (avoids a delay from Nagle's algorithm)
        wbufsize = 1 << 16

        def setup(self):
            super().setup()
            try:
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

        def log_message(self, *args):
            pass    # don't print every request

        def _send(self, code, content_type, body, cache=None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache or "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code=200):
            self._send(code, "application/json", json.dumps(data).encode())

        def do_GET(self):
            url = urlparse(self.path)
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, "text/html; charset=utf-8", PAGE.encode())

                elif url.path == "/api/info":
                    self._json(app.info())

                elif url.path == "/api/state":
                    self._json(app.state)

                elif url.path == "/api/move":
                    lat = float(query["lat"]) if "lat" in query else None
                    lon = float(query["lon"]) if "lon" in query else None
                    self._json(app.move(lat, lon, query.get("mode"), home=query.get("home") == "1"))

                elif url.path == "/api/volume":
                    self._json({"volume": app.set_volume(float(query.get("v", app.volume)))})

                elif url.path.startswith("/tile/"):
                    mode, layer, z, x, y = url.path[6:].removesuffix(".png").split("/")
                    z, x, y = int(z), int(x), int(y)
                    grid = app.grids.get(mode)
                    if grid is None:
                        self._json({"error": "unknown map"}, 404)
                        return
                    valid = 0 <= z <= 12 and 0 <= x < 2 ** (z + 1) and 0 <= y < 2 ** z
                    tile = grid.tile(z, x, y, layer) if valid else BLANK_TILE
                    self._send(200, "image/png", tile, "max-age=3600")

                else:
                    self._json({"error": "not found"}, 404)

            except (BrokenPipeError, ConnectionResetError):
                pass    # the browser went away
            except Exception as exc:
                try:
                    self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
                except Exception:
                    pass

    return Handler


# The web page
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Earth Soundscapes</title>
<style>
  html, body {
    margin: 0; height: 100%; background: #07111d; color: #e8eef5;
    font: 15px system-ui, Segoe UI, sans-serif; overflow: hidden;
    -webkit-user-select: none; user-select: none;
  }
  #map { position: fixed; inset: 0; touch-action: none; cursor: grab; }
  #map.sq { cursor: move; }

  .panel {
    position: fixed; background: rgba(7,17,29,.82); border: 1px solid #1f3550;
    border-radius: 12px; padding: 12px 16px; backdrop-filter: blur(4px);
  }

  /* Top left: tempo and what is playing */
  #info { left: 16px; top: 16px; max-width: min(520px, calc(100vw - 32px)); }
  #bpm { font-size: 30px; font-weight: 600; letter-spacing: .5px; }
  #lines div { margin-top: 5px; color: #b9c7d8; line-height: 1.35; }
  #lines div b { color: #7fd0ff; font-weight: 600; }

  /* Bottom right: buttons */
  #ctl {
    right: 16px; bottom: 44px; display: flex; gap: 8px; flex-wrap: wrap;
    justify-content: flex-end; max-width: calc(100vw - 32px);
  }
  button {
    min-width: 56px; height: 56px; border-radius: 12px; border: 1px solid #2c4a6b;
    background: #10243a; color: #e8eef5; font-size: 22px; cursor: pointer;
  }
  button.on { background: #1d4f82; border-color: #5aaaf0; }
  button.txt { font-size: 15px; padding: 0 14px; }

  /* Bottom left: layer buttons, colour bar and the explanation */
  #lay { left: 16px; bottom: 44px; max-width: min(420px, calc(100vw - 32px)); padding: 10px 12px; }
  #lay .row { display: flex; flex-wrap: wrap; gap: 6px; }
  #lay button { min-width: 0; height: 36px; font-size: 13px; padding: 0 11px; border-radius: 9px; }
  #leg { margin-top: 9px; font-size: 12px; color: #b9c7d8; }
  #bar { height: 10px; border-radius: 5px; margin: 4px 0 2px; border: 1px solid #2c4a6b; }
  #ends { display: flex; justify-content: space-between; }
  #how { margin-top: 9px; font-size: 13px; line-height: 1.4; color: #b9c7d8; border-top: 1px solid #1f3550; padding-top: 8px; }
  #how b { color: #7fd0ff; }
  #cls { margin-top: 6px; font-size: 12px; max-height: 34vh; overflow: auto; }
  #cls div { display: flex; align-items: center; gap: 7px; padding: 1px 0; }
  #cls i { width: 12px; height: 12px; border-radius: 3px; flex: none; border: 1px solid #0006; }
  #cls span { color: #7d92a8; margin-left: auto; }

  #credits { position: fixed; left: 0; right: 0; bottom: 0; padding: 6px 14px; font-size: 11px; color: #7d92a8; background: rgba(7,17,29,.7); }
  #title { position: fixed; right: 18px; top: 14px; font-size: 13px; letter-spacing: 2px; color: #7d92a8; text-transform: uppercase; }
</style></head><body>
<canvas id="map"></canvas>
<div id="title">Earth Soundscapes</div>
<div class="panel" id="info"><div id="bpm">&hellip;</div><div id="lines"></div></div>
<div class="panel" id="lay"><div class="row" id="layers"></div><div id="leg"></div><div id="how"></div></div>
<div class="panel" id="ctl"></div>
<div id="credits"></div>
<script>
const $ = id => document.getElementById(id);
const canvas = $('map');
const ctx = canvas.getContext('2d');

// Page state
let layer = 'map';        // which layer is showing
let INFO = null;          // what the server told us when the page loaded
let mode = null;          // 'world' or 'site'
let state = null;         // where the square is and what is playing
let W = 0, H = 0, dpr = 1;
let down = null;          // the pointer press that is happening right now
let vol = 0.65;
const view = {z: 5, cx: 0, cy: 0};    // zoom and the centre of the screen (lon, lat)
const tiles = new Map();

// Screen pixels per degree at the current zoom
const pixelsPerDegree = () => 256 * 2 ** view.z / 180;
const modeInfo = () => INFO.modes.find(m => m.id === mode);


function resize() {
  dpr = window.devicePixelRatio || 1;
  W = innerWidth;
  H = innerHeight;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  draw();
}
addEventListener('resize', resize);


// Map tiles
function getTile(m, z, x, y) {
  const key = m + '/' + layer + '/' + z + '/' + x + '/' + y;
  let tile = tiles.get(key);
  if (!tile) {
    tile = {img: new Image(), ok: false};
    tile.img.onload = () => { tile.ok = true; requestAnimationFrame(draw); };
    tile.img.src = '/tile/' + key + '.png';
    tiles.set(key, tile);
    // Forget the oldest tile when there are too many
    if (tiles.size > 900) tiles.delete(tiles.keys().next().value);
  }
  return tile;
}

function draw() {
  if (!INFO || !mode) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#07111d';
  ctx.fillRect(0, 0, W, H);

  const p = pixelsPerDegree();
  const tileDegrees = 256 / p;
  const z = view.z;
  const tilesAcross = 2 ** (z + 1);
  const tilesDown = 2 ** z;

  // Which tiles are on screen
  const x0 = Math.max(0, Math.floor((view.cx - W / 2 / p + 180) / tileDegrees));
  const x1 = Math.min(tilesAcross - 1, Math.floor((view.cx + W / 2 / p + 180) / tileDegrees));
  const y0 = Math.max(0, Math.floor((90 - view.cy - H / 2 / p) / tileDegrees));
  const y1 = Math.min(tilesDown - 1, Math.floor((90 - view.cy + H / 2 / p) / tileDegrees));

  for (let ty = y0; ty <= y1; ty++) {
    for (let tx = x0; tx <= x1; tx++) {
      const tile = getTile(mode, z, tx, ty);
      if (!tile.ok) continue;
      const left = W / 2 + (-180 + tx * tileDegrees - view.cx) * p;
      const top = H / 2 - (90 - ty * tileDegrees - view.cy) * p;
      ctx.drawImage(tile.img, left, top, 256, 256);
    }
  }

  // The red square
  if (state && state.mode === mode) {
    const [south, west, north, east] = state.bounds;
    const left = W / 2 + (west - view.cx) * p;
    const right = W / 2 + (east - view.cx) * p;
    const top = H / 2 - (north - view.cy) * p;
    const bottom = H / 2 - (south - view.cy) * p;
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ff4a4a';
    ctx.fillStyle = 'rgba(255,74,74,.14)';
    ctx.fillRect(left, top, right - left, bottom - top);
    ctx.strokeRect(left, top, right - left, bottom - top);
    // When the square is tiny, also draw a circle so it can still be seen
    if (right - left < 14) {
      ctx.beginPath();
      ctx.arc((left + right) / 2, (top + bottom) / 2, 7, 0, 7);
      ctx.stroke();
    }
  }
}

// Screen position -> [lat, lon]
const toLatLon = (x, y) => [view.cy - (y - H / 2) / pixelsPerDegree(), view.cx + (x - W / 2) / pixelsPerDegree()];

function insideSquare(x, y) {
  if (!state || state.mode !== mode) return false;
  const p = pixelsPerDegree();
  const [south, west, north, east] = state.bounds;
  const pad = 8;
  return x >= W / 2 + (west - view.cx) * p - pad && x <= W / 2 + (east - view.cx) * p + pad &&
         y >= H / 2 - (north - view.cy) * p - pad && y <= H / 2 - (south - view.cy) * p + pad;
}

// The zoom that fits the whole grid on screen
function fitZoom(m) {
  const [west, south, east, north] = m.bbox;
  const z = Math.floor(Math.log2(Math.min(W / (east - west), H / (north - south)) * 180 / 256));
  return Math.max(0, Math.min(m.maxZoom, z));
}

function defaultZoom(m) {
  // The world map starts at zoom 5, the site map starts zoomed out to fit
  return m.id === INFO.modes[0].id ? Math.min(5, m.maxZoom) : fitZoom(m);
}

function clampView() {
  const m = modeInfo();
  const minZoom = Math.max(0, fitZoom(m) - 1);
  view.z = Math.max(minZoom, Math.min(m.maxZoom, view.z));
  view.cy = Math.max(-85, Math.min(85, view.cy));
}


// Info panel (top left)
function renderInfo(s) {
  // The server sends "114 BPM | ground: ... | instruments | drums: ..."
  const parts = (s.info || '').split(' | ');
  $('bpm').textContent = parts[0] || '';
  const lines = $('lines');
  lines.textContent = '';
  parts.slice(1).forEach(text => {
    const div = document.createElement('div');
    const colon = text.indexOf(': ');
    if (colon > 0 && colon < 12) {
      // "ground: Water 55%" -> bold label then the text
      const label = document.createElement('b');
      label.textContent = text.slice(0, colon + 1) + ' ';
      div.append(label, text.slice(colon + 2));
    } else {
      div.textContent = text;
    }
    lines.append(div);
  });
}

function applyState(s, recenter) {
  const modeChanged = s.mode !== mode;
  state = s;
  if (modeChanged) {
    mode = s.mode;
    view.z = defaultZoom(modeInfo());
    recenter = true;
    buildControls();
  }
  if (recenter) {
    view.cx = s.lon;
    view.cy = s.lat;
  }
  clampView();
  renderInfo(s);
  draw();
}


// Moving the square
// Only one request at a time. If you keep moving, only the newest position is sent next.
let inflight = false;
let pending = null;

async function sendMove(lat, lon, extra) {
  if (inflight) {
    pending = [lat, lon, extra];
    return;
  }
  inflight = true;
  try {
    const query = new URLSearchParams({lat, lon, mode, ...(extra || {})});
    const response = await fetch('/api/move?' + query);
    applyState(await response.json(), false);
  } catch (e) {}
  inflight = false;
  if (pending) {
    const next = pending;
    pending = null;
    sendMove(...next);
  }
}

// While dragging, send at most one move every 110 ms
let timer = null;
let latest = null;
function throttledMove(lat, lon) {
  latest = [lat, lon];
  if (timer) return;
  timer = setTimeout(() => {
    timer = null;
    sendMove(...latest);
  }, 110);
}


// Mouse and touch
canvas.addEventListener('pointerdown', e => {
  canvas.setPointerCapture(e.pointerId);
  down = {x: e.clientX, y: e.clientY, cx: view.cx, cy: view.cy, moved: false,
          onSquare: insideSquare(e.clientX, e.clientY)};
});

canvas.addEventListener('pointermove', e => {
  if (!down) {
    // Not pressing: just show the move cursor over the square
    canvas.classList.toggle('sq', insideSquare(e.clientX, e.clientY));
    return;
  }
  const dx = e.clientX - down.x;
  const dy = e.clientY - down.y;
  if (!down.moved && Math.hypot(dx, dy) < 6) return;    // ignore tiny wobbles
  down.moved = true;
  if (down.onSquare) {
    // Dragging the square
    const [lat, lon] = toLatLon(e.clientX, e.clientY);
    throttledMove(lat, lon);
  } else {
    // Dragging the map
    view.cx = down.cx - dx / pixelsPerDegree();
    view.cy = down.cy + dy / pixelsPerDegree();
    clampView();
    draw();
  }
});

canvas.addEventListener('pointerup', e => {
  if (!down) return;
  if (!down.moved) {
    // A tap or click: move the square there
    const [lat, lon] = toLatLon(e.clientX, e.clientY);
    sendMove(lat, lon);
  } else if (down.onSquare) {
    // Finished dragging the square: send the final spot
    const [lat, lon] = toLatLon(e.clientX, e.clientY);
    sendMove(lat, lon);
  }
  down = null;
});

// Mouse wheel zoom, one step at a time
let lastWheel = 0;
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const now = Date.now();
  if (now - lastWheel < 180) return;
  lastWheel = now;
  zoom(e.deltaY < 0 ? 1 : -1);
}, {passive: false});

function zoom(step) {
  view.z += step;
  clampView();
  draw();
}

// Arrow keys or WASD move the square half a square
addEventListener('keydown', e => {
  if (!state) return;
  const key = e.key.toLowerCase();
  const step = state.km / 111.32 / 2;
  const up = (key === 'arrowup' || key === 'w') ? 1 : (key === 'arrowdown' || key === 's') ? -1 : 0;
  const right = (key === 'arrowright' || key === 'd') ? 1 : (key === 'arrowleft' || key === 'a') ? -1 : 0;
  if (!up && !right) return;
  e.preventDefault();
  const lat = state.lat + up * step;
  const lon = state.lon + right * step / Math.max(Math.cos(state.lat * Math.PI / 180), .05);
  throttledMove(lat, lon);
});


// Buttons and the layer panel
function makeButton(label, title, onclick, className) {
  const button = document.createElement('button');
  button.textContent = label;
  button.title = title;
  if (className) button.className = className;
  button.onclick = onclick;
  return button;
}

function buildLayers() {
  const m = modeInfo();
  const box = $('layers');
  box.textContent = '';
  if (!m.layers.some(l => l.id === layer)) layer = 'map';

  // One button per layer
  m.layers.forEach(l => {
    const onclick = () => { layer = l.id; buildLayers(); draw(); };
    box.append(makeButton(l.label, 'Show ' + l.label, onclick, l.id === layer ? 'on' : ''));
  });
  // No layer buttons if there's only the map
  box.parentElement.style.display = m.layers.length > 1 ? '' : 'none';

  // Colour bar with the range at each end
  const current = m.layers.find(l => l.id === layer);
  const legend = $('leg');
  legend.textContent = '';
  if (current && current.stops && current.stops.length) {
    const bar = document.createElement('div');
    bar.id = 'bar';
    bar.style.background = 'linear-gradient(90deg,' + current.stops.map(c => 'rgb(' + c.join(',') + ')').join(',') + ')';
    const ends = document.createElement('div');
    ends.id = 'ends';
    const format = v => (Math.abs(v) >= 100 ? Math.round(v) : +v.toFixed(2)) + (current.unit ? ' ' + current.unit : '');
    const low = document.createElement('span');
    const high = document.createElement('span');
    low.textContent = format(current.lo);
    high.textContent = format(current.hi);
    ends.append(low, high);
    legend.append(bar, ends);
  }

  // What this layer does to the music
  const how = $('how');
  how.textContent = '';
  if (current) {
    const heading = document.createElement('b');
    heading.textContent = current.what + ': ';
    how.append(heading, current.how);

    // Land cover also lists every class with its instrument
    if (current.classes) {
      const list = document.createElement('div');
      list.id = 'cls';
      current.classes.forEach(c => {
        const row = document.createElement('div');
        const swatch = document.createElement('i');
        const name = document.createElement('em');
        const instrument = document.createElement('span');
        swatch.style.background = 'rgb(' + c.rgb.join(',') + ')';
        name.textContent = c.name;
        name.style.fontStyle = 'normal';
        instrument.textContent = '\u2192 ' + c.instr;
        row.append(swatch, name, instrument);
        list.append(row);
      });
      how.append(list);
    }
  }
}

function buildControls() {
  buildLayers();
  const controls = $('ctl');
  controls.textContent = '';
  // Buttons to switch between the world and the site map
  if (INFO.modes.length > 1) {
    INFO.modes.forEach(m => {
      const onclick = () => sendMove(0, 0, {mode: m.id, home: '1'});
      controls.append(makeButton(m.label, 'Switch map', onclick, 'txt' + (m.id === mode ? ' on' : '')));
    });
  }
  controls.append(
    makeButton('\u2302', 'Back to start', () => sendMove(0, 0, {home: '1'})),
    makeButton('\u2212', 'Zoom out', () => zoom(-1)),
    makeButton('+', 'Zoom in', () => zoom(1)),
    makeButton('\ud83d\udd09', 'Quieter', () => setVolume(vol - .1)),
    makeButton('\ud83d\udd0a', 'Louder', () => setVolume(vol + .1)));
}

async function setVolume(v) {
  vol = Math.max(0, Math.min(1, v));
  try { await fetch('/api/volume?v=' + vol); } catch (e) {}
}


// Check the server every 1.5 s, in case the idle timer moved the square or another screen did
async function poll() {
  try {
    if (!down) {
      const s = await (await fetch('/api/state')).json();
      if (s.t && (!state || s.t !== state.t)) {
        const p = pixelsPerDegree();
        const offScreen = Math.abs(s.lon - view.cx) * p > W / 2 || Math.abs(s.lat - view.cy) * p > H / 2;
        applyState(s, offScreen);
      }
    }
  } catch (e) {}
  setTimeout(poll, 1500);
}

// Start
(async function init() {
  INFO = await (await fetch('/api/info')).json();
  vol = INFO.volume;
  $('credits').textContent = INFO.credits + (INFO.silent ? '   [silent preview]' : '');
  resize();
  applyState(await (await fetch('/api/state')).json(), true);
  poll();
})();
</script></body></html>
"""


# Start everything
def main():
    parser = argparse.ArgumentParser(description="Earth Soundscapes kiosk", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--world", help="baked world grid, like worlddata/world.u8")
    parser.add_argument("--site", help="baked high detail site grid, like sitedata/site.u8")
    parser.add_argument("--world-km", type=float, default=64, help="square size on the world grid (default 64)")
    parser.add_argument("--site-km", type=float, default=32, help="square size on the site grid (default 32)")
    parser.add_argument("--home", help="start position 'lat,lon' (default: the centre of the first grid)")
    parser.add_argument("--idle", type=float, default=300, help="seconds with no input before going back to the start (0 = never)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to let other devices on the network connect")
    parser.add_argument("--open", action="store_true", help="open the page in your browser")
    parser.add_argument("--volume", type=float, default=0.65, help="start volume 0-1")
    parser.add_argument("--soundfont", help="General MIDI .sf2 file (default: the first .sf2 in this folder)")
    parser.add_argument("--fluidsynth", help="path to the fluidsynth program")
    parser.add_argument("--synth", choices=["auto", "cli", "py"], default="auto", help="cli = fluidsynth program, py = pyfluidsynth")
    parser.add_argument("--driver", help="FluidSynth audio driver (Windows: wasapi, dsound; Mac: coreaudio; Linux: pulseaudio, alsa)")
    parser.add_argument("--no-sound", action="store_true", help="silent preview")
    parser.add_argument("--test-sound", action="store_true", help="play a test chord and quit")
    args = parser.parse_args()

    if args.test_sound:
        out = make_backend(args)
        print("Playing a C major chord, a bass note and a drum hit...")
        test_sound(out)
        out.close()
        print("Done. If you heard nothing, see fluidsynth.log and try --driver wasapi or --driver dsound.")
        return

    if not (args.world or args.site):
        sys.exit("Give --world and/or --site (the baked grids). Example: --world worlddata/world.u8")

    grids = {}
    if args.world:
        grids["world"] = Grid("world", args.world, args.world_km, f"World ({args.world_km:g} km square)")
    if args.site:
        grids["site"] = Grid("site", args.site, args.site_km, f"Convention site ({args.site_km:g} km square)")

    home = tuple(float(x) for x in args.home.split(",")) if args.home else None
    out = make_backend(args)
    player = es.LoopPlayer(out=out)
    app = App(grids, player, home, args.idle, args.volume)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.daemon_threads = True
    url = f"http://localhost:{args.port}"
    print(f"Earth Soundscapes running at {url}  (Ctrl+C to stop)")
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        player.close()
        out.close()


if __name__ == "__main__":
    main()
