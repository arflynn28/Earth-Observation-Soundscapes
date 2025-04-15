# Earth Observation Soundscapes
This project transforms remote sensing datasets into musical compositions. Using sonification techniques, spatial patterns from Earth observation data — like elevation, land surface temperature, vegetation, cloud frequency, and urban activity — are turned into musical attributes such as pitch, rhythm, instrument selection, and volume.

---

## Project Goals

- Convert geospatial raster datasets into time-sequenced music.
- Map environmental variables to musical elements (e.g., elevation to pitch, SAR backscatter to percussion).
- Explore the artistic intersection of Earth science and sound design.

---

## Datasets Used

| Dataset                     | Musical Use              |
|----------------------------|--------------------------|
| ALOS DSM (Elevation)       | Pitch                    |
| MODIS LST                  | Pitch                    |
| Sentinel-1 SAR (VV)        | Percussion Intensity     |
| VIIRS Nighttime Lights     | Volume & Urban Booster   |
| MODIS Land Cover (IGBP)    | Instrument Assignment    |
| Sentinel-2 Cloud Frequency | Rhythm/Tempo             |
| Landsat EVI                | Volume for Natural Classes |

---

## 🎼 Musical Mapping

| Music Element     | Controlled By          | Logic                                                |
|------------------|------------------------|------------------------------------------------------|
| **Pitch**        | Elevation + Temperature| Higher temp/elevation → pitch within MIDI range      |
| **Rhythm**       | Cloud Frequency        | Higher cloud freq → denser rhythm (faster tempo)     |
| **Volume**       | Night Lights + EVI     | Bright lights & green areas boost instrument loudness|
| **Instruments**  | Land Cover Classes     | IGBP 17-class scheme → instrument types              |
| **Percussion**   | SAR VV Backscatter     | More return → more layered/complex percussion        |

---

## Current Status

- ✅ Data preprocessing, clipping, and normalization complete
- ✅ Pitch, rhythm, volume, and percussion logic implemented
- ✅ Instrument assignment based on land cover developed
- ⏳ MIDI generation and music synthesis in progress

---

## Preview

The data is processed in columns to simulate time flow, where:
- **Each column = a moment in time**
- **Each row = musical layers or instruments**

Music is generated column-by-column, assigning pitch, rhythm, volume, and instruments based on that snapshot of Earth.

---

## Tools & Libraries

- Google Earth Engine (GEE)
- Python (NumPy, Rasterio, geemap, etc.)
- Jupyter/Colab
- (Coming soon: `mido`, `pretty_midi`, or `music21` for MIDI export)

---

## Artistic Direction

This is both a data science and storytelling piece. The soundscape reflects different ecological and urban patterns:
- Forests come alive with warm strings and woodwinds
- Cities thrum with night-amplified synths and basslines
- Cloudy columns create stormy, rapid-fire percussion

---

## Next Steps

- [ ] Finalize MIDI generation and export
- [ ] Begin multi-region compositions (e.g., alpine vs urban)
- [ ] Implement visual overlays for audio-visual presentation
- [ ] Polish and package as an open-access creative toolkit

---

## License

- **Code** is licensed under the MIT License. See `LICENSE`.
- **Music and Audio Output** are licensed under [CC BY 4.0](music-license.txt).


---

## 🎧 Listen Soon

_MIDI demos and DAW exports coming in future commits!_
