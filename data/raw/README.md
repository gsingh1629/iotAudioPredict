# Raw Audio Dataset Layout

Place WAV recordings from the actual Soundbox under the matching class folder.
The folder name is the label. Put language, unit, volume, position, recorder,
and take metadata in the filename.

Recommended filename pattern:

```text
<class>_<variant>_unit<id>_vol<level>_pos<position>_rec<device>_take<n>.wav
```

Examples:

```text
01_power_on/power_on_en_unit001_vol070_pos_center_rec_laptop_take001.wav
02_power_off/power_off_hi_unit003_vol100_pos_left1cm_rec_mobile_take002.wav
03_beep/beep_unit002_vol030_pos_center_rec_laptop_take001.wav
07_silence/silence_factory_idle_rec_laptop_take001.wav
08_interference/talking_rec_mobile_take001.wav
```

Keep `09_distortion` empty. `dataset_builder.py` generates it from clean
classes `01_power_on` through `06_bind`.
