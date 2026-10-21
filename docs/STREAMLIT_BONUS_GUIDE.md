# Streamlit Bonus Guide (+2 Marks) — Wi-Fi Fingerprint Indoor Localization
> 23CSE301 ML Capstone | Ganesh 519 - Suhas 503 - Balaji 510
> Hosted dashboard: Track-1 UJI model + Track-2 live campus scan. Zero-bug checklist inside.

## 1. What the examiner must see (tick before demo)

- [ ] Public URL opens without login (`https://<app>.streamlit.app`)
- [ ] Tab 1 UJI: champions KNN R2 0.9966 + SVM F1 0.9937, 10-model table, ROC + residuals + spatial scatter
- [ ] Tab 2 UJI Predict: paste/slide fingerprint -> Building/Floor + Longitude/Latitude
- [ ] Tab 3 Campus Live: Scan Now (Windows) + Upload CSV (Cloud) -> Block/Floor/Room + Lat/Lon + Top-3 nearest rooms in metres + star on map
- [ ] Tab 4 Health: CV mean+-std, tuning table, official 1111 validation benchmark
- [ ] One palette everywhere (no neon). Hexes in section 2.

## 2. Human-professional palette (use in every plot + CSS)

| Token | Hex | Use |
|---|---|---|
| Paper bg | `#F8F6F1` | app background |
| Card | `#FFFFFF` | cards, tables |
| Ink | `#1A2A32` | text |
| Primary teal-navy | `#1B3A4B` | headers, buttons |
| Slate | `#5C6B73` | secondary text |
| Terracotta | `#C67C4E` | predictions, CTA |
| Sage | `#6AA99E` | success, true label |
| Muted gold | `#D9A441` | best-model badge |
| Categorical 8 | `#1B3A4B,#C67C4E,#6AA99E,#8AA8C8,#E0C084,#7A6C5D,#4A7C7E,#B7836B` | blocks A-H |

```python
PAL = {"bg":"#F8F6F1","card":"#FFFFFF","ink":"#1A2A32","primary":"#1B3A4B",
       "accent":"#C67C4E","sage":"#6AA99E","gold":"#D9A441","slate":"#5C6B73"}
CAT8 = ["#1B3A4B","#C67C4E","#6AA99E","#8AA8C8","#E0C084","#7A6C5D","#4A7C7E","#B7836B"]
plt.rcParams.update({"figure.facecolor":PAL["bg"],"axes.facecolor":PAL["card"],
                     "text.color":PAL["ink"],"axes.labelcolor":PAL["ink"]})
```

Theme file `.streamlit/config.toml`:
```toml
[theme]
base="light"
primaryColor="#1B3A4B"
backgroundColor="#F8F6F1"
secondaryBackgroundColor="#FFFFFF"
textColor="#1A2A32"
font="sans serif"
```

## 3. Repo layout (rubric §8)

```
.streamlit/config.toml
app/app.py + app/app_utils/     # bonus GUI (Streamlit entry: app/app.py)
scripts/train_champions.py      # retrains UJI + campus champions
models/uji + models/campus      # organized binaries, metadata, centroids, results
notebooks/regression.ipynb, classification.ipynb, clustering.ipynb
data/raw + data/campus
docs/STREAMLIT_BONUS_GUIDE.md
```

Why two pipelines? UJI=520 WAPs (465 active) vs campus=219 WAPs. Never mix orders.
Each `*_meta.json` stores its own `feature_order`.

<!--PART2-->
## 4. Collect campus data (Windows, admin terminal)

```powershell
python scripts/campus_collector.py
# enter Block A-H, Floor 0-3, Room e.g. CP LAB, bursts 10-20 per spot
# auto-GPS via get_gps.ps1, manual lat,lon fallback if UNKNOWN
python scripts/build_campus_dataset.py
# -> data/campus/trainingData_campus.csv (524 x 219 + meta) + wap_dictionary.json
```

Need >=10 bursts per room, >=2 rooms per floor per demo block.

## 5. Train once, reuse everywhere

```powershell
pip install -r requirements.txt
python scripts/train_champions.py
# or, after a campus-data-only update:
python scripts/train_champions.py --dataset campus
streamlit run app.py
```

Lock: `+100 -> -105`, zero-var fitted on train only, 5 feats
`n_visible_aps, mean_rssi, max_rssi, std_rssi, rssi_range`, scaler fit train only.
UJI champs: KNN k=5 distance (lon/lat) + SVM RBF C=10 (floor).
Campus (n=524): KNN k=3 distance (lat/lon) + RF block/floor + KNN room + 3-fold CV.

## 6. Live scan flow (say this in viva)

Scan Now -> `netsh wlan show networks mode=bssid` -> `% -> dBm (pct/2-100)`
-> BSSID->WAP via wap_dictionary.json (unknown ignored, missing=-105)
-> order by campus_meta feature_order -> scaler.transform
-> predict block/floor/room + lat/lon -> haversine to room_centroids.csv
-> Top-3 + star on map. Cloud has no netsh: use Upload CSV path.

## 7. Hosting (free Cloud)

1. Push to GitHub. 2. share.streamlit.io New app -> main, file `app.py`, py 3.11.
3. requirements must include streamlit, plotly, sklearn, pandas, numpy, matplotlib, seaborn, joblib.
4. Reboot on ModuleNotFound. 5. Demo on Cloud via Upload CSV. 6. Freeze after green deploy.

## 8. Viva one-liners

- Why -105? Below -104 sensitivity floor, preserves order for distance models.
- Why KNN wins? Fingerprint neighbours equal physical neighbours.
- Why 99% test but 85% official validation? Time/device shift, honest gap.
- Why separate campus models? Different WAP universe (219 vs 465).
- Why K=3 + RF on campus? n=524 small; 3-fold + Leave-One-Room-Out is honest.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| X has 219 features, expected 470 | loaded wrong meta; use uji vs campus meta |
| All -105 after scan | admin terminal, Wi-Fi on, netsh must list BSSIDs |
| GPS UNKNOWN | allow Location privacy, else manual lat,lon |
| Empty Scan on Cloud | expected on Linux; use Upload demo CSV |
| joblib load fails | retrain via train_champions.py; generated pkls remain git-ignored |

## 10. Demo order

1. build_campus_dataset (show 524x219) 2. train_champions (show CV)
3. app.py: Overview -> UJI Predict (1 validation row) -> Campus Scan Now -> star + Near: A-CP LAB 3.2 m
4. Open public Cloud URL on phone as hosting proof.

