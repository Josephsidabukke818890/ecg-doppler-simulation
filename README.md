# ECG Doppler Simulation System

Sistem simulasi ECG (Electrocardiogram) dan Doppler untuk pembelajaran dan penelitian biomedis.

## Fitur Utama

- **ECG Signal Simulation**: Mensimulasikan sinyal jantung normal dan abnormal
- **Doppler Simulation**: Mensimulasikan efek Doppler untuk aliran darah
- **Real-time Visualization**: Visualisasi grafik sinyal secara real-time
- **Signal Processing**: Filter dan analisis sinyal
- **Interactive Dashboard**: Antarmuka interaktif untuk kontrol simulasi

## Instalasi

```bash
pip install -r requirements.txt
```

## Penggunaan

### ECG Simulation
```python
from ecg_simulator import ECGSimulator

simulator = ECGSimulator(heart_rate=72, duration=10)
signal = simulator.generate_signal()
simulator.plot()
```

### Doppler Simulation
```python
from doppler_simulator import DopplerSimulator

doppler = DopplerSimulator(velocity=0.5, frequency=2e6)
signal = doppler.generate_signal()
doppler.plot()
```

## Struktur Proyek

```
ecg-doppler-simulation/
├── ecg_simulator.py          # ECG signal simulator
├── doppler_simulator.py      # Doppler effect simulator
├── signal_processor.py       # Signal processing utilities
├── visualizer.py             # Visualization tools
├── main.py                   # Main application
├── requirements.txt          # Dependencies
└── tests/                    # Unit tests
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- PyQt5 atau Tkinter

## Lisensi

MIT License
