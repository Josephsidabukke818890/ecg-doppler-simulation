import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from typing import Tuple, Optional

class ECGSimulator:
    """
    Simulator untuk sinyal ECG (Electrocardiogram).
    
    Menghasilkan sinyal jantung yang realistis dengan komponen P, QRS, T.
    """
    
    def __init__(self, heart_rate: float = 72, sampling_rate: int = 500, duration: float = 10):
        """
        Inisialisasi ECG Simulator.
        
        Args:
            heart_rate: Detak jantung per menit (default: 72 bpm)
            sampling_rate: Frekuensi sampling dalam Hz (default: 500 Hz)
            duration: Durasi sinyal dalam detik (default: 10 detik)
        """
        self.heart_rate = heart_rate
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.t = np.linspace(0, duration, int(sampling_rate * duration))
        self.signal = None
        
    def generate_signal(self) -> np.ndarray:
        """
        Generate sinyal ECG realistis.
        
        Returns:
            Array sinyal ECG
        """
        # Konversi heart rate menjadi frekuensi
        frequency = self.heart_rate / 60  # Hz
        
        # Komponen P wave (atrial depolarization)
        p_wave = 0.15 * np.sin(2 * np.pi * frequency * self.t + np.pi/4) * \
                 np.exp(-((self.t % (1/frequency) - 0.1) ** 2) / (2 * 0.01**2))
        
        # Komponen QRS complex (ventricular depolarization)
        qrs_wave = 1.0 * np.sin(2 * np.pi * frequency * 5 * self.t + np.pi/2) * \
                   np.exp(-((self.t % (1/frequency) - 0.1) ** 2) / (2 * 0.005**2))
        
        # Komponen T wave (ventricular repolarization)
        t_wave = 0.3 * np.sin(2 * np.pi * frequency * self.t - np.pi/4) * \
                 np.exp(-((self.t % (1/frequency) - 0.2) ** 2) / (2 * 0.02**2))
        
        # Baseline wander (low frequency noise)
        baseline = 0.1 * np.sin(2 * np.pi * 0.1 * self.t)
        
        # High frequency noise (EMG noise)
        noise = 0.05 * np.random.randn(len(self.t))
        
        # Kombinasi semua komponen
        self.signal = p_wave + qrs_wave + t_wave + baseline + noise
        
        return self.signal
    
    def add_arrhythmia(self, arrhythmia_type: str = 'premature_beat', position: float = 0.5) -> np.ndarray:
        """
        Tambahkan aritmia ke sinyal ECG.
        
        Args:
            arrhythmia_type: Tipe aritmia ('premature_beat', 'pause', 'flutter')
            position: Posisi aritmia dalam sinyal (0-1)
            
        Returns:
            Sinyal ECG dengan aritmia
        """
        if self.signal is None:
            self.generate_signal()
        
        signal_modified = self.signal.copy()
        idx = int(position * len(signal_modified))
        window_size = int(self.sampling_rate * 0.1)  # 100ms window
        
        if arrhythmia_type == 'premature_beat':
            # Tambahkan QRS yang lebih awal
            window = signal_modified[idx:idx+window_size]
            signal_modified[idx:idx+window_size] = window * 1.5
            
        elif arrhythmia_type == 'pause':
            # Pause (tidak ada detak)
            signal_modified[idx:idx+window_size] = signal_modified[idx:idx+window_size] * 0.1
            
        elif arrhythmia_type == 'flutter':
            # Atrial flutter (frekuensi tinggi)
            flutter = 0.3 * np.sin(2 * np.pi * 6 * self.t[idx:idx+window_size])
            signal_modified[idx:idx+window_size] += flutter
        
        return signal_modified
    
    def detect_peaks(self) -> Tuple[np.ndarray, dict]:
        """
        Deteksi puncak QRS kompleks dalam sinyal ECG.
        
        Returns:
            Tuple (indices of peaks, statistics dictionary)
        """
        if self.signal is None:
            self.generate_signal()
        
        # Gunakan scipy untuk deteksi puncak
        peaks, properties = scipy_signal.find_peaks(
            self.signal, 
            height=np.max(self.signal) * 0.3,
            distance=int(self.sampling_rate * 0.3)  # Min 300ms antara peaks
        )
        
        # Hitung statistik
        if len(peaks) > 1:
            intervals = np.diff(peaks) / self.sampling_rate  # Dalam detik
            hr = 60 / np.mean(intervals) if np.mean(intervals) > 0 else 0
        else:
            intervals = []
            hr = 0
        
        stats = {
            'num_peaks': len(peaks),
            'heart_rate': hr,
            'rr_intervals': intervals,
            'mean_interval': np.mean(intervals) if len(intervals) > 0 else 0
        }
        
        return peaks, stats
    
    def plot(self, show_peaks: bool = True, arrhythmia_signal: Optional[np.ndarray] = None):
        """
        Plot sinyal ECG.
        
        Args:
            show_peaks: Jika True, tampilkan peak detection
            arrhythmia_signal: Sinyal alternatif untuk di-plot (misal dengan aritmia)
        """
        if self.signal is None:
            self.generate_signal()
        
        signal_to_plot = arrhythmia_signal if arrhythmia_signal is not None else self.signal
        
        plt.figure(figsize=(14, 6))
        plt.plot(self.t, signal_to_plot, 'b-', linewidth=1.5, label='ECG Signal')
        
        if show_peaks:
            peaks, stats = self.detect_peaks()
            plt.plot(self.t[peaks], signal_to_plot[peaks], 'rx', markersize=8, label='QRS Peak')
            plt.title(f"ECG Signal (HR: {stats['heart_rate']:.1f} bpm)")
        else:
            plt.title("ECG Signal")
        
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (mV)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def get_statistics(self) -> dict:
        """
        Dapatkan statistik sinyal ECG.
        
        Returns:
            Dictionary dengan statistik sinyal
        """
        if self.signal is None:
            self.generate_signal()
        
        peaks, stats = self.detect_peaks()
        
        return {
            'heart_rate': stats['heart_rate'],
            'num_beats': stats['num_peaks'],
            'rr_intervals': stats['rr_intervals'],
            'mean_rr_interval': stats['mean_interval'],
            'signal_mean': np.mean(self.signal),
            'signal_std': np.std(self.signal),
            'signal_min': np.min(self.signal),
            'signal_max': np.max(self.signal),
            'hrv': np.std(stats['rr_intervals']) if len(stats['rr_intervals']) > 1 else 0  # Heart Rate Variability
        }


if __name__ == "__main__":
    # Contoh penggunaan
    print("=== ECG Simulator Demo ===")
    
    # Normal ECG
    ecg = ECGSimulator(heart_rate=72, duration=5)
    signal = ecg.generate_signal()
    stats = ecg.get_statistics()
    
    print(f"\nNormal ECG Statistics:")
    print(f"  Heart Rate: {stats['heart_rate']:.1f} bpm")
    print(f"  Number of Beats: {stats['num_beats']}")
    print(f"  HRV: {stats['hrv']:.4f}")
    
    # ECG dengan aritmia
    signal_arrhythmia = ecg.add_arrhythmia('premature_beat', position=0.5)
    
    ecg.plot(show_peaks=True, arrhythmia_signal=signal_arrhythmia)
