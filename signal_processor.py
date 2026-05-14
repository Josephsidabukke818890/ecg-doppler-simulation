import numpy as np
from scipy import signal as scipy_signal
from scipy.fftpack import fft, fftfreq
from typing import Tuple, Optional

class SignalProcessor:
    """
    Utility class untuk pemrosesan sinyal ECG dan Doppler.
    """
    
    @staticmethod
    def apply_lowpass_filter(signal: np.ndarray, 
                            cutoff_freq: float, 
                            sampling_rate: float,
                            order: int = 4) -> np.ndarray:
        """
        Aplikasikan lowpass filter pada sinyal.
        
        Args:
            signal: Input signal
            cutoff_freq: Frekuensi cutoff dalam Hz
            sampling_rate: Sampling rate dalam Hz
            order: Order filter
            
        Returns:
            Filtered signal
        """
        nyquist_freq = sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist_freq
        
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99
        
        b, a = scipy_signal.butter(order, normalized_cutoff, btype='low')
        filtered = scipy_signal.filtfilt(b, a, signal)
        
        return filtered
    
    @staticmethod
    def apply_highpass_filter(signal: np.ndarray,
                             cutoff_freq: float,
                             sampling_rate: float,
                             order: int = 4) -> np.ndarray:
        """
        Aplikasikan highpass filter pada sinyal.
        
        Args:
            signal: Input signal
            cutoff_freq: Frekuensi cutoff dalam Hz
            sampling_rate: Sampling rate dalam Hz
            order: Order filter
            
        Returns:
            Filtered signal
        """
        nyquist_freq = sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist_freq
        
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99
        
        b, a = scipy_signal.butter(order, normalized_cutoff, btype='high')
        filtered = scipy_signal.filtfilt(b, a, signal)
        
        return filtered
    
    @staticmethod
    def apply_bandpass_filter(signal: np.ndarray,
                             low_freq: float,
                             high_freq: float,
                             sampling_rate: float,
                             order: int = 4) -> np.ndarray:
        """
        Aplikasikan bandpass filter pada sinyal.
        
        Args:
            signal: Input signal
            low_freq: Frekuensi cutoff bawah dalam Hz
            high_freq: Frekuensi cutoff atas dalam Hz
            sampling_rate: Sampling rate dalam Hz
            order: Order filter
            
        Returns:
            Filtered signal
        """
        nyquist_freq = sampling_rate / 2
        normalized_low = low_freq / nyquist_freq
        normalized_high = high_freq / nyquist_freq
        
        normalized_low = np.clip(normalized_low, 0.01, 0.99)
        normalized_high = np.clip(normalized_high, 0.01, 0.99)
        
        b, a = scipy_signal.butter(order, [normalized_low, normalized_high], btype='band')
        filtered = scipy_signal.filtfilt(b, a, signal)
        
        return filtered
    
    @staticmethod
    def remove_baseline_wander(signal: np.ndarray,
                              sampling_rate: float,
                              window_size: float = 0.5) -> np.ndarray:
        """
        Hilangkan baseline wander dari sinyal.
        
        Args:
            signal: Input signal
            sampling_rate: Sampling rate dalam Hz
            window_size: Ukuran window untuk median filter dalam detik
            
        Returns:
            Signal tanpa baseline wander
        """
        window_length = int(window_size * sampling_rate)
        if window_length % 2 == 0:
            window_length += 1
        
        # Apply median filter untuk baseline
        baseline = scipy_signal.medfilt(signal, kernel_size=window_length)
        
        # Subtract baseline
        corrected = signal - baseline
        
        return corrected
    
    @staticmethod
    def compute_fft(signal: np.ndarray, 
                   sampling_rate: float,
                   fft_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hitung FFT dari sinyal.
        
        Args:
            signal: Input signal
            sampling_rate: Sampling rate dalam Hz
            fft_size: FFT size (default: length of signal)
            
        Returns:
            Tuple (frequency array, magnitude spectrum)
        """
        if fft_size is None:
            fft_size = len(signal)
        
        # Apply window
        window = scipy_signal.hann(len(signal))
        windowed = signal * window
        
        # Compute FFT
        spectrum = fft(windowed, fft_size)
        magnitude = np.abs(spectrum) / len(signal)
        
        # Frequency array
        freqs = fftfreq(fft_size, 1/sampling_rate)
        
        # Return only positive frequencies
        positive_idx = freqs >= 0
        return freqs[positive_idx], magnitude[positive_idx]
    
    @staticmethod
    def normalize_signal(signal: np.ndarray, 
                        method: str = 'minmax') -> np.ndarray:
        """
        Normalisasi sinyal.
        
        Args:
            signal: Input signal
            method: Metode normalisasi ('minmax', 'zscore')
            
        Returns:
            Normalized signal
        """
        if method == 'minmax':
            signal_min = np.min(signal)
            signal_max = np.max(signal)
            normalized = (signal - signal_min) / (signal_max - signal_min)
            
        elif method == 'zscore':
            mean = np.mean(signal)
            std = np.std(signal)
            normalized = (signal - mean) / std if std > 0 else signal
        
        else:
            normalized = signal
        
        return normalized
    
    @staticmethod
    def compute_power_spectrum(signal: np.ndarray,
                              sampling_rate: float,
                              window_type: str = 'hann') -> Tuple[np.ndarray, np.ndarray]:
        """
        Hitung power spectral density.
        
        Args:
            signal: Input signal
            sampling_rate: Sampling rate dalam Hz
            window_type: Tipe window ('hann', 'hamming', 'blackman')
            
        Returns:
            Tuple (frequency array, PSD)
        """
        freqs, psd = scipy_signal.welch(signal, sampling_rate, 
                                       window=window_type,
                                       nperseg=min(1024, len(signal)))
        
        return freqs, psd
    
    @staticmethod
    def detect_peaks(signal: np.ndarray,
                    height: Optional[float] = None,
                    distance: Optional[int] = None,
                    prominence: Optional[float] = None) -> np.ndarray:
        """
        Deteksi peak dalam sinyal.
        
        Args:
            signal: Input signal
            height: Minimum height untuk peak
            distance: Minimum distance antara peaks
            prominence: Minimum prominence untuk peak
            
        Returns:
            Array indeks peak
        """
        peaks, _ = scipy_signal.find_peaks(signal, 
                                           height=height,
                                           distance=distance,
                                           prominence=prominence)
        
        return peaks
    
    @staticmethod
    def compute_derivative(signal: np.ndarray,
                          sampling_rate: float) -> np.ndarray:
        """
        Hitung derivative (first derivative) dari sinyal.
        
        Args:
            signal: Input signal
            sampling_rate: Sampling rate dalam Hz
            
        Returns:
            Derivative signal
        """
        dt = 1 / sampling_rate
        derivative = np.gradient(signal, dt)
        
        return derivative
    
    @staticmethod
    def compute_correlation(signal1: np.ndarray,
                           signal2: np.ndarray) -> np.ndarray:
        """
        Hitung cross-correlation antara dua sinyal.
        
        Args:
            signal1: First signal
            signal2: Second signal
            
        Returns:
            Cross-correlation array
        """
        correlation = scipy_signal.correlate(signal1, signal2, mode='same')
        correlation = correlation / np.max(np.abs(correlation))
        
        return correlation


if __name__ == "__main__":
    # Contoh penggunaan
    import matplotlib.pyplot as plt
    
    print("=== Signal Processing Demo ===")
    
    # Generate test signal
    sampling_rate = 500  # Hz
    duration = 2  # detik
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Signal terdiri dari beberapa komponen
    signal = (
        np.sin(2 * np.pi * 5 * t) +  # 5 Hz
        0.5 * np.sin(2 * np.pi * 15 * t) +  # 15 Hz
        0.1 * np.random.randn(len(t))  # noise
    )
    
    # Test filtering
    filtered = SignalProcessor.apply_bandpass_filter(signal, 3, 20, sampling_rate)
    
    # Test FFT
    freqs, magnitude = SignalProcessor.compute_fft(filtered, sampling_rate)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    axes[0].plot(t, signal, 'b-', alpha=0.7, label='Original')
    axes[0].plot(t, filtered, 'r-', alpha=0.7, label='Filtered')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(freqs, magnitude)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_xlim([0, 50])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
