import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from typing import Tuple, Optional

class DopplerSimulator:
    """
    Simulator untuk efek Doppler pada aliran darah.
    
    Mensimulasikan perubahan frekuensi akibat gerakan darah ke arah/jauh dari transduser.
    """
    
    def __init__(self, 
                 carrier_frequency: float = 2e6,  # 2 MHz
                 sampling_rate: int = 1e7,  # 10 MHz
                 duration: float = 5,
                 max_velocity: float = 1.0):  # m/s
        """
        Inisialisasi Doppler Simulator.
        
        Args:
            carrier_frequency: Frekuensi carrier dalam Hz (default: 2 MHz)
            sampling_rate: Frekuensi sampling dalam Hz (default: 10 MHz)
            duration: Durasi sinyal dalam detik (default: 5 detik)
            max_velocity: Kecepatan maksimal aliran dalam m/s
        """
        self.carrier_frequency = carrier_frequency
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.max_velocity = max_velocity
        self.speed_of_sound = 1540  # m/s (dalam darah)
        self.t = np.linspace(0, duration, int(sampling_rate * duration))
        self.signal = None
        self.velocity_profile = None
    
    def calculate_doppler_shift(self, velocity: float) -> float:
        """
        Hitung pergeseran frekuensi Doppler.
        
        Rumus: f_d = 2 * f_0 * v * cos(theta) / c
        
        Args:
            velocity: Kecepatan aliran (m/s)
            
        Returns:
            Pergeseran frekuensi dalam Hz
        """
        theta = 0  # Sudut insonasi (0 = optimal)
        doppler_shift = 2 * self.carrier_frequency * velocity * np.cos(np.radians(theta)) / self.speed_of_sound
        return doppler_shift
    
    def generate_velocity_profile(self, profile_type: str = 'pulsatile') -> np.ndarray:
        """
        Generate profil kecepatan aliran darah.
        
        Args:
            profile_type: Tipe profil ('constant', 'pulsatile', 'turbulent')
            
        Returns:
            Array profil kecepatan
        """
        if profile_type == 'constant':
            # Kecepatan konstan
            velocity = np.ones_like(self.t) * self.max_velocity * 0.7
            
        elif profile_type == 'pulsatile':
            # Aliran pulsatif (seperti arteri)
            # Sistolik: tinggi, diastolik: rendah
            heartbeat_freq = 1.2  # Hz (72 bpm)
            systolic_duration = 0.3  # 30% dari cycle
            
            phase = (self.t * heartbeat_freq) % 1
            velocity = np.where(
                phase < systolic_duration,
                self.max_velocity * (1 + 0.5 * np.sin(2 * np.pi * phase / systolic_duration)),
                self.max_velocity * 0.3 * np.cos(2 * np.pi * (phase - systolic_duration) / (1 - systolic_duration))
            )
            
        elif profile_type == 'turbulent':
            # Aliran turbulen (abnormal)
            base_velocity = self.max_velocity * 0.5
            turbulence = 0.3 * np.sin(2 * np.pi * 3 * self.t) + 0.2 * np.cos(2 * np.pi * 7 * self.t)
            velocity = base_velocity + turbulence
            velocity = np.clip(velocity, 0, self.max_velocity)
        
        else:
            velocity = np.zeros_like(self.t)
        
        self.velocity_profile = velocity
        return velocity
    
    def generate_signal(self, profile_type: str = 'pulsatile') -> np.ndarray:
        """
        Generate sinyal Doppler dengan efek pergeseran frekuensi.
        
        Args:
            profile_type: Tipe profil kecepatan
            
        Returns:
            Array sinyal Doppler
        """
        # Generate profil kecepatan
        velocity = self.generate_velocity_profile(profile_type)
        
        # Hitung frekuensi sesaat (carrier + doppler shift)
        doppler_shifts = self.calculate_doppler_shift(velocity)
        instantaneous_freq = self.carrier_frequency + doppler_shifts
        
        # Generate sinyal dengan modulasi fase
        phase = np.cumsum(2 * np.pi * instantaneous_freq / self.sampling_rate)
        
        # Sinyal carrier termodulasi
        carrier = np.cos(phase)
        
        # Envelope dari sinyal Doppler (amplitude lebih tinggi saat aliran cepat)
        envelope = 0.5 + 0.5 * velocity / self.max_velocity
        
        # Modulasi amplitude
        self.signal = carrier * envelope
        
        # Tambahkan noise
        noise = 0.05 * np.random.randn(len(self.t))
        self.signal += noise
        
        return self.signal
    
    def apply_gate(self, gate_depth: float, gate_width: float = 1e-6) -> np.ndarray:
        """
        Aplikasikan gating pada sinyal (sampling pada kedalaman tertentu).
        
        Args:
            gate_depth: Kedalaman gating dalam mm
            gate_width: Lebar gating dalam detik
            
        Returns:
            Sinyal Doppler tergated
        """
        if self.signal is None:
            self.generate_signal()
        
        # Konversi kedalaman menjadi indeks time
        gate_time = 2 * gate_depth / 1000 / self.speed_of_sound
        gate_idx = int(gate_time * self.sampling_rate)
        gate_width_idx = int(gate_width * self.sampling_rate)
        
        # Aplikasikan gating window
        gated_signal = np.zeros_like(self.signal)
        if gate_idx + gate_width_idx < len(self.signal):
            gated_signal[gate_idx:gate_idx + gate_width_idx] = \
                self.signal[gate_idx:gate_idx + gate_width_idx]
        
        return gated_signal
    
    def estimate_velocity_spectrum(self, fft_size: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimasi spektrum kecepatan dari sinyal Doppler.
        
        Returns:
            Tuple (frequency array, magnitude spectrum)
        """
        if self.signal is None:
            self.generate_signal()
        
        # Windowing
        window = scipy_signal.hann(min(fft_size, len(self.signal)))
        windowed_signal = self.signal[:len(window)] * window
        
        # FFT
        spectrum = np.fft.fft(windowed_signal, fft_size)
        magnitude = np.abs(spectrum)
        
        # Frequency axis
        freqs = np.fft.fftfreq(fft_size, 1/self.sampling_rate)
        
        # Kembali hanya frekuensi positif
        positive_freqs = freqs[:fft_size//2]
        magnitude = magnitude[:fft_size//2]
        
        return positive_freqs, magnitude
    
    def plot(self, show_velocity: bool = True, show_spectrum: bool = True):
        """
        Plot sinyal dan analisis Doppler.
        
        Args:
            show_velocity: Tampilkan profil kecepatan
            show_spectrum: Tampilkan spektrum
        """
        if self.signal is None:
            self.generate_signal()
        
        fig, axes = plt.subplots(2 if show_spectrum else 1, 2 if show_velocity else 1, 
                                  figsize=(14, 8))
        
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot sinyal Doppler
        t_display = self.t[:min(1000, len(self.t))]  # Display hanya 1000 points
        signal_display = self.signal[:len(t_display)]
        
        axes[0, 0].plot(t_display * 1e6, signal_display, 'b-', linewidth=0.5)
        axes[0, 0].set_xlabel('Time (μs)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Doppler Signal')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot profil kecepatan
        if show_velocity:
            axes[0, 1].plot(self.t, self.velocity_profile * 100, 'r-', linewidth=1)
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Velocity (cm/s)')
            axes[0, 1].set_title('Blood Flow Velocity')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot spektrum
        if show_spectrum:
            freqs, magnitude = self.estimate_velocity_spectrum()
            freqs_khz = (freqs - self.carrier_frequency) / 1e3  # Relative to carrier
            
            axes[1, 0].plot(freqs_khz, 10 * np.log10(magnitude + 1e-10), 'g-', linewidth=1)
            axes[1, 0].set_xlabel('Frequency relative to carrier (kHz)')
            axes[1, 0].set_ylabel('Magnitude (dB)')
            axes[1, 0].set_title('Frequency Spectrum')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xlim([-100, 100])
        
        plt.tight_layout()
        plt.show()
    
    def get_statistics(self) -> dict:
        """
        Dapatkan statistik aliran Doppler.
        
        Returns:
            Dictionary dengan statistik
        """
        if self.velocity_profile is None:
            self.generate_velocity_profile()
        
        return {
            'mean_velocity': np.mean(self.velocity_profile),
            'max_velocity': np.max(self.velocity_profile),
            'min_velocity': np.min(self.velocity_profile),
            'velocity_std': np.std(self.velocity_profile),
            'mean_doppler_shift': self.calculate_doppler_shift(np.mean(self.velocity_profile)),
            'max_doppler_shift': self.calculate_doppler_shift(np.max(self.velocity_profile)),
        }


if __name__ == "__main__":
    # Contoh penggunaan
    print("=== Doppler Simulator Demo ===")
    
    # Normal pulsatile flow
    doppler = DopplerSimulator(duration=3)
    signal = doppler.generate_signal('pulsatile')
    stats = doppler.get_statistics()
    
    print(f"\nPulsatile Flow Statistics:")
    print(f"  Mean Velocity: {stats['mean_velocity']*100:.1f} cm/s")
    print(f"  Max Velocity: {stats['max_velocity']*100:.1f} cm/s")
    print(f"  Max Doppler Shift: {stats['max_doppler_shift']/1e3:.1f} kHz")
    
    doppler.plot()
