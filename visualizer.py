import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Callable

class Visualizer:
    """
    Utility class untuk visualisasi sinyal real-time.
    """
    
    @staticmethod
    def plot_signals(t: np.ndarray, 
                    signals: dict,
                    title: str = "Signal Visualization",
                    figsize: tuple = (14, 8),
                    colors: Optional[dict] = None) -> None:
        """
        Plot multiple signals side by side.
        
        Args:
            t: Time array
            signals: Dictionary {signal_name: signal_array}
            title: Plot title
            figsize: Figure size
            colors: Dictionary {signal_name: color}
        """
        num_signals = len(signals)
        fig, axes = plt.subplots(num_signals, 1, figsize=figsize)
        
        if num_signals == 1:
            axes = [axes]
        
        for idx, (name, signal) in enumerate(signals.items()):
            color = colors.get(name, 'b') if colors else 'b'
            axes[idx].plot(t, signal, color=color, linewidth=1)
            axes[idx].set_ylabel(name)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim([t[0], t[-1]])
        
        axes[-1].set_xlabel('Time (s)')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_comparison(t: np.ndarray,
                       signal1: np.ndarray,
                       signal2: np.ndarray,
                       label1: str = "Signal 1",
                       label2: str = "Signal 2",
                       title: str = "Signal Comparison",
                       figsize: tuple = (14, 6)) -> None:
        """
        Bandingkan dua sinyal dalam satu plot.
        
        Args:
            t: Time array
            signal1: First signal
            signal2: Second signal
            label1: Label untuk signal 1
            label2: Label untuk signal 2
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.plot(t, signal1, 'b-', linewidth=1.5, label=label1, alpha=0.7)
        plt.plot(t, signal2, 'r-', linewidth=1.5, label=label2, alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_frequency_domain(freqs: np.ndarray,
                             magnitude: np.ndarray,
                             title: str = "Frequency Spectrum",
                             figsize: tuple = (12, 6),
                             freq_range: Optional[tuple] = None,
                             log_scale: bool = False) -> None:
        """
        Plot spektrum frekuensi.
        
        Args:
            freqs: Frequency array
            magnitude: Magnitude spectrum
            title: Plot title
            figsize: Figure size
            freq_range: Tuple (min_freq, max_freq) untuk zoom
            log_scale: Gunakan log scale untuk magnitude
        """
        plt.figure(figsize=figsize)
        
        if log_scale:
            magnitude_plot = 10 * np.log10(magnitude + 1e-10)
            ylabel = 'Magnitude (dB)'
        else:
            magnitude_plot = magnitude
            ylabel = 'Magnitude'
        
        plt.plot(freqs, magnitude_plot, 'b-', linewidth=1)
        
        if freq_range:
            plt.xlim(freq_range)
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_statistics(stats: dict,
                       title: str = "Signal Statistics",
                       figsize: tuple = (10, 8)) -> None:
        """
        Plot statistik dalam bentuk bar chart.
        
        Args:
            stats: Dictionary dengan nama dan nilai statistik
            title: Plot title
            figsize: Figure size
        """
        # Filter hanya numeric values
        numeric_stats = {k: v for k, v in stats.items() 
                        if isinstance(v, (int, float, np.number))}
        
        if not numeric_stats:
            print("No numeric statistics to plot")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        names = list(numeric_stats.keys())
        values = list(numeric_stats.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
        bars = ax.bar(names, values, color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Value')
        ax.set_title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_2d_scatter(x: np.ndarray,
                       y: np.ndarray,
                       c: Optional[np.ndarray] = None,
                       xlabel: str = "X",
                       ylabel: str = "Y",
                       title: str = "2D Scatter Plot",
                       figsize: tuple = (10, 8),
                       cmap: str = 'viridis') -> None:
        """
        Plot 2D scatter.
        
        Args:
            x: X coordinates
            y: Y coordinates
            c: Color values (optional)
            xlabel: Label untuk X axis
            ylabel: Label untuk Y axis
            title: Plot title
            figsize: Figure size
            cmap: Colormap
        """
        plt.figure(figsize=figsize)
        
        scatter = plt.scatter(x, y, c=c, cmap=cmap, alpha=0.6, s=50)
        
        if c is not None:
            plt.colorbar(scatter, label='Value')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_heatmap(data: np.ndarray,
                    xlabel: str = "X",
                    ylabel: str = "Y",
                    title: str = "Heatmap",
                    figsize: tuple = (10, 8),
                    cmap: str = 'jet') -> None:
        """
        Plot heatmap/image.
        
        Args:
            data: 2D array data
            xlabel: Label untuk X axis
            ylabel: Label untuk Y axis
            title: Plot title
            figsize: Figure size
            cmap: Colormap
        """
        plt.figure(figsize=figsize)
        
        im = plt.imshow(data, aspect='auto', cmap=cmap, origin='lower')
        plt.colorbar(im, label='Value')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.show()


class AnimatedVisualizer:
    """
    Visualizer untuk animasi real-time.
    """
    
    def __init__(self, figsize: tuple = (12, 6), update_interval: int = 50):
        """
        Inisialisasi animated visualizer.
        
        Args:
            figsize: Figure size
            update_interval: Update interval dalam milidetik
        """
        self.fig = None
        self.figsize = figsize
        self.update_interval = update_interval
        self.lines = {}
        self.axes = {}
    
    def create_animation(self, 
                        data_source: Callable,
                        num_frames: int = 100,
                        num_subplots: int = 1) -> FuncAnimation:
        """
        Buat animasi dari data source.
        
        Args:
            data_source: Function yang mengembalikan data baru setiap frame
            num_frames: Jumlah frames
            num_subplots: Jumlah subplot
            
        Returns:
            FuncAnimation object
        """
        self.fig, self.axes = plt.subplots(num_subplots, 1, figsize=self.figsize)
        
        if num_subplots == 1:
            self.axes = [self.axes]
        
        # Initialize lines untuk setiap subplot
        for ax in self.axes:
            line, = ax.plot([], [])
            self.lines[id(ax)] = line
            ax.set_ylim(-2, 2)
            ax.grid(True, alpha=0.3)
        
        anim = FuncAnimation(self.fig, self._update_frame,
                           frames=num_frames,
                           fargs=(data_source,),
                           interval=self.update_interval,
                           blit=False)
        
        return anim
    
    def _update_frame(self, frame, data_source):
        """
        Update frame untuk animasi.
        
        Args:
            frame: Frame number
            data_source: Function untuk mendapatkan data
        """
        data = data_source()
        
        if isinstance(data, dict):
            for idx, (key, values) in enumerate(data.items()):
                if idx < len(self.axes):
                    line = self.lines[id(self.axes[idx])]
                    line.set_data(range(len(values)), values)
                    self.axes[idx].set_xlim(0, len(values))
                    self.axes[idx].set_ylim(min(values) - 0.5, max(values) + 0.5)
        else:
            line = self.lines[id(self.axes[0])]
            line.set_data(range(len(data)), data)
            self.axes[0].set_xlim(0, len(data))
            self.axes[0].set_ylim(min(data) - 0.5, max(data) + 0.5)
        
        return list(self.lines.values())


if __name__ == "__main__":
    # Contoh penggunaan
    print("=== Visualizer Demo ===")
    
    # Generate test data
    t = np.linspace(0, 5, 500)
    signal1 = np.sin(2 * np.pi * 1 * t)
    signal2 = 0.5 * np.cos(2 * np.pi * 2 * t)
    
    # Plot comparison
    Visualizer.plot_comparison(t, signal1, signal2, 
                               label1="sin(2πt)",
                               label2="0.5*cos(4πt)")
