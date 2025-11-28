"""Input Shaping Analyzer for OctoPrint Plugin Pinput_Shaping"""

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, find_peaks

MAX_BYTES_32 = 2_000_000_000  # ~ 2 gi b


class InputShapingAnalyzer:
    """Class to analyze input shaping data from a CSV file.
    It loads the data, applies low-pass filtering, computes the Power Spectral Density (PSD),
    generates input shapers, applies them, and generates graphs.
    It also provides methods to get recommendations for input shaping commands.
    """

    def __init__(
        self,
        save_dir,
        csv_path,
        damping=0.5,
        cutoff_freq=100,
        axis=None,
        freq_min=10,
        freq_max=200,
        freq_window=8.0,
        logger=None,
    ) -> None:
        """
        Initializes the InputShapingAnalyzer with the given parameters.
        :param save_dir: Directory to save the results.
        :param csv_path: Path to the CSV file containing the raw acceleration data.
        :param
        damping: Damping factor for the input shapers (default is 0.5).
        :param cutoff_freq: Cutoff frequency for the low-pass filter (default is 100 Hz).
        :param axis: Axis to analyze (e.g., "X", "Y", "Z"). If None, it will be set to "X".
        :param logger: Optional logger for logging messages. If None, a default logger will be used.
        """

        self._plugin_logger = logger or logging.getLogger(
            "octoprint.plugins.Pinput_Shaping"
        )
        self.csv_path = csv_path
        self.damping = damping
        self.cutoff_freq = cutoff_freq
        self.axis = (axis or "X").upper()
        self.freq_min = min(freq_min, freq_max)
        self.freq_max = max(freq_min, freq_max)
        self.freq_window = max(1.0, float(freq_window))
        self.result_dir = save_dir
        self.best_shaper = None
        self.base_freq = None
        self.secondary_freqs = []
        self.shaper_results = {}
        self.time = None
        self.raw = None
        self.filtered = None
        self.sampling_rate = None
        self.freqs = None
        self.psd = None
        self._band_mask = None

    def load_data(self) -> None:
        """Loads the data from the CSV file and processes it."""

        self._plugin_logger.info(
            f"Loading data from CSV file {self.csv_path} for axis {self.axis}"
        )
        df = pd.read_csv(self.csv_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # Time
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])

        # selected axis
        axis_col = self.axis.lower()  # "x" o "y" o "z"
        if axis_col not in df.columns:
            raise ValueError(f"Column '{axis_col}' not found in CSV")

        df[axis_col] = pd.to_numeric(df[axis_col], errors="coerce")
        df = df.dropna(subset=[axis_col])

        self.time = df["time"].to_numpy(dtype=np.float64)
        self.raw = df[axis_col].to_numpy(dtype=np.float64)

        self.sampling_rate = 1.0 / np.mean(np.diff(self.time))

    def _get_band_mask(self) -> np.ndarray:
        """Return boolean mask for the user-requested frequency band."""

        mask = (self.freqs >= self.freq_min) & (self.freqs <= self.freq_max)
        if not np.any(mask):
            self._plugin_logger.warning(
                "Requested frequency band has no samples. Falling back to full spectrum."
            )
            mask = np.ones_like(self.freqs, dtype=bool)
        return mask

    def _detect_resonant_peaks(self, psd: np.ndarray, freqs: np.ndarray):
        """Return a list of (frequency, amplitude) tuples ordered by prominence."""

        if len(psd) == 0:
            return []

        prominence = float(np.max(psd)) * 0.05
        peaks, _ = find_peaks(psd, prominence=prominence)

        if peaks.size == 0:
            idx = int(np.argmax(psd))
            return [(float(freqs[idx]), float(psd[idx]))]

        ordering = np.argsort(psd[peaks])[::-1]
        return [
            (float(freqs[peak_idx]), float(psd[peak_idx])) for peak_idx in peaks[ordering]
        ]

    def lowpass_filter(self, data, order=4) -> np.ndarray:
        """Applies a low-pass Butterworth filter to the data.
        :param data: The input data to filter.
        :param order: The order of the Butterworth filter (default is 4).
        :return: The filtered data.
        """

        nyq = 0.5 * self.sampling_rate
        cutoff = min(self.cutoff_freq, nyq * 0.99)
        norm_cutoff = cutoff / nyq
        self._plugin_logger.info(
            f"lowpass_filter: cutoff={cutoff}, nyq={nyq}, norm_cutoff={norm_cutoff}, sampling_rate={self.sampling_rate}"
        )
        if not (0 < norm_cutoff < 1):
            self._plugin_logger.error(
                f"Invalid norm_cutoff: {norm_cutoff} (cutoff={cutoff}, nyq={nyq})"
            )
            raise ValueError(
                f"Digital filter critical frequencies must be 0 < Wn < 1 (got {norm_cutoff}, cutoff={cutoff}, nyq={nyq})"
            )
        b, a = butter(order, norm_cutoff, btype="low")
        return filtfilt(b, a, data)

    def generate_shapers(self, freq) -> dict:
        """Generates input shapers based on the given frequency."""

        t = 1 / freq
        K = np.exp(-self.damping * np.pi / np.sqrt(1 - self.damping**2))
        shapers = {}

        # Zero Vibration (ZV)
        shapers["ZV"] = [(0, 1 / (1 + K)), (t, K / (1 + K))]

        # Zero Vibration Derivative (ZVD)
        denom_zvd = 1 + 2 * K + K**2
        shapers["ZVD"] = [
            (0, 1 / denom_zvd),
            (t, 2 * K / denom_zvd),
            (2 * t, K**2 / denom_zvd),
        ]

        # Zero Vibration Double Derivative (ZVDD)
        denom_zvdd = 1 + 3 * K + 3 * K**2 + K**3
        shapers["ZVDD"] = [
            (0, 1 / denom_zvdd),
            (t, 3 * K / denom_zvdd),
            (2 * t, 3 * K**2 / denom_zvdd),
            (3 * t, K**3 / denom_zvdd),
        ]

        # Modified ZV (MFA)
        shapers["MZV"] = [
            (0,   1 / (1 + K + K**2)),
            (t,   K / (1 + K + K**2)),
            (2*t, K**2 / (1 + K + K**2)),
        ]

        # Extra insensitive (her)
        shapers["EI"] = [
            (0,   1 / (1 + 3*K + 3*K**2 + K**3)),
            (t,   3*K / (1 + 3*K + 3*K**2 + K**3)),
            (2*t, 3*K**2 / (1 + 3*K + 3*K**2 + K**3)),
            (3*t, K**3 / (1 + 3*K + 3*K**2 + K**3)),
        ]

        # 2-Hump no
        shapers["2HUMP_EI"] = [
            (0,   1 / (1 + 4*K + 6*K**2 + 4*K**3 + K**4)),
            (t,   4*K / (1 + 4*K + 6*K**2 + 4*K**3 + K**4)),
            (2*t, 6*K**2 / (1 + 4*K + 6*K**2 + 4*K**3 + K**4)),
            (3*t, 4*K**3 / (1 + 4*K + 6*K**2 + 4*K**3 + K**4)),
            (4*t, K**4 / (1 + 4*K + 6*K**2 + 4*K**3 + K**4)),
        ]

        # 3-Hump no
        shapers["3HUMP_EI"] = [
            (0,   1 / (1 + 6*K + 15*K**2 + 20*K**3 + 15*K**4 + 6*K**5 + K**6)),
            (t,   6*K / (1 + 6*K + 15*K**2 + 20*K**3 + 15*K**4 + 6*K**5 + K**6)),
            (2*t, 15*K**2 / (1 + 6*K + 15*K**2 + 20*K**3 + 15*K**4 + 6*K**5 + K**6)),
            (3*t, 20*K**3 / (1 + 6*K + 15*K**2 + 20*K**3 + 15*K**4 + 6*K**5 + K**6)),
            (4*t, 15*K**4 / (1 + 6*K + 15*K**2 + 20*K**3 + 15*K**4 + 6*K**5 + K**6)),
            (5*t, 6*K**5 / (1 + 6*K + 15*K**2 + 20*K**3 + 15*K**4 + 6*K**5 + K**6)),
            (6*t, K**6 / (1 + 6*K + 15*K**2 + 20*K**3 + 15*K**4 + 6*K**5 + K**6)),
        ]

        return shapers

    def apply_shaper(self, signal, time, shaper) -> np.ndarray:
        """Applies the input shaper to the signal.
        :param signal: The input signal to shape.
        :param time: The time vector corresponding to the signal.
        :param shaper: The input shaper to apply, defined as a list of (delay, amplitude) tuples.
        :return: The shaped signal.
        """

        dt = np.mean(np.diff(time))
        n = len(signal)
        shaped = np.zeros(n)
        for delay, amp in shaper:
            shift = int(np.round(delay / dt))
            if shift < n:
                shaped[shift:] += amp * signal[: n - shift]
        return shaped

    def _shaper_frequency_response(self, shaper, freqs=None) -> np.ndarray:
        """Compute the magnitude response of a shaper for the given frequency grid."""

        if freqs is None:
            freqs = self.freqs
        if freqs is None or len(freqs) == 0:
            return np.array([], dtype=np.float64)

        delays = np.array([d for d, _ in shaper], dtype=np.float64)
        amps = np.array([a for _, a in shaper], dtype=np.float64)
        if delays.size == 0:
            return np.ones_like(freqs)

        exp_term = np.exp(-1j * 2 * np.pi * np.outer(freqs, delays))
        response = exp_term @ amps
        return np.abs(response)

    def compute_psd(self, signal: np.ndarray) -> tuple:
        """Computes the Power Spectral Density (PSD) of the signal using Welch's method.
        :param signal: The input signal to analyze.
        :return: A tuple containing the frequencies and the corresponding PSD values.
        """

        # Adaptive Welch that guarantees not exceeding the limit of 2 gib.
        sig = signal.astype(np.float32, copy=False)

        # starting point
        nperseg = min(4096, len(sig) // 8)
        nperseg = max(nperseg, 256)

        while True:
            n_win = len(sig) - nperseg + 1
            est_mem = n_win * nperseg * sig.itemsize
            if est_mem < MAX_BYTES_32 or nperseg <= 256:
                break
            nperseg //= 2  # reduces half and try again

        self._plugin_logger.debug(
            f"Welch: nperseg={nperseg}, windows={n_win}, "
            f"est_mem={est_mem / 1e6:.1f} MB, len={len(sig)}"
        )

        return welch(sig, fs=self.sampling_rate, nperseg=nperseg)

    def analyze(self) -> str:
        """Analyzes the input shaping data and returns the best shaper."""

        self.load_data()
        try:
            self.filtered = self.lowpass_filter(self.raw)
        except ValueError as e:
            self._plugin_logger.error(f"Lowpass filter failed: {e}")
            raise
        self.freqs, self.psd = self.compute_psd(self.filtered)

        self._band_mask = self._get_band_mask()
        band_freqs = self.freqs[self._band_mask]
        band_psd = self.psd[self._band_mask]

        peaks = self._detect_resonant_peaks(band_psd, band_freqs)
        if not peaks:
            raise ValueError("No resonant peaks detected in the provided data.")

        self.base_freq = peaks[0][0]
        self.secondary_freqs = [p[0] for p in peaks[1:3]]

        EPS = 1e-12
        self.band_energy = float(np.trapz(band_psd, band_freqs))
        self.band_energy = max(self.band_energy, EPS)

        window_mask = (self.freqs >= self.base_freq - self.freq_window) & (
            self.freqs <= self.base_freq + self.freq_window
        )
        if not np.any(window_mask):
            window_mask = self._band_mask
        self._window_mask = window_mask
        window_psd = self.psd[self._window_mask]
        self.window_freqs = self.freqs[self._window_mask]
        self.window_energy = float(np.trapz(window_psd, self.window_freqs))
        self.window_energy = max(self.window_energy, EPS)
        self.original_peak = float(np.max(window_psd)) if window_psd.size else EPS

        shapers = self.generate_shapers(self.base_freq)

        dt = np.mean(np.diff(self.time))
        for name, shaper in shapers.items():
            shaped = self.apply_shaper(self.filtered, self.time, shaper)
            accel = max(np.abs(np.gradient(shaped, dt)))

            response_mag = self._shaper_frequency_response(shaper)
            if response_mag.size != self.freqs.size:
                response_mag = np.resize(response_mag, self.freqs.size)

            residual_psd = self.psd * (response_mag**2)
            band_psd_shaped = residual_psd[self._band_mask]
            window_psd_shaped = residual_psd[self._window_mask]

            band_energy = float(np.trapz(band_psd_shaped, band_freqs))
            band_energy = max(band_energy, EPS)
            window_peak = (
                float(np.max(window_psd_shaped))
                if window_psd_shaped.size
                else float(np.max(band_psd_shaped))
            )

            reduction_db = 10 * np.log10(self.band_energy / band_energy)
            peak_reduction_db = 10 * np.log10(
                (self.original_peak + EPS) / (window_peak + EPS)
            )

            self.shaper_results[name] = {
                "psd": residual_psd,
                "vibr": band_energy,  # backwards compatibility
                "residual_energy": band_energy,
                "residual_ratio": band_energy / self.band_energy,
                "reduction_db": reduction_db,
                "peak_reduction_db": peak_reduction_db,
                "accel": accel,
            }

        self.best_shaper = max(
            self.shaper_results,
            key=lambda s: (
                self.shaper_results[s]["reduction_db"],
                -self.shaper_results[s]["accel"],
            ),
        )
        return self.best_shaper

    def generate_graphs(self) -> tuple:
        """Generates graphs for the original and filtered signals, and the PSD with input shapers.
        :return: A tuple containing the paths to the generated graphs and the shaper results.
        """

        # get the date from csv file which format is Raw_accel_values_AXIS_X_20250416T133919.csv
        date = os.path.basename(self.csv_path).split("_")[-1].split(".")[0]

        # Signal Graph
        signal_path = os.path.join(self.result_dir, f"{self.axis}_signal_{date}.png")
        plt.figure(figsize=(14, 5))
        plt.plot(
            self.time[::50],
            self.raw[::50],
            label="Original",
            alpha=0.4,
            color="#007bff",
        )
        plt.plot(
            self.time[::50],
            self.filtered[::50],
            label="Filtered",
            linewidth=2.0,
            color="#ff7f0e",
        )
        plt.title(f"Signal - Axis {self.axis}", fontsize=14)
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(signal_path, dpi=150)
        plt.close()

        # PSD Graph
        psd_path = os.path.join(self.result_dir, f"{self.axis}_psd_{date}.png")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.freqs, self.psd, label="Original", color="black", linewidth=1.5)

        for name, result in self.shaper_results.items():
            label = (
                f"{name} ({self.base_freq:.1f} Hz)  "
                f"Δ={result['reduction_db']:.1f} dB  "
                f"accel={result['accel']:.1f}"
            )
            ax.plot(
                self.freqs, result["psd"], linestyle="--", linewidth=1.2, label=label
            )

        ax.set_title(f"PSD with Input Shapers - Axis {self.axis}", fontsize=14)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density (PSD)")
        ax.grid(True, linestyle="--", alpha=0.4)
        lower = max(0, self.freq_min - 10)
        upper = max(self.freq_max * 1.1, self.base_freq * 1.2)
        ax.set_xlim(lower, upper)
        ax.set_ylim(0, np.max(self.psd) * 1.1)
        ax.legend(loc="upper right", fontsize=8)
        # Adjust lower space
        plt.subplots_adjust(bottom=0.35)
        # Recommended text
        recommendation_text = (
            f"Recommended: {self.best_shaper} ({self.base_freq:.1f} Hz)\n"
            f"Marlin CMD: M593 {self.axis} F{self.base_freq:.1f} D{self.damping} S{self.best_shaper}"
        )

        # Add the box behind the text
        fig.text(
            0.5,
            0.08,
            recommendation_text,
            ha="right",
            va="bottom",
            fontsize=10,
            zorder=2,
            bbox={"facecolor": 'white', "edgecolor": 'gray', "boxstyle": 'round,pad=0.5'},
        )

        plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leaves space for text at bottom
        plt.savefig(psd_path, dpi=150)
        plt.close()

        return (
            signal_path,
            psd_path,
            self.shaper_results,
            self.best_shaper,
            self.base_freq
        )

    def get_recommendation(self) -> str:
        """Generates a recommendation string for the best input shaper."""

        return f"M593 {self.axis} F{self.base_freq:.1f} D{self.damping} S{self.best_shaper}"

    def get_klipper_recommendation(self) -> str:
        """Return a Klipper-style SET_INPUT_SHAPER command."""

        return (
            f"SET_INPUT_SHAPER SHAPER={self.best_shaper} "
            f"FREQ={self.base_freq:.2f} "
            f"DAMPING={self.damping:.2f}"
        )

    def build_summary(self) -> dict:
        """Build a Klipper-style summary of all shapers."""

        shaper_rows = sorted(
            [
                {
                    "name": name,
                    "residual_ratio": float(result["residual_ratio"]),
                    "reduction_db": float(result["reduction_db"]),
                    "peak_reduction_db": float(result["peak_reduction_db"]),
                    "residual_energy": float(result["residual_energy"]),
                    "accel": float(result["accel"]),
                }
                for name, result in self.shaper_results.items()
            ],
            key=lambda row: row["reduction_db"],
            reverse=True,
        )

        return {
            "axis": self.axis,
            "base_freq": float(self.base_freq),
            "secondary_freqs": [float(freq) for freq in self.secondary_freqs],
            "freq_limits": [float(self.freq_min), float(self.freq_max)],
            "freq_window": float(self.freq_window),
            "band_energy": float(self.band_energy),
            "shapers": shaper_rows,
            "marlin_command": self.get_recommendation(),
            "klipper_command": self.get_klipper_recommendation(),
        }

    def export_summary(self, output_path: str) -> str:
        """Export the summary to a JSON file."""

        data = self.build_summary()
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        return output_path

    # def get_plotly_data(self):
    #     data = {
    #         "time": self.time[::5].tolist(),  # reduces size if necessary
    #         "raw": self.raw[::5].tolist(),
    #         "filtered": self.filtered[::5].tolist(),
    #         "freqs": self.freqs.tolist(),
    #         "psd_original": self.psd.tolist(),
    #         "shapers": {},
    #         "base_freq": round(self.base_freq, 2),
    #         "best_shaper": self.best_shaper
    #     }

    #     for name, result in self.shaper_results.items():
    #         data["shapers"][name] = {
    #             "psd": result["psd"].tolist(),
    #             "vibr": round(result["vibr"], 3),
    #             "accel": round(result["accel"], 2)
    #         }

    #     return data

    def get_plotly_data(self) -> dict:
        """Generates data for Plotly visualization."""

        return {
            "time": [float(t) for t in self.time[::5]],
            "raw": [float(r) for r in self.raw[::5]],
            "filtered": [float(f) for f in self.filtered[::5]],
            "freqs": [float(f) for f in self.freqs],
            "psd_original": [float(p) for p in self.psd],
            "shapers": {
                name: {
                    "psd": [float(p) for p in result["psd"]],
                    "vibr": round(float(result["vibr"]), 3),
                    "reduction_db": round(float(result["reduction_db"]), 2),
                    "peak_reduction_db": round(float(result["peak_reduction_db"]), 2),
                    "residual_ratio": round(float(result["residual_ratio"]), 4),
                    "accel": round(float(result["accel"]), 2),
                }
                for name, result in self.shaper_results.items()
            },
            "base_freq": round(float(self.base_freq), 2),
            "secondary_freqs": [round(float(freq), 2) for freq in self.secondary_freqs],
            "freq_limits": [float(self.freq_min), float(self.freq_max)],
            "freq_window": float(self.freq_window),
            "band_energy": float(self.band_energy),
            "best_shaper": str(self.best_shaper),
            "marlin_cmd": self.get_recommendation(),
            "klipper_cmd": self.get_klipper_recommendation(),
        }
