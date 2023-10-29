__all__ = [
    'annotate_peaks',
    'highpass_filter',
    'perform_fft',
    'max_fft_from_output_csv'
]

from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sestoolkit.array.timeseries as ts

def annotate_peaks(frequencies, amplitudes, ax, all_weld_names=None):
    # Find peaks with a prominence threshold
    peaks, _ = find_peaks(amplitudes, prominence=1)  # The prominence parameter helps identify true peaks
    peaks = peaks[np.argsort(amplitudes[peaks])][-3:]  # Sort the peaks by amplitude and select the top 3

    # Annotate the peaks
    for peak in peaks:
        if all_weld_names:  # If a list of weld names is provided, extract the name for the current peak
            weld_name = all_weld_names[peak]
            annotation_text = f'{weld_name}\n({frequencies[peak]:.2f}, {amplitudes[peak]:.2f})'
        else:
            annotation_text = f'({frequencies[peak]:.2f}, {amplitudes[peak]:.2f})'
        
        plt.annotate(annotation_text, 
                    (frequencies[peak], amplitudes[peak]),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')   

def highpass_filter(data, cutoff, fs, order=5):
    """ High-pass filter for the data """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y

def max_fft_from_output_csv(weld_output_filepath, time_step):
    df = pd.read_excel(weld_output_filepath, sheet_name="HSS Array")
    df.drop(columns=['Distance', 'Inner Node ID', 'Outer Node ID'], inplace=True)
    df /= 1e9
    
    fs = 1 / time_step
    all_fftAmp = [ts.fft(highpass_filter(row.values, 0.5, fs))[1] for _, row in df.iterrows()]
    max_fftAmp = np.max(all_fftAmp, axis=0)

    return ts.fft(df.iloc[0].values)[0], max_fftAmp  # Frequency remains the same for all rows