"""
This file is designed to contain the various Python functions used to configure tasks.

The functions will be imported by the __init__.py file in this folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import os
import FemapAPI
import time

import FLife
from .signal_processing import annotate_peaks, max_fft_from_output_csv
from .user_input import get_user_inputs, get_file_path, get_directory_path, prompt_continue_or_exit
from .file_processing import read_file, check_directory_in_directory
from .time_counter import get_elapsed_time

def plot_fft(f, max_fft_amp, save_directory, file_name, freq_limit=None, y_limit=None, subtitle="", display=False, save=False):
    plt.figure(figsize=(15, 6))
    plt.plot(f, max_fft_amp)
    plt.title(f"Max Hot Spot Stress Across All Weld Nodes\n{subtitle}\n{file_name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Hot Spot Stress (MPa)")
    
    annotate_peaks(f, max_fft_amp, plt.gca())  # Call the function to annotate the peaks

    if y_limit:
        plt.ylim(0, y_limit)

    if freq_limit:
        plt.xlim(freq_limit)
        major_locator = MultipleLocator(freq_limit[1]/5)
        plt.gca().xaxis.set_major_locator(major_locator)

        minor_locator = MultipleLocator(freq_limit[1]/10)
        plt.gca().xaxis.set_minor_locator(minor_locator)

        tertiary_locator = MultipleLocator(freq_limit[1]/50)
        minor_ticks = list(set(tertiary_locator.tick_values(freq_limit[0], freq_limit[1])) - set(minor_locator.tick_values(freq_limit[0], freq_limit[1])))
        plt.gca().set_xticks(minor_ticks, minor=True)

        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
    if display:
        plt.show()

    if save:
        output_file = os.path.join(save_directory, f"{file_name}_MaxFFT.png")
        plt.savefig(output_file)
        print(f"Saved data for: {file_name}.")

    plt.close()

def process_weld_file_directory(directory, subtitle, max_freq, plot_summary_graph=None):

    weld_files = [file for file in os.listdir(directory) if file.endswith('_HSS.xlsx')]
    fft_data_collection = []

    # Ensure the plots directory exists and get its path
    save_directory = check_directory_in_directory(directory, "plots")
    
    all_amplitudes = []
    all_frequencies = []
    all_weld_names = []

    for index, filename in enumerate(weld_files):
        f, max_fftAmp = max_fft_from_output_csv(os.path.join(directory, filename))
        all_amplitudes.extend(max_fftAmp)
        all_frequencies.extend(f)
        label_name = os.path.splitext(filename)[0]  # Removing the .xlsx extension
        all_weld_names.extend([label_name for _ in range(len(f))])
        plot_fft(f, max_fftAmp, save_directory, label_name, freq_limit=(0, max_freq), subtitle=subtitle, save=True)
        fft_data_collection.append((f, max_fftAmp, filename))
    
    # Convert lists to numpy arrays for easier indexing
    all_amplitudes = np.array(all_amplitudes)
    all_frequencies = np.array(all_frequencies)

    # Plotting all the data on a single graph
    # Using the "Paired" colormap for colorblind-friendly colors
    colors = plt.cm.Paired(np.linspace(0,1,len(fft_data_collection)))
    plt.figure(figsize=(15, 6))
    for idx, (f, max_fftAmp, filename) in enumerate(fft_data_collection):
        label_name = os.path.splitext(filename)[0]
        plt.plot(f, max_fftAmp, label=label_name, color=colors[idx])

    annotate_peaks(all_frequencies, all_amplitudes, plt.gca(), all_weld_names=all_weld_names)  # Call the function to annotate the peaks

    plt.title(f"Summary FFT Amplitude Across All Nodes and all Welds\n{subtitle}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Hot Spot Stress (MPa)")
    plt.legend(loc='best')
    plt.xlim(0, max_freq)

    # Save and show the summary plot
    plt.savefig(os.path.join(save_directory, "Summary_Plot.png"))
    if plot_summary_graph == 1:  # Only plot the summary graph if one directory is selected
        plt.show()
    else:  
        plt.close()

    print(f"All files in {directory} processed!")

## FEA Post Processing

def match_HSS_nodes(inner_nodes, outer_nodes):
    
    tree = scipy.spatial.KDTree(inner_nodes[:, 1:4])
    
    combined_data = []
    distances = []

    used_inner_node_ids = set()  # Keep track of used inner nodes

    for outer_node in outer_nodes:
        dist, inner_node_id = tree.query(outer_node[1:4])

        # Skip if this inner_node_id has already been used
        if inner_node_id in used_inner_node_ids:
            continue

        used_inner_node_ids.add(inner_node_id)  # Mark this inner node as used

        _inner_match = inner_nodes[inner_node_id, :]
        combined_data.append(np.hstack((_inner_match, outer_node)))
        distances.append(dist)

    combined_data_df = pd.DataFrame(combined_data)
    combined_data_df.columns = ['Inner Node ID', 'Inner X', 'Inner Y', 'Inner Z', 'Outer Node ID', 'Outer X', 'Outer Y', 'Outer Z']

    return combined_data_df, distances

def extract_MPS_HSS(app, start_set, end_set, matched_nodes, stress_vector, name=None):
        
    num_nodes = len(matched_nodes)
    total_iterations = int(end_set) - int(start_set)

    inner_stress_array = np.zeros((num_nodes, total_iterations))
    outer_stress_array = np.zeros((num_nodes, total_iterations))

    # Create a Femap set for inner and outer nodes in matched_nodes
    inner_node_set = app.create_set_of_nodes(node_list=list(matched_nodes['Inner Node ID']))
    outer_node_set = app.create_set_of_nodes(node_list=list(matched_nodes['Outer Node ID']))

    # Create a Femap set for the output sets
    output_sets = app.create_set_of_outputs(output_ids=range(int(start_set), int(end_set)))

    if not name:
        inner_node_name=None
        outer_node_name=None
    else:
        inner_node_name=f"{name} inner node "
        outer_node_name=f"{name} outer node "

    print(f"Fetching {inner_node_name}results from Femap:")

    inner_stress_array_fetched = app.get_node_results(output_sets, inner_node_set, [stress_vector])

    print(f"Fetching {outer_node_name}results from Femap:")
    outer_stress_array_fetched = app.get_node_results(output_sets, outer_node_set, [stress_vector])

    # Convert to NumPy arrays if they are DataFrames
    if isinstance(inner_stress_array_fetched, pd.DataFrame):
        inner_stress_array_fetched = inner_stress_array_fetched.to_numpy()
    if isinstance(outer_stress_array_fetched, pd.DataFrame):
        outer_stress_array_fetched = outer_stress_array_fetched.to_numpy()

    # Create a dictionary mapping node IDs to their index positions in the original DataFrame
    inner_node_id_to_index = {node_id: index for index, node_id in enumerate(matched_nodes['Inner Node ID'])}
    outer_node_id_to_index = {node_id: index for index, node_id in enumerate(matched_nodes['Outer Node ID'])}

    # Sort node IDs in descending order (the order in fetched arrays)
    sorted_inner_node_ids = sorted(matched_nodes['Inner Node ID'], reverse=True)
    sorted_outer_node_ids = sorted(matched_nodes['Outer Node ID'], reverse=True)

    # Create index arrays for reordering
    inner_index_mapping = [inner_node_id_to_index[node_id] for node_id in sorted_inner_node_ids]
    outer_index_mapping = [outer_node_id_to_index[node_id] for node_id in sorted_outer_node_ids]

    # Then proceed with the reordering
    try:
        inner_stress_array = inner_stress_array_fetched[np.array(inner_index_mapping), :]
    except IndexError as e:
        print("IndexError encountered:", e)

    outer_stress_array = outer_stress_array_fetched[np.array(outer_index_mapping), :]

    # Compute HSS and print diagnostics:
    hss_array = 1.67 * inner_stress_array - 0.67 * outer_stress_array
    print(f"HSS array shape: {hss_array.shape}")
    print(f"HSS array min, max, mean values: {np.min(hss_array)}, {np.max(hss_array)}, {np.mean(hss_array)}")
    
    return inner_stress_array, outer_stress_array, hss_array

def save_weld_HSS_summary(results, constants, save_directory):
    results_df = pd.DataFrame(list(results.items()), columns=['Weld', 'Max HSS (MPa)'])
    results_df['Constants'] = constants[:len(results_df)]

    results_df['Scaled HSS (MPa)'] = results_df['Max HSS (MPa)'] * results_df['Constants']
    results_df['Scaled HSS pk-pk (MPa)'] = results_df['Scaled HSS (MPa)'] * 2

    summary_save_filepath = os.path.join(save_directory, "Weld_Stress_Summary.xlsx")
    try:
        with pd.ExcelWriter(summary_save_filepath) as writer:
            results_df.to_excel(writer, sheet_name="Weld Stress Summary", index=False)
        os.system(f'start excel "{summary_save_filepath}"')
    except Exception as e:
        print(f"Error saving Weld Stress Summary: {e}.")


    
def read_HSS_node_data(df_weld):
    inner_nodes = df_weld[df_weld["Extrapolation Point"] == "0.4t"].drop(columns=['CSys ID', 'Weld', 'Extrapolation Point']).dropna().to_numpy()
    outer_nodes = df_weld[df_weld["Extrapolation Point"] == "1t"].drop(columns=['CSys ID', 'Weld', 'Extrapolation Point']).dropna().to_numpy()

    return inner_nodes, outer_nodes

def save_single_weld_results(weld, hss_df, inner_stress_df, outer_stress_df, hss_summary_df, matched_nodes, save_directory, savename):
    # Save all DataFrames to the same Excel file but in different worksheets
    
    try:
        with pd.ExcelWriter(os.path.join(save_directory, savename)) as writer:
            
            hss_summary_df.to_excel(writer, sheet_name="HSS Summary", index=False)
            hss_df.to_excel(writer, sheet_name="HSS Array", index=False)
            inner_stress_df.to_excel(writer, sheet_name="0.4t Stress Array", index=False)
            outer_stress_df.to_excel(writer, sheet_name="1t Stress Array", index=False)
            matched_nodes.to_excel(writer, sheet_name="matched_nodes", index=False)
            
        print(f"Saved results for {weld} to {savename}.")

    except Exception as e:
        print(f"Error saving results for {weld}: {e}.")
    
def create_time_array(time_step, no_sets):
    num_decimal_places = len(str(time_step).split('.')[-1])
    time_array = np.arange(0, no_sets*time_step, time_step)
    return num_decimal_places, time_array

def create_HSS_output_dfs(distances_df, num_decimal_places, time_array, hss_array, inner_stress_array, outer_stress_array):
    #1 "HSS", "Inner Stress", "Outer Stress"
    # Convert the extracted output to Data Frames
    hss_df = pd.DataFrame(hss_array)
    hss_df = hss_df.drop(hss_df.columns[0], axis=1)
    inner_stress_df = pd.DataFrame(inner_stress_array)
    outer_stress_df = pd.DataFrame(outer_stress_array)
    
    #2 Create column names for excel sheets
    hss_columns = list(hss_df.columns)
    inner_stress_columns = list(inner_stress_df.columns)
    outer_stress_columns = list(outer_stress_df.columns)
    
    #3 Set node ID columns for inner and outer node sheets only
    inner_stress_columns[0] = f"Inner Node ID"
    outer_stress_columns[0] = f"Outer Node ID"

    for i in range(0, len(hss_columns)):
        hss_columns[i] = f"MPS (HSS); Time {time_array[i]:.{num_decimal_places}f}"
        inner_stress_columns[i+1] = f"MPS (Inner Stress); Time {time_array[i]:.{num_decimal_places}f}"
        outer_stress_columns[i+1] = f"MPS (Outer Stress); Time {time_array[i]:.{num_decimal_places}f}"
    
    hss_df.columns = hss_columns
    inner_stress_df.columns = inner_stress_columns
    outer_stress_df.columns = outer_stress_columns
    
    #4 Put node distances into HSS array    
    hss_temp_df1 = distances_df.join(hss_df)
    hss_temp_df2 = outer_stress_df[["Outer Node ID"]].join(hss_temp_df1)
    hss_df = inner_stress_df[["Inner Node ID"]].join(hss_temp_df2)
    
    return hss_df, inner_stress_df, outer_stress_df

def create_HSS_summary_df(distances_df, hss_df, inner_stress_df, outer_stress_df):
    # Calculate Statistics for HSS summary dataframe
    max_hss_values = hss_df.max(axis=1)
    min_hss_values = hss_df.min(axis=1)
    mean_hss_values = hss_df.mean(axis=1)
    median_hss_values = hss_df.median(axis=1)
    hss_summary_df = pd.DataFrame({
        'Max HSS': max_hss_values,
        'Min HSS': min_hss_values,
        'Mean HSS': mean_hss_values,
        'Median HSS': median_hss_values
    })

    hss_summary_temp_df1 = distances_df.join(hss_summary_df)
    hss_summary_temp_df2 = outer_stress_df[["Outer Node ID"]].join(hss_summary_temp_df1)
    hss_summary_df = inner_stress_df[["Inner Node ID"]].join(hss_summary_temp_df2) 
    
    return hss_summary_df

def add_data_to_overall_summary_df(weld, hss_df, results):  
    # Grab max max HSS for dataframe
    overall_max_hss = hss_df.max().max()
    results[f"{weld}"] = overall_max_hss * 1E-6
    
def post_process_and_save_solid_HSS(time_step, length, start_set, weld_node_filepath, save_directory, constants):
               
    timer_start_time = time.time()
    app = FemapAPI.App()
    print(f"Connected to Femap. Elapsed time: {get_elapsed_time(timer_start_time)}")
    
    no_sets = length / time_step
    end_set = start_set + no_sets
    
    rc = app._rbo.VectorExistsV2(nSetID = start_set, nVectorID = 24000000)

    if rc == 0:
        print(f"Nodal stress vector '24000000' not detected. Convert Max Principal Stress (vector id = 60016) to nodal stress?")
        prompt_continue_or_exit()
        print(f"Beginning Conversion of Max Principal Stress for {start_set} to {end_set}.")
        app.convert_nodal_stress(range(int(start_set), int(end_set)), 60016, 105)
    else:
        print(f"Vector '24000000' detected. Use this vector to calculate HSS?")
        prompt_continue_or_exit()
    
    df = read_file(weld_node_filepath)
    unique_welds = df["Weld"].unique()
    results = {f"{weld}": None for weld in unique_welds}        
    
    for weld in unique_welds:
     
        #1. Get HSS node data
        
        print(f"Beginning extraction of {weld} HSS nodes. Elapsed time: {get_elapsed_time(timer_start_time)}")
        inner_nodes, outer_nodes = read_HSS_node_data(df[df["Weld"] == weld])
        print(f"Extracted {weld} HSS nodes. Elapsed time: {get_elapsed_time(timer_start_time)}")

        #2. Match HSS nodes
        
        print(f"Beginning {weld} node pairing for {len(inner_nodes)} inner nodes and {len(outer_nodes)} outer nodes. Elapsed time: {get_elapsed_time(timer_start_time)}")
        matched_nodes, dist_array = match_HSS_nodes(inner_nodes, outer_nodes)
        print(f"Created {len(matched_nodes['Inner Node ID'])} node pairs for {weld}. Elapsed time: {get_elapsed_time(timer_start_time)}")

        #3. Extract nodal stress from Femap
        
        print(f"Begin extraction of {weld} nodal hotspot stresses. Elapsed time: {get_elapsed_time(timer_start_time)}")
        inner_stress_array, outer_stress_array, hss_array = extract_MPS_HSS(app, start_set, end_set, matched_nodes, 24000000, name = f"{weld}")
        print(f"Completed extraction of {weld} nodal hotspot stresses. Elapsed time: {get_elapsed_time(timer_start_time)}")

        #4. Create DFs for excel sheet
        
        distances_df = pd.DataFrame(dist_array,columns=["Distance"])
        num_decimal_places, time_array = create_time_array(time_step, no_sets)
        hss_df, inner_stress_df, outer_stress_df = create_HSS_output_dfs(distances_df, num_decimal_places, time_array, hss_array, inner_stress_array, outer_stress_array)
        hss_summary_df = create_HSS_summary_df(distances_df, hss_df, inner_stress_df, outer_stress_df) 

        #5. Save weld results to excel file
        
        savename=f"{weld}_HSS.xlsx"
        save_single_weld_results(weld, hss_df, inner_stress_df, outer_stress_df, hss_summary_df, matched_nodes, save_directory, savename)

        #6. Add max to results summary df
        
        add_data_to_overall_summary_df(weld, hss_df, results)  
        
    save_weld_HSS_summary(results, constants, save_directory)
    
def ask_HSS_inputs():
    prompts = ["Timestep for analysis", "Input total time length for analysis", "Input first output set"]
    initialvalues = [0.002, 18, 69]
    datatypes = ["float", "float", "float"]
    time_step, length, start_set = get_user_inputs(prompts, initialvalues, datatypes)
    constants = [0.929, 0.904, 1.169, 0.796, 1.331]
    print(f"Scaling factors imported are: {constants}")
    weld_node_filepath = get_file_path("Select the weld node .xlsx file")
    save_directory = get_directory_path("Select the save directory")
    
    return time_step, length, start_set, weld_node_filepath, save_directory, constants    

def user_input_extract_and_save_HSS():

    time_step, length, start_set, weld_node_filepath, save_directory, constants = ask_HSS_inputs()
    post_process_and_save_solid_HSS(time_step, length, start_set, weld_node_filepath, save_directory, constants)
