import os
import h5py
import pandas as pd

def inspect_h5_keys(data_dir, selected_tag=None): # list all files if selected_tag is None
    """
    Inspect the keys available in HDF5 files in the specified directory.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the HDF5 files
    """

    print("Available keys in each dataset:")
    print("-" * 40)   

    for filename in os.listdir(data_dir):
        if filename.endswith(".h5"):
            dataset_tag = os.path.splitext(filename)[0]
            
            # Skip EB_val and EB_train
            if dataset_tag in ['EB_val', 'EB_train']:
                continue

            # Skip if selected_tag is specified and doesn't match
            if selected_tag is not None and dataset_tag != selected_tag:
                continue
            
            file_path = os.path.join(data_dir, filename)
            try:
                with h5py.File(file_path, 'r') as f:
                    print(f"\n{dataset_tag}:")
                    # Get all keys for this dataset
                    keys = list(f.keys())
                    # Print keys in a formatted way
                    for key in sorted(keys):
                        print(f"  - {key}")
                        if key == 'HLT_data':
                            print(f"    --> shape: {f[key].shape}")
                        if key == 'HLT_latent_reps':
                            print(f"    --> shape: {f[key].shape}")
                        if key == 'raw_HLT_pt':
                            print(f"    --> shape: {f[key].shape}")

            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")



def load_dataframes_from_h5(data_dir):
    """
    Loads from HDF5 files in a directory and put it into a dataframe.
    
    Args:
        data_dir (str): The directory where the HDF5 files are stored.
    
    Returns:
        dataframes (pd.DataFrame): A dictionary of dataframes for each dataset (EB, signals, etc.).
    """

    # Define column names for HLT_data
    columns = ['j0pt', 'j0eta', 'j0phi', 'j1pt', 'j1eta', 'j1phi', 
               'j2pt', 'j2eta', 'j2phi', 'j3pt', 'j3eta', 'j3phi',
               'j4pt', 'j4eta', 'j4phi', 'j5pt', 'j5eta', 'j5phi',
               'e0pt', 'e0eta', 'e0phi', 'e1pt', 'e1eta', 'e1phi',
               'e2pt', 'e2eta', 'e2phi', 'mu0pt', 'mu0eta', 'mu0phi',
               'mu1pt', 'mu1eta', 'mu1phi', 'mu2pt', 'mu2eta', 'mu2phi',
               'ph0pt', 'ph0eta', 'ph0phi', 'ph1pt', 'ph1eta', 'ph1phi',
               'ph2pt', 'ph2eta', 'ph2phi', 'METpt', 'METeta', 'METphi']


    dataframes = {}
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".h5"):
            dataset_tag = os.path.splitext(filename)[0]

            # Skip EB_val and EB_train
            if dataset_tag in ['EB_val', 'EB_train']:
                continue
            
            file_path = os.path.join(data_dir, filename)
            with h5py.File(file_path, 'r') as f:
                # Create main DataFrame with HLT_data
                df = pd.DataFrame(f['HLT_data'][:], columns=columns)

                # Get raw_HLT_pt values
                raw_pt_values = f['raw_HLT_pt'][:]
                raw_pt_df = pd.DataFrame(raw_pt_values, columns=[
                    'j0pt',  'j1pt',  'j2pt',  'j3pt',
                    'j4pt',  'j5pt',  'e0pt',  'e1pt',
                    'e2pt',  'mu0pt', 'mu1pt', 'mu2pt',
                    'ph0pt', 'ph1pt', 'ph2pt', 'METpt'
                ])

                # Replace pt columns with raw values
                for col in raw_pt_df.columns:
                    df[col] = raw_pt_df[col]

                # Add scores and weights
                df['HLT_AD_scores'] = f['HLT_AD_scores'][:]
                df['weights'] = f['weights'][:]
                df['passHLT'] = f['passHLT'][:]
                
                # Create DataFrame for latent representations
                latent_reps = pd.DataFrame(
                    f['HLT_latent_reps'][:], 
                    columns=['LS1', 'LS2', 'LS3', 'LS4']
                )
                
                # Combine with main DataFrame
                df = pd.concat([df, latent_reps], axis=1)
                
                dataframes[dataset_tag] = df
            print(f"Loaded {dataset_tag} from {file_path}")
    
    return dataframes


