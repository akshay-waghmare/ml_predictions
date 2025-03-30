import os
import pandas as pd

def get_all_match_files(base_dirs):
    """
    Traverse the directory structure to find all match CSV files.
    
    Args:
        base_dirs (list): List of base directories to search for match files.
        
    Returns:
        list: List of paths to match CSV files.
    """
    match_files = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file == 'unified_match_data_enriched.csv':
                    match_file_path = os.path.join(root, file)
                    match_files.append(match_file_path)
                    print(f"Found match file: {match_file_path}")  # Debug log
    return match_files

def read_match_data(file_paths):
    """
    Read each CSV file into a DataFrame.
    
    Args:
        file_paths (list): List of paths to match CSV files.
        
    Returns:
        list: List of DataFrames containing match data.
    """
    data_frames = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            data_frames.append(df)
            print(f"Read data from: {file_path}")  # Debug log
        except Exception as e:
            print(f"Error reading {file_path}: {e}")  # Debug log
    return data_frames

def aggregate_data(data_frames):
    """
    Combine all DataFrames into a single DataFrame.
    
    Args:
        data_frames (list): List of DataFrames to combine.
        
    Returns:
        DataFrame: Aggregated DataFrame containing all match data.
    """
    try:
        aggregated_data = pd.concat(data_frames, ignore_index=True)
        print(f"Aggregated data contains {len(aggregated_data)} rows.")  # Debug log
        return aggregated_data
    except Exception as e:
        print(f"Error aggregating data: {e}")  # Debug log
        return pd.DataFrame()  # Return empty DataFrame on error

def main():
    # Define base directories to search for match files
    base_dirs = [
        r'C:\Project\crawler_learning\ipl_prediction_model\ipl_scraper\indian-premier-league-2023-1345038',
        r'C:\Project\crawler_learning\ipl_prediction_model\ipl_scraper\indian-premier-league-2024-1410320'
    ]
    
    # Step 1: Identify all match folders
    match_files = get_all_match_files(base_dirs)
    print(f"Found {len(match_files)} match files.")
    
    # Step 2: Read unified match data
    match_data_frames = read_match_data(match_files)
    print(f"Read {len(match_data_frames)} match data frames.")
    
    # Step 3: Aggregate data
    aggregated_data = aggregate_data(match_data_frames)
    print(f"Aggregated data contains {len(aggregated_data)} rows.")
    
    # Step 4: Handle missing values and ensure venue consistency
    from data_cleaning import handle_missing_values, fill_missing_venue_data
    cleaned_data = handle_missing_values(aggregated_data)
    cleaned_data = fill_missing_venue_data(cleaned_data)
    
    # Save the aggregated data to a CSV file
    output_file = 'aggregated_match_data.csv'
    try:
        cleaned_data.to_csv(output_file, index=False)
        print(f"Aggregated data saved to '{output_file}'.")
    except Exception as e:
        print(f"Error saving aggregated data to '{output_file}': {e}")  # Debug log

if __name__ == "__main__":
    main()
