from data_merger import merge_match_data
import os

def test_merge():
    # Set paths for the specific match
    match_dir = r"C:\Project\crawler_learning\crex_scrapper_python\ipl_scraper\ipl-2020-21-1210595\1216492"
    ball_by_ball_dir = os.path.join(match_dir, "ball_by_ball")
    
    try:
        # Run the merge function
        merged_data = merge_match_data(match_dir, ball_by_ball_dir)
        print("\nMerge completed successfully!")
        print(f"Shape of merged data: {merged_data.shape}")
        print("\nFirst few rows of merged data:")
        print(merged_data.head())
        
    except Exception as e:
        print(f"Error during merge: {e}")

if __name__ == "__main__":
    test_merge()