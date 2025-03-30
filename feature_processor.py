import pandas as pd
import numpy as np
import pickle
import os
from fuzzywuzzy import fuzz
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
import re

class FeatureProcessor:
    """
    Feature processor for cricket match data to prepare for model prediction.
    Handles feature engineering tasks like player name standardization,
    team encoding, and target encoding.
    """
    def __init__(self, model_dir='model_artifacts'):
        """
        Initialize the FeatureProcessor with required artifacts.
        
        Args:
            model_dir (str): Directory containing model artifacts
        """
        self.model_dir = model_dir
        self.artifacts = self._load_artifacts()
        
        # Initialize feature engineering transformer first
        self.feature_engineering = FeatureEngineeringTransformer(self.artifacts)
        
        # Then create the pipeline using the initialized transformer
        self.feature_pipeline = self._create_pipeline()

    def _load_artifacts(self):
        """Load all required artifacts for feature processing"""
        artifacts = {}
        
        try:
            # Load player name mapping
            player_path = os.path.join(self.model_dir, 'player_mapping.pkl')
            if os.path.exists(player_path):
                with open(player_path, 'rb') as f:
                    artifacts['player_mapping'] = pickle.load(f)
            else:
                artifacts['player_mapping'] = {}
                
            # Load team dummy columns structure
            team_dummies_path = os.path.join(self.model_dir, 'team_dummy_columns.pkl')
            if os.path.exists(team_dummies_path):
                with open(team_dummies_path, 'rb') as f:
                    artifacts['team_dummy_columns'] = pickle.load(f)
            else:
                # Default IPL teams - you might want to expand this
                teams = ['MI', 'CSK', 'RCB', 'KKR', 'DC', 'PBKS', 'RR', 'SRH', 'GT', 'LSG']
                artifacts['team_dummy_columns'] = {
                    'batting_team': [f'batting_team_{team}' for team in teams],
                    'favored_team': [f'favored_team_{team}' for team in teams],
                    'toss_winner': [f'toss_winner_{team}' for team in teams]
                }
            
            # Load target encoding mappings
            target_path = os.path.join(self.model_dir, 'target_encoding_mapping.pkl')
            if os.path.exists(target_path):
                with open(target_path, 'rb') as f:
                    artifacts['target_encoding_mapping'] = pickle.load(f)
            else:
                artifacts['target_encoding_mapping'] = {}
                
            # Load global mean for target encoding fallback
            global_mean_path = os.path.join(self.model_dir, 'global_mean.pkl')
            if os.path.exists(global_mean_path):
                with open(global_mean_path, 'rb') as f:
                    artifacts['global_mean'] = pickle.load(f)
            else:
                artifacts['global_mean'] = 0.5
                
            # Load model feature list (columns expected by model)
            feature_list_path = os.path.join(self.model_dir, 'model_features.pkl')
            if os.path.exists(feature_list_path):
                with open(feature_list_path, 'rb') as f:
                    artifacts['model_features'] = pickle.load(f)
            else:
                artifacts['model_features'] = []
                
            # Add encoding dictionaries for player names
            artifacts['player_encoders'] = {}
            
            # Try to load player encoders if available
            encoder_path = os.path.join(self.model_dir, 'player_encoders.pkl')
            if os.path.exists(encoder_path):
                try:
                    with open(encoder_path, 'rb') as f:
                        artifacts['player_encoders'] = pickle.load(f)
                    print(f"Loaded encoders for {len(artifacts['player_encoders'])} player columns")
                except Exception as e:
                    print(f"Error loading player encoders: {e}")
            
            print(f"Loaded {len(artifacts)} artifact sets for feature processing")
            artifacts['model_dir'] = self.model_dir
            return artifacts
            
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            # Return default empty artifacts
            return {
                'player_mapping': {},
                'team_dummy_columns': {},
                'target_encoding_mapping': {},
                'global_mean': 0.5,
                'model_features': [],
                'player_encoders': {}
            }

    def _create_pipeline(self):
        """Create the feature engineering pipeline"""
        return Pipeline([
            ('feature_engineering', self.feature_engineering),  # Use the already initialized instance
            ('column_alignment', ColumnAlignmentTransformer(self.artifacts.get('model_features', [])))
        ])

    def process_row(self, row_data, update_encoders=False):
        """
        Process a single row of data for prediction.
        
        Args:
            row_data (dict): A dictionary with raw match data features
            update_encoders (bool): Whether to update player encoders with new names
            
        Returns:
            pandas.DataFrame: A processed dataframe ready for model prediction
        """
        # Convert to DataFrame if it's a dictionary
        if isinstance(row_data, dict):
            df = pd.DataFrame([row_data])
        else:
            df = pd.DataFrame([row_data.to_dict()])
        
        # Optionally update encoders with new player names
        if update_encoders:
            self._create_or_update_player_encoders(row_data)
            
        # Apply the pipeline
        try:
            processed = self.feature_pipeline.transform(df)
            return processed
        except Exception as e:
            print(f"Error processing row: {e}")
            return None

    def _create_or_update_player_encoders(self, data):
        """
        Create or update player encoders based on new data.
        
        Args:
            data (dict or DataFrame): New cricket match data
        """
        player_columns = ['batsman1_name', 'batsman2_name', 'bowler1_name', 'bowler2_name']
        
        # Convert to DataFrame if it's a dictionary
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = df.copy()
        
        # Get existing encoders or initialize new ones
        if not hasattr(self, 'player_encoders') or self.player_encoders is None:
            self.player_encoders = {}
        
        # Process each player column
        for col in player_columns:
            if col in df.columns and df[col].notna().any():
                # Get the player name
                player_name = df[col].iloc[0]
                if player_name and player_name != 'Unknown':
                    # Check if encoder exists for this column
                    if col not in self.player_encoders:
                        # Create new encoder
                        self.player_encoders[col] = LabelEncoder()
                        self.player_encoders[col].fit([player_name, 'Unknown'])  # Start with this player and Unknown
                        print(f"Created new encoder for {col}")
                    else:
                        # Update existing encoder with new class if needed
                        if player_name not in self.player_encoders[col].classes_:
                            # Need to refit the encoder with the new class
                            new_classes = np.append(self.player_encoders[col].classes_, player_name)
                            self.player_encoders[col].classes_ = new_classes
                            print(f"Updated encoder for {col} with new player: {player_name}")

        # Save updated encoders if directory exists
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            with open(os.path.join(self.model_dir, 'player_encoders.pkl'), 'wb') as f:
                pickle.dump(self.player_encoders, f)
            print(f"Saved {len(self.player_encoders)} updated player encoders")
        except Exception as e:
            print(f"Error saving player encoders: {e}")
        
        return self.player_encoders

    def create_encoders_from_data(self, data_dict):
        """
        Create or update encoders using new data.
        
        Args:
            data_dict (dict): Raw cricket match data
        """
        player_columns = ['batsman1_name', 'batsman2_name', 'bowler1_name', 'bowler2_name']
        
        for col in player_columns:
            if col in data_dict:
                player_name = data_dict[col]
                
                if player_name and player_name != 'Unknown':
                    if col not in self.player_encoders:
                        # Create new encoder
                        le = LabelEncoder()
                        le.fit([player_name, 'Unknown'])
                        self.player_encoders[col] = le
                        print(f"Created encoder for {col} with player {player_name}")
                        
                    elif player_name not in self.player_encoders[col].classes_:
                        # Update existing encoder
                        new_classes = np.append(self.player_encoders[col].classes_, player_name)
                        self.player_encoders[col].classes_ = new_classes
                        print(f"Added {player_name} to {col} encoder")

    def _load_player_encodings_from_csv(self, player_columns):
        """Load player encodings from CSV files created during training"""
        encoders = {}
        
        for col in player_columns:
            csv_path = f'{col}_encoding_map.csv'
            alt_path = os.path.join(self.model_dir, f'{col}_encoding_map.csv')
            
            # Try both possible locations
            for path in [csv_path, alt_path]:
                if os.path.exists(path):
                    try:
                        # Load the mapping
                        mapping_df = pd.read_csv(path)
                        
                        # Create and configure encoder from the mapping
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        le.classes_ = mapping_df['Player'].values
                        
                        # Store the encoder
                        encoders[col] = le
                        print(f"Loaded player encoding for {col} from {path} with {len(le.classes_)} players")
                        break
                    except Exception as e:
                        print(f"Error loading encoding from {path}: {e}")
        
        return encoders

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Transformer for cricket feature engineering"""
    
    def __init__(self, artifacts):
        self.player_mapping = artifacts.get('player_mapping', {})
        self.team_dummy_columns = artifacts.get('team_dummy_columns', {})
        self.target_encoding_mapping = artifacts.get('target_encoding_mapping', {})
        self.global_mean = artifacts.get('global_mean', 0.5)
        self.model_dir = artifacts.get('model_dir', 'model_artifacts')
        self.player_encoders = artifacts.get('player_encoders', {})
        
        # Initialize smoothed encodings on first use
        self._smoothed_encodings = None
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Make a copy to avoid modifying the original
        result = X.copy()
        
        # Process each row individually
        processed_rows = []
        for _, row in result.iterrows():
            processed = self._process_single_row(row)
            processed_rows.append(processed)
            
        # Combine back into a DataFrame
        return pd.DataFrame(processed_rows)
    
    def _load_player_encodings_from_csv(self, player_columns):
        encoders = {}
        
        for col in player_columns:
            csv_path = f'{col}_encoding_map.csv'
            alt_path = os.path.join(self.model_dir, f'{col}_encoding_map.csv')
            
            # Try both possible locations
            for path in [csv_path, alt_path]:
                if os.path.exists(path):
                    try:
                        # Load the mapping
                        mapping_df = pd.read_csv(path)
                        
                        # Create and configure encoder from the mapping
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        le.classes_ = mapping_df['Player'].values
                        
                        # Store the encoder
                        encoders[col] = le
                        print(f"Loaded player encoding for {col} from {path} with {len(le.classes_)} players")
                        break
                    except Exception as e:
                        print(f"Error loading encoding from {path}: {e}")
        
        return encoders
    
    def _process_single_row(self, row):
        """Process a single cricket data row"""
        # Import fuzz locally to ensure it's available in this method
        from fuzzywuzzy import fuzz
        
        row_dict = row.to_dict()
        
        # Standardize player names
        for col in ['batsman1_name', 'batsman2_name', 'bowler1_name', 'bowler2_name']:
            if col in row_dict:
                player_name = row_dict[col]
                row_dict[col] = self.player_mapping.get(player_name, player_name)
        
        # Create striker relationship features
        striker = row_dict.get('striker_batsman')
        batsman1 = row_dict.get('batsman1_name')
        batsman2 = row_dict.get('batsman2_name')
        
        row_dict['batsman1_on_strike'] = 0
        row_dict['batsman2_on_strike'] = 0
        
        if striker and batsman1 and batsman2:
            # Use fuzzy matching to determine who is on strike
            score1 = fuzz.token_set_ratio(str(striker).lower(), str(batsman1).lower())
            score2 = fuzz.token_set_ratio(str(striker).lower(), str(batsman2).lower())
            
            if score1 > score2 and score1 > 50:
                row_dict['batsman1_on_strike'] = 1
            elif score2 > score1 and score2 > 50:
                row_dict['batsman2_on_strike'] = 1
            else:
                # Default to batsman1 if we can't determine
                row_dict['batsman1_on_strike'] = 1
        
        # Add player performance features
        self._add_player_performance_features(row_dict)
        
        # One-hot encode teams
        self._apply_team_encoding(row_dict)
        
        # Apply target encoding for players
        self._apply_target_encoding(row_dict)
        
        # Apply label encoding for player names
        player_columns = ['batsman1_name', 'batsman2_name', 'bowler1_name', 'bowler2_name']

        # Load encoders from CSV files if not already loaded
        if not hasattr(self, '_csv_encoders_loaded'):
            self._csv_encoders_loaded = True
            csv_encoders = self._load_player_encodings_from_csv(player_columns)
            
            # Update player_encoders with any loaded from CSV
            for col, encoder in csv_encoders.items():
                self.player_encoders[col] = encoder

        # Now process each player column
        for col in player_columns:
            if col in row_dict:
                # Save original name before encoding
                original_name = row_dict[col]
                
                # Initialize encoders if needed
                if not hasattr(self, 'player_encoders'):
                    self.player_encoders = {}
                
                if col not in self.player_encoders:
                    # Create a new encoder with standard players
                    from sklearn.preprocessing import LabelEncoder
                    self.player_encoders[col] = LabelEncoder()
                    initial_players = ['MS Dhoni', 'Virat Kohli', 'Rohit Sharma', 'Jasprit Bumrah', 
                                    'KL Rahul', 'Hardik Pandya', 'Unknown', original_name]
                    self.player_encoders[col].fit(initial_players)
                    print(f"Created new encoder for {col}")
                    
                    # Save this mapping to CSV for future reference
                    try:
                        mapping_df = pd.DataFrame({
                            'Player': self.player_encoders[col].classes_,
                            'Encoded_Value': range(len(self.player_encoders[col].classes_))
                        })
                        mapping_df.to_csv(os.path.join(self.model_dir, f'{col}_encoding_map.csv'), index=False)
                        print(f"Saved {col} encoding mapping to CSV")
                    except Exception as e:
                        print(f"Error saving encoding to CSV: {e}")
                
                # Now encode the player name
                known_values = self.player_encoders[col].classes_
                
                if original_name in known_values:
                    # Direct encoding if name exists
                    encoded_value = self.player_encoders[col].transform([original_name])[0]
                else:
                    # Try fuzzy matching first - import fuzz locally to ensure it's available
                    from fuzzywuzzy import fuzz
                    
                    best_match = None
                    best_score = 0
                    for player in known_values:
                        score = fuzz.ratio(original_name.lower(), player.lower())
                        if score > best_score and score > 80:  # Higher threshold for player names
                            best_score = score
                            best_match = player
                    
                    if best_match:
                        encoded_value = self.player_encoders[col].transform([best_match])[0]
                        print(f"Fuzzy matched {original_name} to {best_match} with score {best_score}")
                    else:
                        # If no match, add as new player
                        from sklearn.preprocessing import LabelEncoder
                        import numpy as np
                        
                        # Create a new encoder with updated classes
                        new_encoder = LabelEncoder()
                        new_classes = np.append(known_values, original_name)
                        new_encoder.fit(new_classes)
                        
                        # Replace the old encoder and get the value
                        self.player_encoders[col] = new_encoder
                        encoded_value = new_encoder.transform([original_name])[0]
                        print(f"Added new player {original_name} to {col} encoder")
                        
                        # Update CSV file with new mapping
                        try:
                            mapping_df = pd.DataFrame({
                                'Player': new_encoder.classes_,
                                'Encoded_Value': range(len(new_encoder.classes_))
                            })
                            mapping_df.to_csv(os.path.join(self.model_dir, f'{col}_encoding_map.csv'), index=False)
                        except Exception as e:
                            print(f"Error updating CSV mapping: {e}")
                
                # Add encoded column
                row_dict[col + '_encoded'] = encoded_value
                
                # Remove original column to match training process
                row_dict.pop(col)
        
        # Handle venue one-hot encoding - similar to how training does it
        if 'venue' in row_dict and row_dict['venue']:
            venue_value = row_dict['venue']
            
            # Get known venue columns from training (if available)
            venue_columns = [col for col in self.team_dummy_columns.get('venue', []) 
                            if col.startswith('venue_')]
            
            # If no venue columns are found in artifacts, try to detect from model features
            if not venue_columns and hasattr(self, 'model_features') and self.model_features:
                venue_columns = [col for col in self.model_features if col.startswith('venue_')]
            
            # If we found expected venue columns
            if venue_columns:
                # Initialize all venue columns to 0
                for venue_col in venue_columns:
                    row_dict[venue_col] = 0
                
                # Try to find matching venue column
                matched = False
                for venue_col in venue_columns:
                    # Extract venue name from column (after 'venue_' prefix)
                    venue_name = venue_col[6:]
                    
                    # Use fuzzy matching for venue names
                    if fuzz.ratio(venue_value.lower(), venue_name.lower()) > 80:
                        row_dict[venue_col] = 1
                        print(f"Matched venue '{venue_value}' to feature column '{venue_col}'")
                        matched = True
                        break
                
                if not matched:
                    print(f"Warning: Could not match venue '{venue_value}' to any known venue columns")
            else:
                # Fallback: No known venue columns, create new ones dynamically
                # (Note: This will likely cause feature mismatch with the model)
                clean_venue_name = re.sub(r'[^a-zA-Z0-9]', ' ', venue_value)
                venue_col = f"venue_{clean_venue_name}"
                row_dict[venue_col] = 1
                print(f"Warning: Created new venue column '{venue_col}' - model may not expect this")
            
            # Remove original venue column
            row_dict.pop('venue')
        else:
            # If venue is missing, set all venue features to 0
            venue_columns = [col for col in self.team_dummy_columns.get('venue', []) 
                            if col.startswith('venue_')]
            if venue_columns:
                for venue_col in venue_columns:
                    row_dict[venue_col] = 0

        # Handle venue one-hot encoding
        if 'venue' in row_dict and row_dict['venue'] and isinstance(row_dict['venue'], str):
            venue_value = row_dict['venue']
            
            # Look for existing venue columns from the team_dummy_columns
            venue_columns = self.team_dummy_columns.get('venue', [])
            
            # If no venue columns found in team_dummy_columns, try to detect from model features
            if not venue_columns and hasattr(self, 'model_features') and self.model_features:
                venue_columns = [col for col in self.model_features if col.startswith('venue_')]
            
            # If we have venue columns from model training
            if venue_columns:
                # Initialize all venue columns to 0
                for venue_col in venue_columns:
                    row_dict[venue_col] = 0
                
                # Try to match the venue to one of our expected venue columns
                venue_matched = False
                
                # First try exact match (after standardizing format)
                venue_clean = re.sub(r'[^a-zA-Z0-9 ,]', '', venue_value)
                expected_venue_col = f"venue_{venue_clean}"
                
                for venue_col in venue_columns:
                    # Extract venue name from column name
                    col_venue = venue_col[6:]  # Remove "venue_" prefix
                    
                    # Check for exact match
                    if venue_col == expected_venue_col:
                        row_dict[venue_col] = 1
                        venue_matched = True
                        print(f"Exact match for venue: {venue_value} -> {venue_col}")
                        break
                    
                    # Try fuzzy matching if no exact match
                    if not venue_matched and fuzz.ratio(venue_clean.lower(), col_venue.lower()) > 80:
                        row_dict[venue_col] = 1
                        venue_matched = True
                        print(f"Fuzzy match for venue: {venue_value} -> {venue_col}")
                        break
                
                if not venue_matched:
                    # Use "Unknown" or default venue if available
                    default_venue = next((col for col in venue_columns if "unknown" in col.lower()), None)
                    if default_venue:
                        row_dict[default_venue] = 1
                        print(f"Using default venue for {venue_value} -> {default_venue}")
                    else:
                        # As a fallback, use first venue in list to avoid breaking the model
                        row_dict[venue_columns[0]] = 1
                        print(f"No match found for venue '{venue_value}', defaulting to {venue_columns[0]}")
            else:
                # No venue columns found - this is a warning situation as model may expect them
                print(f"Warning: No venue columns found in model features, but venue '{venue_value}' provided")
                
            # Remove the original venue column as it was done during training
            row_dict.pop('venue')
        else:
            # If venue is missing or empty, check if we need to add venue columns anyway
            venue_columns = self.team_dummy_columns.get('venue', [])
            if not venue_columns and hasattr(self, 'model_features') and self.model_features:
                venue_columns = [col for col in self.model_features if col.startswith('venue_')]
            
            if venue_columns:
                # Initialize all venue columns to 0
                for venue_col in venue_columns:
                    row_dict[venue_col] = 0

        # Generate historical stats corrections if these were used in training
        # Historical stats as boolean flags with default False
        row_dict['bowler1_historical_average_corrected'] = False
        row_dict['bowler1_historical_strike_rate_corrected'] = False
        row_dict['bowler2_historical_average_corrected'] = False 
        row_dict['bowler2_historical_strike_rate_corrected'] = False
        
        # Add phase-specific features
        #self._add_phase_features(row_dict)
        
        # Remove striker_batsman as we now have relationship features
        if 'striker_batsman' in row_dict:
            row_dict.pop('striker_batsman')
        
        # Process toss_decision binary encoding with robust matching
        if 'toss_decision' in row_dict and row_dict['toss_decision']:
            decision = str(row_dict['toss_decision']).lower()
            
            # Set to 1 if batting first, 0 if fielding first
            if 'bat' in decision:
                row_dict['toss_decision_binary'] = 1
            elif 'field' in decision or 'bowl' in decision or 'chase' in decision:
                row_dict['toss_decision_binary'] = 0
            else:
                # If the string doesn't contain recognizable terms, use default
                print(f"Warning: Unrecognized toss decision format: '{decision}'. Defaulting to 0 (field)")
                row_dict['toss_decision_binary'] = 0
            
            # Remove original column to match training process
            row_dict.pop('toss_decision')
            print(f"Binary encoded toss_decision: '{decision}' → {row_dict['toss_decision_binary']}")
        else:
            # Set default if missing
            row_dict['toss_decision_binary'] = 0
            print("Set default toss_decision_binary=0 (field) due to missing value")
            
        return row_dict
        
    def _add_player_performance_features(self, row):
        """Add derived features based on player performance"""
        # Create run rate and contribution ratios
        if 'batsman1_runs' in row and 'batsman1_balls_faced' in row and row['batsman1_balls_faced'] > 0:
            row['batsman1_strike_rate'] = row['batsman1_runs'] / row['batsman1_balls_faced'] * 100
        else:
            row['batsman1_strike_rate'] = 0
            
        if 'batsman2_runs' in row and 'batsman2_balls_faced' in row and row['batsman2_balls_faced'] > 0:
            row['batsman2_strike_rate'] = row['batsman2_runs'] / row['batsman2_balls_faced'] * 100
        else:
            row['batsman2_strike_rate'] = 0
            
        # Create bowler economy rates
        if 'bowler1_runs_conceded' in row and 'bowler1_overs_bowled' in row and row['bowler1_overs_bowled'] > 0:
            row['bowler1_economy'] = row['bowler1_runs_conceded'] / row['bowler1_overs_bowled']
        else:
            row['bowler1_economy'] = 0
            
        if 'bowler2_runs_conceded' in row and 'bowler2_overs_bowled' in row and row['bowler2_overs_bowled'] > 0:
            row['bowler2_economy'] = row['bowler2_runs_conceded'] / row['bowler2_overs_bowled'] 
        else:
            row['bowler2_economy'] = 0
            
        return row
        
    def _apply_team_encoding(self, row):
        """
        Apply one-hot encoding for team columns exactly like during training.
        Handles both full team names and abbreviated team names.
        """
        # Team columns that need encoding
        team_columns = ['batting_team', 'favored_team', 'toss_winner']
        
        # Standard mapping between full names and abbreviations
        team_name_to_abbr = {
            'mumbai indians': 'MI',
            'chennai super kings': 'CSK',
            'royal challengers bengaluru': 'RCB',
            'royal challengers bangalore': 'RCB',
            'kolkata knight riders': 'KKR',
            'delhi capitals': 'DC',
            'punjab kings': 'PBKS',
            'kings xi punjab': 'PBKS',  # Old name
            'rajasthan royals': 'RR',
            'sunrisers hyderabad': 'SRH',
            'gujarat titans': 'GT',
            'lucknow super giants': 'LSG'
        }
        
        # Set of valid abbreviations for quick checking
        valid_abbrs = {'MI', 'CSK', 'RCB', 'KKR', 'DC', 'PBKS', 'RR', 'SRH', 'GT', 'LSG'}
        
        for col in team_columns:
            if col in row:
                team_val = row.get(col)
                
                if not team_val or team_val == 'Unknown':
                    # Handle empty team values
                    continue
                    
                # Standardize team name to abbreviation
                if isinstance(team_val, str):
                    # Check if already abbreviated
                    if team_val in valid_abbrs:
                        std_team = team_val
                    else:
                        # Try to map from full name to abbreviation
                        std_team = team_name_to_abbr.get(team_val.lower(), team_val)
                        
                        # If still not recognized, try fuzzy matching as last resort
                        if std_team not in valid_abbrs:
                            best_match = None
                            best_score = 0
                            for full_name, abbr in team_name_to_abbr.items():
                                score = fuzz.ratio(team_val.lower(), full_name)
                                if score > best_score and score > 60:
                                    best_score = score
                                    best_match = abbr
                            
                            if best_match:
                                std_team = best_match
                                print(f"Fuzzy matched team '{team_val}' to '{std_team}'")
                    
                    # Create dummy columns for this team feature
                    dummy_cols = self.team_dummy_columns.get(col, [])
                    if not dummy_cols:
                        # If no dummy columns defined, use default teams
                        for team in valid_abbrs:
                            dummy_cols.append(f"{col}_{team}")
                    
                    # Set all dummy columns to 0 initially
                    for dummy in dummy_cols:
                        row[dummy] = 0
                    
                    # Set the matching team column to 1
                    target_col = f"{col}_{std_team}"
                    if target_col in dummy_cols:
                        row[target_col] = 1
                        print(f"Set {target_col} = 1 for team {team_val}")
                    else:
                        # If no match found, use original value to create a new feature
                        print(f"Warning: No match found for team {team_val} in {col}")
                    
                    # Remove original column
                    row.pop(col, None)
                
        return row
        
    def _apply_target_encoding(self, row):
        """Apply target encoding with enhanced logic matching training process"""
        # Load or initialize smoothed target encodings if not already done
        if not hasattr(self, '_smoothed_encodings') or self._smoothed_encodings is None:
            # Try to load from cache first
            cache_path = os.path.join(self.model_dir, 'smoothed_target_encodings.pkl')
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        self._smoothed_encodings = pickle.load(f)
                    print(f"Loaded {len(self._smoothed_encodings)} smoothed target encodings from cache")
                except Exception:
                    # Calculate fresh if loading fails
                    self._smoothed_encodings = self._initialize_target_encodings()
            else:
                # Calculate fresh if no cache exists
                self._smoothed_encodings = self._initialize_target_encodings()
        
        # Split player columns into batsmen and bowlers
        batsman_columns = ['batsman1_name', 'batsman2_name']
        bowler_columns = ['bowler1_name', 'bowler2_name']
        
        # Apply individual bowler target encoding
        for col in bowler_columns:
            if col in row:
                encoded_col = f"{col}_target_encoded"
                player = row[col]
                
                # Try to find player-specific encoding
                key = f"{col}:{player}"
                if self._smoothed_encodings is not None and key in self._smoothed_encodings:
                    row[encoded_col] = self._smoothed_encodings[key]
                else:
                    # Use global mean as fallback
                    row[encoded_col] = self.global_mean
        
        # Apply combined batsman target encoding
        for col in batsman_columns:
            if col in row:
                player = row[col]
                key = f"batsman_avg:{player}"
                
                if self._smoothed_encodings is not None and key in self._smoothed_encodings:
                    # Store the combined encoding if this is batsman1
                    if col == 'batsman1_name':
                        row['batsman_avg_target_encoded'] = self._smoothed_encodings[key]
                        
                    # If this is batsman2 and we haven't set the combined encoding yet
                    elif col == 'batsman2_name' and 'batsman_avg_target_encoded' not in row:
                        row['batsman_avg_target_encoded'] = self._smoothed_encodings[key]
                else:
                    # Individual target encoding as fallback
                    ind_key = f"{col}:{player}"
                    if self._smoothed_encodings is not None and ind_key in self._smoothed_encodings:
                        if col == 'batsman1_name':
                            row['batsman_avg_target_encoded'] = self._smoothed_encodings[ind_key]
                        elif col == 'batsman2_name' and 'batsman_avg_target_encoded' not in row:
                            row['batsman_avg_target_encoded'] = self._smoothed_encodings[ind_key]
                    else:
                        # Global mean as last resort
                        if 'batsman_avg_target_encoded' not in row:
                            row['batsman_avg_target_encoded'] = self.global_mean
        
        # If we have both batsmen, but couldn't find combined encodings, average the individual ones
        if all(f"{col}_target_encoded" in row for col in batsman_columns):
            if 'batsman_avg_target_encoded' not in row:
                b1_encoding = row[f"{batsman_columns[0]}_target_encoded"]
                b2_encoding = row[f"{batsman_columns[1]}_target_encoded"]
                row['batsman_avg_target_encoded'] = (b1_encoding + b2_encoding) / 2
                
                # Clean up individual batsman encodings that might have been added
                for col in batsman_columns:
                    if f"{col}_target_encoded" in row:
                        del row[f"{col}_target_encoded"]
        
        return row
        
    def _initialize_target_encodings(self):
        """
        Initialize target encodings with enhanced logic:
        1. Calculate individual player encodings
        2. Create combined batsman average encoding
        3. Apply proper smoothing based on match counts
        """
        try:
            # Try to load historical match data
            data_file = 'aggregated_match_data_rolling_avg.csv'
            if not os.path.exists(data_file):
                data_file = os.path.join(self.model_dir, 'aggregated_match_data_rolling_avg.csv')
            
            if os.path.exists(data_file):
                print(f"Loading match data from {data_file} for target encoding")
                df = pd.read_csv(data_file)
                
                # Create target variable (1 for batting team win, 0 otherwise)
                if 'batting_team' in df.columns and 'winner' in df.columns:
                    df['is_winner'] = (df['batting_team'] == df['winner']).astype(int)
                    
                    # Calculate global mean win rate
                    global_mean = df['is_winner'].mean()
                    self.global_mean = global_mean
                    print(f"Global win rate: {global_mean:.4f}")
                    
                    # Split player columns into batsmen and bowlers
                    batsman_columns = ['batsman1_name', 'batsman2_name']
                    bowler_columns = ['bowler1_name', 'bowler2_name']
                    player_columns = batsman_columns + bowler_columns
                    
                    # Initialize smoothed mappings dictionary
                    smoothed_mappings = {}
                    temp_mappings = {}  # For storing encodings to calculate combined features
                    
                    # Apply manual target encoding for player columns
                    for col in player_columns:
                        if col in df.columns:
                            # Apply name standardization first
                            df[col] = df[col].fillna('Unknown').map(
                                lambda x: self.player_mapping.get(x, x))
                            
                            # Group by player and calculate stats
                            player_stats = df.groupby(col)['is_winner'].agg(['count', 'mean'])
                            
                            # Apply smoothing - more weight to global mean for players with few matches
                            m = 10  # Smoothing factor - adjust as needed
                            smoothed_means = (player_stats['count'] * player_stats['mean'] + 
                                            m * global_mean) / (player_stats['count'] + m)
                            
                            # Store the individual mappings
                            temp_mappings[col] = smoothed_means.to_dict()
                            
                            # Create mapping dictionary from player to encoded value
                            for player, value in smoothed_means.to_dict().items():
                                if player and player != 'Unknown':
                                    key = f"{col}:{player}"
                                    smoothed_mappings[key] = value
                    
                    # Create combined batsman encoding
                    combined_batsmen = {}
                    for player in set(df['batsman1_name'].dropna().unique()) | set(df['batsman2_name'].dropna().unique()):
                        # Get individual encodings, default to global mean if not found
                        batsman1_encoding = temp_mappings['batsman1_name'].get(player, global_mean)
                        batsman2_encoding = temp_mappings['batsman2_name'].get(player, global_mean)
                        
                        # Store combined average
                        key = f"batsman_avg:{player}"
                        combined_batsmen[key] = (batsman1_encoding + batsman2_encoding) / 2
                        smoothed_mappings[key] = combined_batsmen[key]
                    
                    print(f"Generated {len(smoothed_mappings)} target encodings ({len(combined_batsmen)} combined batsman)")
                    
                    # Save to cache file for future use
                    cache_path = os.path.join(self.model_dir, 'smoothed_target_encodings.pkl')
                    os.makedirs(self.model_dir, exist_ok=True)
                    try:
                        with open(cache_path, 'wb') as f:
                            pickle.dump(smoothed_mappings, f)
                    except Exception as e:
                        print(f"Could not save target encodings to cache: {e}")
                    print(f"Saved {len(smoothed_mappings)} smoothed target encodings to cache")
                    
                    return smoothed_mappings
                else:
                    print("Missing required columns for target encoding calculation")
            else:
                print(f"Match data file not found: {data_file}")
        
        except Exception as e:
            print(f"Error calculating target encodings: {e}")
            import traceback
            traceback.print_exc()
        
        return {}

    def _add_phase_features(self, row):
        """Add features related to match phase"""
        # Check if phase indicators already exist
        if 'powerplay' not in row and 'middle_overs' not in row and 'death_overs' not in row:
            if 'over_number' in row:
                over = row['over_number']
                row['powerplay'] = int(over < 6)
                row['middle_overs'] = int(over >= 6 and over < 16) 
                row['death_overs'] = int(over >= 16)
        
        # Calculate remaining information if needed
        if 'innings_num' in row and row['innings_num'] == 2:
            if 'balls_remaining' in row and 'runs_needed' in row:
                # Add required run rate if not present
                if 'required_run_rate' not in row and row['balls_remaining'] > 0:
                    row['required_run_rate'] = (row['runs_needed'] * 6) / row['balls_remaining']
                    
        return row

class ColumnAlignmentTransformer(BaseEstimator, TransformerMixin):
    """Ensure the dataframe has all columns expected by the model"""
    
    def __init__(self, expected_columns):
        self.expected_columns = expected_columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if not self.expected_columns:
            return X
            
        # Add any missing columns with default values
        for col in self.expected_columns:
            if col not in X.columns:
                X[col] = 0
                
        # Ensure the order matches
        return X[self.expected_columns]

def align_features_for_model(processed_features, model):
    """
    Aligns features to exactly match what the model expects.
    
    Args:
        processed_features: DataFrame with processed features
        model: Trained model with feature_names_in_ attribute
        
    Returns:
        DataFrame with exactly the features the model expects
    """
    import pandas as pd
    
    # Get the exact feature names expected by the model
    if not hasattr(model, 'feature_names_in_'):
        print("Model doesn't have feature_names_in_ attribute!")
        return processed_features
        
    model_features = model.feature_names_in_
    print(f"Model expects exactly {len(model_features)} features")
    
    # Create a DataFrame with exactly the columns the model expects
    aligned_df = pd.DataFrame(columns=model_features)
    aligned_df.loc[0] = 0  # Initialize with zeros
    
    # Copy values from processed features where names match exactly
    for col in model_features:
        if col in processed_features.columns:
            aligned_df.loc[0, col] = processed_features[col].iloc[0]
            
    # Look for close matches for features we couldn't find exact matches for
    missing_features = [col for col in model_features if col not in processed_features.columns]
    if missing_features:
        print(f"Looking for close matches for {len(missing_features)} missing features")
        
        for missing_col in missing_features:
            # Check for common variations
            base_name = missing_col.split('_')[0]
            for col in processed_features.columns:
                # Check if this is likely a match
                if base_name in col:
                    aligned_df.loc[0, missing_col] = processed_features[col].iloc[0]
                    print(f"Matched {missing_col} to {col}")
                    break
    
    return aligned_df

# Example usage:
if __name__ == "__main__":
    # For testing
    processor = FeatureProcessor()
    
    # Sample test data
    test_row = {
        'innings_num': 2,
        'batting_team': 'PBKS',
        'over_number': 5,
        'ball_number': 6,
        'runs_scored': 0,
        'boundaries': 0,
        'dot_balls': 1,
        'wickets': 0,
        'extras': 0,
        'favored_team': 'MI',
        'win_percentage': 96.34,
        'striker_batsman': 'Harpreet',
        'toss_winner': 'PBKS',
        'toss_decision': 'field first',
        'batsman1_name': 'Shashank Singh',
        'batsman1_runs': 12,
        'batsman1_balls_faced': 10,
        'batsman1_fours': 2,
        'batsman1_sixes': 0,
        'batsman2_name': 'Harpreet Singh',
        'batsman2_runs': 12,
        'batsman2_balls_faced': 13,
        'batsman2_fours': 2,
        'batsman2_sixes': 0,
        'bowler1_name': 'Hardik Pandya',
        'bowler1_overs_bowled': 1.0,
        'bowler1_maidens_bowled': 0,
        'bowler1_runs_conceded': 1,
        'bowler1_wickets_taken': 0,
        'bowler2_name': 'Akash Madhwal',
        'bowler2_overs_bowled': 1.0,
        'bowler2_maidens_bowled': 0,
        'bowler2_runs_conceded': 11,
        'bowler2_wickets_taken': 0,
        'venue': 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
        'powerplay': 1,
        'middle_overs': 0,
        'death_overs': 0,
        'current_run_rate': 4.5,
        'required_run_rate': 6.8,
        'projected_score': 145
    }
    
    # Process the row
    processed = processor.process_row(test_row, update_encoders=True)
    if processed is not None:
        print("Processed feature count:", len(processed.columns))
        print("First 10 feature names:", list(processed.columns)[:10])