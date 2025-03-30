# prediction_display.py
import tkinter as tk
import threading
import time
import os
import pandas as pd
import sys

def create_simple_prediction_display():
    """Create a minimal floating window to display prediction metrics."""
    # Create the main window
    root = tk.Tk()
    root.title("Cricket Match Prediction")
    root.geometry("350x150")
    root.configure(bg="#1e3c72")
    root.attributes("-topmost", True)  # Always on top
    
    # Create frame for prediction values
    frame = tk.Frame(root, bg="#1e3c72", padx=15, pady=15)
    frame.pack(expand=True, fill=tk.BOTH)
    
    # Create prediction display labels
    model_label = tk.Label(
        frame, 
        text="Model Prediction:", 
        font=("Arial", 11), 
        fg="white", 
        bg="#1e3c72",
        anchor="w"
    )
    model_label.grid(row=0, column=0, sticky="w", pady=2)
    
    model_value = tk.Label(
        frame, 
        text="0.0%", 
        font=("Arial", 11, "bold"), 
        fg="#00ff00", 
        bg="#1e3c72",
        anchor="e"
    )
    model_value.grid(row=0, column=1, sticky="e", pady=2)
    
    market_label = tk.Label(
        frame, 
        text="Market Probability:", 
        font=("Arial", 11), 
        fg="white", 
        bg="#1e3c72",
        anchor="w"
    )
    market_label.grid(row=1, column=0, sticky="w", pady=2)
    
    market_value = tk.Label(
        frame, 
        text="0.0%", 
        font=("Arial", 11, "bold"), 
        fg="#00ff00", 
        bg="#1e3c72",
        anchor="e"
    )
    market_value.grid(row=1, column=1, sticky="e", pady=2)
    
    edge_label = tk.Label(
        frame, 
        text="Model Edge:", 
        font=("Arial", 11), 
        fg="white", 
        bg="#1e3c72",
        anchor="w"
    )
    edge_label.grid(row=2, column=0, sticky="w", pady=2)
    
    edge_value = tk.Label(
        frame, 
        text="0.0%", 
        font=("Arial", 11, "bold"), 
        fg="#ffffff", 
        bg="#1e3c72",
        anchor="e"
    )
    edge_value.grid(row=2, column=1, sticky="e", pady=2)
    
    # Configure the grid columns
    frame.columnconfigure(0, weight=2)
    frame.columnconfigure(1, weight=1)
    
    # Add status bar
    status_var = tk.StringVar(value="Waiting for data...")
    status_bar = tk.Label(
        root,
        textvariable=status_var,
        font=("Arial", 8),
        fg="#cccccc",
        bg="#152a55",
        anchor="w",
        padx=5,
        pady=2
    )
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    # Keep track of last update time
    last_update_time = [None]
    
    def update_display():
        """Monitor for changes in prediction data and update the display."""
        while True:
            try:
                # Try to find the latest CSV file
                match_dir = find_latest_match_dir()
                
                if match_dir:
                    # Find ball_feeders.csv file
                    csv_files = [f for f in os.listdir(match_dir) if f.endswith('_ball_feeders.csv')]
                    
                    if csv_files:
                        # Get the latest file
                        csv_path = os.path.join(match_dir, csv_files[0])
                        
                        # Check if file has been modified since last check
                        mod_time = os.path.getmtime(csv_path)
                        
                        if last_update_time[0] != mod_time:
                            # Read the last row of the CSV with Python engine to handle inconsistent fields
                            df = pd.read_csv(csv_path, engine='python')
                            if not df.empty:
                                last_row = df.iloc[-1]
                                
                                # Update the display with new values
                                model_prob = last_row.get('model_win_probability', 0) * 100
                                
                                # Calculate market probability correctly
                                market_prob = last_row.get('win_percentage', 0)
                                if last_row.get('favored_team') != last_row.get('batting_team'):
                                    market_prob = 100 - market_prob
                                
                                edge = model_prob - market_prob
                                
                                # Update UI on main thread
                                root.after(0, lambda: model_value.config(text=f"{model_prob:.1f}%"))
                                root.after(0, lambda: market_value.config(text=f"{market_prob:.1f}%"))
                                root.after(0, lambda: edge_value.config(text=f"{edge:+.1f}%"))
                                
                                # Set color based on edge
                                if abs(edge) > 10:  # Significant edge
                                    root.after(0, lambda: edge_value.config(
                                        fg="#ff0000" if edge > 0 else "#00ff00"
                                    ))
                                else:
                                    root.after(0, lambda: edge_value.config(fg="#ffffff"))
                                
                                # Update status bar
                                current_over = f"{int(last_row.get('over_number', 0))}.{int(last_row.get('ball_number', 0))}"
                                status_text = f"Last update: Over {current_over}"
                                root.after(0, lambda t=status_text: status_var.set(t))
                                
                                # Remember this update time
                                last_update_time[0] = mod_time
                    else:
                        root.after(0, lambda: status_var.set("Waiting for ball data..."))
                else:
                    root.after(0, lambda: status_var.set("Waiting for match to start..."))
            
            except Exception as e:
                print(f"Error updating display: {e}")
                root.after(0, lambda e=str(e): status_var.set(f"Error: {e}"))
            
            # Check every 2 seconds
            time.sleep(2)
    
    def find_latest_match_dir():
        """Find the most recently modified match directory."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Look for IPL directories
            ipl_dirs = [d for d in os.listdir(current_dir) if d.startswith("ipl") and os.path.isdir(os.path.join(current_dir, d))]
            
            latest_dir = None
            latest_time = 0
            
            for ipl_dir in ipl_dirs:
                ipl_path = os.path.join(current_dir, ipl_dir)
                match_dirs = [d for d in os.listdir(ipl_path) if os.path.isdir(os.path.join(ipl_path, d))]
                
                for match_dir in match_dirs:
                    match_path = os.path.join(ipl_path, match_dir)
                    ball_by_ball_dir = os.path.join(match_path, "ball_by_ball")
                    
                    if os.path.exists(ball_by_ball_dir):
                        mod_time = os.path.getmtime(ball_by_ball_dir)
                        if mod_time > latest_time:
                            latest_time = mod_time
                            latest_dir = ball_by_ball_dir
            
            return latest_dir
        except Exception as e:
            print(f"Error finding match directory: {e}")
            return None
    
    # Start the update thread
    threading.Thread(target=update_display, daemon=True).start()
    
    # Run the GUI
    root.mainloop()

def start_with_csv(csv_path):
    """Start the display with a specific CSV file."""
    # Create the main window
    root = tk.Tk()
    root.title("Cricket Match Prediction")
    root.geometry("450x200")  # Made wider for more odds info
    root.configure(bg="#1e3c72")
    root.attributes("-topmost", True)  # Always on top
    
    # Create frame for prediction values
    frame = tk.Frame(root, bg="#1e3c72", padx=15, pady=15)
    frame.pack(expand=True, fill=tk.BOTH)
    
    # Create prediction display labels
    model_label = tk.Label(
        frame, 
        text="Model Prediction:", 
        font=("Arial", 11), 
        fg="white", 
        bg="#1e3c72",
        anchor="w"
    )
    model_label.grid(row=0, column=0, sticky="w", pady=2)
    
    model_value = tk.Label(
        frame, 
        text="0.0%", 
        font=("Arial", 11, "bold"), 
        fg="#00ff00", 
        bg="#1e3c72",
        anchor="e"
    )
    model_value.grid(row=0, column=1, sticky="e", pady=2)
    
    # Add model odds
    model_odds_label = tk.Label(
        frame,
        text="(0.0/0.0)",
        font=("Arial", 9),
        fg="#aaaaaa",
        bg="#1e3c72",
        anchor="e"
    )
    model_odds_label.grid(row=0, column=2, sticky="e", padx=(5, 0), pady=2)
    
    market_label = tk.Label(
        frame, 
        text="Market Probability:", 
        font=("Arial", 11), 
        fg="white", 
        bg="#1e3c72",
        anchor="w"
    )
    market_label.grid(row=1, column=0, sticky="w", pady=2)
    
    market_value = tk.Label(
        frame, 
        text="0.0%", 
        font=("Arial", 11, "bold"), 
        fg="#00ff00", 
        bg="#1e3c72",
        anchor="e"
    )
    market_value.grid(row=1, column=1, sticky="e", pady=2)
    
    # Add market odds
    market_odds_label = tk.Label(
        frame,
        text="(0.0/0.0)",
        font=("Arial", 9),
        fg="#aaaaaa",
        bg="#1e3c72",
        anchor="e"
    )
    market_odds_label.grid(row=1, column=2, sticky="e", padx=(5, 0), pady=2)
    
    edge_label = tk.Label(
        frame, 
        text="Model Edge:", 
        font=("Arial", 11), 
        fg="white", 
        bg="#1e3c72",
        anchor="w"
    )
    edge_label.grid(row=2, column=0, sticky="w", pady=2)
    
    edge_value = tk.Label(
        frame, 
        text="0.0%", 
        font=("Arial", 11, "bold"), 
        fg="#ffffff", 
        bg="#1e3c72",
        anchor="e"
    )
    edge_value.grid(row=2, column=1, sticky="e", pady=2)
    
    # Add edge in odds
    edge_odds_label = tk.Label(
        frame,
        text="(+0.0)",
        font=("Arial", 9),
        fg="#aaaaaa",
        bg="#1e3c72",
        anchor="e"
    )
    edge_odds_label.grid(row=2, column=2, sticky="e", padx=(5, 0), pady=2)
    
    # Add team labels
    teams_frame = tk.Frame(frame, bg="#1e3c72", pady=5)
    teams_frame.grid(row=3, column=0, columnspan=3, sticky="ew")
    
    batting_team_var = tk.StringVar(value="Batting Team")
    bowling_team_var = tk.StringVar(value="Bowling Team")
    
    tk.Label(
        teams_frame,
        text="Teams:",
        font=("Arial", 10, "bold"),
        fg="white",
        bg="#1e3c72"
    ).pack(side=tk.LEFT)
    
    tk.Label(
        teams_frame,
        textvariable=batting_team_var,
        font=("Arial", 10),
        fg="#00ff88",
        bg="#1e3c72",
        padx=5
    ).pack(side=tk.LEFT)
    
    tk.Label(
        teams_frame,
        text="vs",
        font=("Arial", 10),
        fg="white",
        bg="#1e3c72",
        padx=3
    ).pack(side=tk.LEFT)
    
    tk.Label(
        teams_frame,
        textvariable=bowling_team_var,
        font=("Arial", 10),
        fg="#00ff88",
        bg="#1e3c72",
        padx=5
    ).pack(side=tk.LEFT)
    
    # Configure the grid columns
    frame.columnconfigure(0, weight=2)
    frame.columnconfigure(1, weight=1)
    frame.columnconfigure(2, weight=1)
    
    # Add status bar
    status_var = tk.StringVar(value=f"Monitoring: {os.path.basename(csv_path)}")
    status_bar = tk.Label(
        root,
        textvariable=status_var,
        font=("Arial", 8),
        fg="#cccccc",
        bg="#152a55",
        anchor="w",
        padx=5,
        pady=2
    )
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    # Legend for odds notation
    legend_var = tk.StringVar(value="Odds: (Batting/Bowling)")
    legend_bar = tk.Label(
        root,
        textvariable=legend_var,
        font=("Arial", 8),
        fg="#aaaaaa",
        bg="#1e3c72",
        anchor="e",
        padx=5
    )
    legend_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    # Keep track of last update time
    last_update_time = [None]
    
    def calculate_team_odds(prob):
        """Calculate decimal odds for both teams from a single probability."""
        if prob <= 0 or prob >= 1:
            return 0.0, 0.0
        
        batting_odds = 1.0 / prob
        bowling_odds = 1.0 / (1.0 - prob)
        
        return batting_odds, bowling_odds
    
    def update_display():
        """Monitor for changes in the specified CSV file and update the display."""
        while True:
            try:
                # Check if file exists and has been modified
                if os.path.exists(csv_path):
                    mod_time = os.path.getmtime(csv_path)
                    
                    if last_update_time[0] != mod_time:
                        # Read the last row of the CSV
                        df = pd.read_csv(csv_path)
                        if not df.empty:
                            last_row = df.iloc[-1]
                            
                            # Update team names
                            batting_team = last_row.get('batting_team', 'Batting Team')
                            bowling_team = last_row.get('bowling_team', 'Bowling Team')
                            root.after(0, lambda t=batting_team: batting_team_var.set(t))
                            root.after(0, lambda t=bowling_team: bowling_team_var.set(t))
                            
                            # Update the display with new values
                            model_prob = last_row.get('model_win_probability', 0) 
                            model_prob_pct = model_prob * 100
                            
                            # Calculate market probability correctly
                            market_prob = last_row.get('win_percentage', 0) / 100
                            if last_row.get('favored_team') != last_row.get('batting_team'):
                                market_prob = 1 - market_prob
                            market_prob_pct = market_prob * 100
                            
                            # Calculate edge
                            edge = model_prob_pct - market_prob_pct
                            
                            # Calculate both teams' odds
                            model_bat_odds, model_bowl_odds = calculate_team_odds(model_prob)
                            market_bat_odds, market_bowl_odds = calculate_team_odds(market_prob)
                            
                            # Calculate edge in batting team odds
                            bat_odds_edge = model_bat_odds - market_bat_odds
                            
                            # Update UI on main thread
                            root.after(0, lambda: model_value.config(text=f"{model_prob_pct:.1f}%"))
                            root.after(0, lambda: model_odds_label.config(text=f"({model_bat_odds:.2f}/{model_bowl_odds:.2f})"))
                            
                            root.after(0, lambda: market_value.config(text=f"{market_prob_pct:.1f}%"))
                            root.after(0, lambda: market_odds_label.config(text=f"({market_bat_odds:.2f}/{market_bowl_odds:.2f})"))
                            
                            root.after(0, lambda: edge_value.config(text=f"{edge:+.1f}%"))
                            root.after(0, lambda: edge_odds_label.config(text=f"({bat_odds_edge:+.2f})"))
                            
                            # Set color based on edge
                            if abs(edge) > 10:  # Significant edge
                                root.after(0, lambda: edge_value.config(
                                    fg="#ff0000" if edge > 0 else "#00ff00"
                                ))
                                # Also color the odds edge
                                root.after(0, lambda: edge_odds_label.config(
                                    fg="#ff0000" if edge > 0 else "#00ff00"
                                ))
                            else:
                                root.after(0, lambda: edge_value.config(fg="#ffffff"))
                                root.after(0, lambda: edge_odds_label.config(fg="#aaaaaa"))
                            
                            # Update status bar
                            current_over = f"{int(last_row.get('over_number', 0))}.{int(last_row.get('ball_number', 0))}"
                            status_text = f"Last update: Over {current_over}"
                            root.after(0, lambda t=status_text: status_var.set(t))
                            
                            # Remember this update time
                            last_update_time[0] = mod_time
                else:
                    root.after(0, lambda: status_var.set(f"Waiting for {os.path.basename(csv_path)}..."))
            
            except Exception as e:
                print(f"Error updating display: {e}")
                root.after(0, lambda e=str(e): status_var.set(f"Error: {e}"))
            
            # Check every 2 seconds
            time.sleep(2)
    
    # Start the update thread
    threading.Thread(target=update_display, daemon=True).start()
    
    # Run the GUI
    root.mainloop()

# Add this function to calculate both teams' odds
def calculate_team_odds(prob):
    """Calculate decimal odds for both teams from a single probability."""
    if prob <= 0 or prob >= 1:
        return 0.0, 0.0
    
    batting_odds = 1.0 / prob
    bowling_odds = 1.0 / (1.0 - prob)
    
    return batting_odds, bowling_odds

if __name__ == "__main__":
    # Check if CSV file is provided as command line argument
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if os.path.exists(csv_path) or os.path.dirname(csv_path):
            print(f"Monitoring CSV file: {csv_path}")
            start_with_csv(csv_path)
        else:
            print(f"CSV file not found: {csv_path}")
            create_simple_prediction_display()
    else:
        create_simple_prediction_display()