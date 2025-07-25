import torch
import numpy as np
from game import Pong
from agent import Agent
import cv2

def save_observation_images():
    """Save actual observation images to disk for visual inspection"""
    print("=== SAVING OBSERVATION IMAGES FOR VISUAL INSPECTION ===\n")
    
    # Create environment and agent
    env = Pong(player1="ai", player2="ai", render_mode="rgbarray")
    agent = Agent(eval=False)
    
    # Reset and take several steps to ensure game is visible
    obs, _ = env.reset()
    
    # Take more steps to ensure paddles and ball are visible
    for i in range(10):
        obs, _, _, _, _, _ = env.step(player_1_action=1 if i < 5 else 2, 
                                       player_2_action=2 if i < 5 else 1)
    
    # Get the raw observation
    raw_obs = obs[0].numpy()
    print(f"Raw observation shape: {raw_obs.shape}")
    print(f"Raw observation unique values: {np.unique(raw_obs)}")
    
    # Process observation
    player_1_obs, player_2_obs = agent.process_observation(obs, clear_stack=True)
    
    # Extract single frames
    p1_frame = player_1_obs[0].numpy()
    p2_frame = player_2_obs[0].numpy()
    
    # Save images
    cv2.imwrite('debug_raw_obs.png', raw_obs)
    cv2.imwrite('debug_p1_view.png', p1_frame)
    cv2.imwrite('debug_p2_view.png', p2_frame)
    
    print("Saved images:")
    print("- debug_raw_obs.png (original observation)")
    print("- debug_p1_view.png (Player 1's view)")
    print("- debug_p2_view.png (Player 2's view - should be horizontally flipped)")
    
    # Also create a side-by-side comparison
    comparison = np.hstack([p1_frame, np.ones((84, 10)) * 127, p2_frame])
    cv2.imwrite('debug_comparison.png', comparison)
    print("- debug_comparison.png (side-by-side comparison)")
    
    # Print some statistics
    print(f"\nP1 frame statistics:")
    print(f"  Min: {p1_frame.min()}, Max: {p1_frame.max()}")
    print(f"  Unique values: {np.unique(p1_frame)}")
    
    print(f"\nP2 frame statistics:")
    print(f"  Min: {p2_frame.min()}, Max: {p2_frame.max()}")
    print(f"  Unique values: {np.unique(p2_frame)}")
    
    # Check specific regions
    print("\nChecking specific regions:")
    print(f"P1 left edge (0-5): {p1_frame[:, :5].sum()}")
    print(f"P1 right edge (79-84): {p1_frame[:, 79:].sum()}")
    print(f"P2 left edge (0-5): {p2_frame[:, :5].sum()}")
    print(f"P2 right edge (79-84): {p2_frame[:, 79:].sum()}")
    
    # Check if flip is working by comparing specific pixels
    print("\nChecking if horizontal flip is working:")
    print(f"P1[0,0] = {p1_frame[0,0]}, P2[0,83] = {p2_frame[0,83]} (should be equal)")
    print(f"P1[0,83] = {p1_frame[0,83]}, P2[0,0] = {p2_frame[0,0]} (should be equal)")

if __name__ == "__main__":
    save_observation_images()