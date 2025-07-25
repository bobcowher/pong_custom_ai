import torch
import numpy as np
from game import Pong
from agent import Agent
import time

def verify_observation_flipping_with_movement():
    """Verify observation flipping after some game steps to ensure paddles are visible"""
    print("=== VERIFYING OBSERVATION FLIPPING WITH VISIBLE PADDLES ===\n")
    
    # Create environment and agent
    env = Pong(player1="ai", player2="ai", render_mode="rgbarray")
    agent = Agent(eval=False)
    
    # Reset and take a few steps to ensure paddles are rendered
    obs, _ = env.reset()
    
    # Take a few steps to ensure the game is rendered
    for _ in range(5):
        obs, _, _, _, _, _ = env.step(player_1_action=0, player_2_action=0)
    
    # Process observation
    player_1_obs, player_2_obs = agent.process_observation(obs, clear_stack=True)
    
    # Extract single frames
    p1_frame = player_1_obs[0].numpy()
    p2_frame = player_2_obs[0].numpy()
    
    print(f"P1 frame shape: {p1_frame.shape}")
    print(f"P2 frame shape: {p2_frame.shape}")
    
    # Check if observations are properly flipped
    print(f"\nAre observations different? {not torch.equal(player_1_obs[0], player_2_obs[0])}")
    
    # Check paddle positions more carefully
    # For P1: their green paddle should be on the right
    # For P2: after flip, their purple paddle should appear on the right
    
    # Sum pixels in vertical strips to detect paddles
    strip_width = 5
    p1_left_strip = p1_frame[:, :strip_width].sum()
    p1_right_strip = p1_frame[:, -strip_width:].sum()
    p2_left_strip = p2_frame[:, :strip_width].sum()
    p2_right_strip = p2_frame[:, -strip_width:].sum()
    
    print(f"\nPixel sums in edge strips (width={strip_width}):")
    print(f"P1 view - Left: {p1_left_strip:.0f}, Right: {p1_right_strip:.0f}")
    print(f"P2 view - Left: {p2_left_strip:.0f}, Right: {p2_right_strip:.0f}")
    
    # Also check middle area for ball
    mid_start = 40
    mid_end = 44
    p1_middle = p1_frame[:, mid_start:mid_end].sum()
    p2_middle = p2_frame[:, mid_start:mid_end].sum()
    print(f"\nMiddle area sums (for ball detection):")
    print(f"P1 view: {p1_middle:.0f}, P2 view: {p2_middle:.0f}")
    
    # Manual verification
    print("\nExpected behavior:")
    print("- P1 (right player) should see high values on right edge")
    print("- P2 (left player) should see high values on right edge (after flip)")
    print("- Middle values should be equal if ball is centered")
    
    # Print some actual pixel values to debug
    print(f"\nSample pixel values from P1 frame:")
    print(f"Top-left corner: {p1_frame[:5, :5].max()}")
    print(f"Top-right corner: {p1_frame[:5, -5:].max()}")
    
    print(f"\nSample pixel values from P2 frame:")
    print(f"Top-left corner: {p2_frame[:5, :5].max()}")
    print(f"Top-right corner: {p2_frame[:5, -5:].max()}")

if __name__ == "__main__":
    verify_observation_flipping_with_movement()