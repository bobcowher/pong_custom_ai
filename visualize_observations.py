import torch
import numpy as np
from game import Pong
from agent import Agent
import matplotlib.pyplot as plt
import cv2

def visualize_observations():
    """Create a visual comparison of observations"""
    print("=== CREATING VISUAL OBSERVATION COMPARISON ===\n")
    
    # Create environment and agent
    env = Pong(player1="ai", player2="ai", render_mode="rgbarray")
    agent = Agent(eval=False)
    
    # Reset and take steps to ensure visible game state
    obs, _ = env.reset()
    
    # Take steps with different actions to move paddles
    for i in range(15):
        # Move paddles in opposite directions
        p1_action = 1 if i < 7 else 2  # Up then down
        p2_action = 2 if i < 7 else 1  # Down then up
        obs, p1_reward, p2_reward, done, _, _ = env.step(player_1_action=p1_action, 
                                                          player_2_action=p2_action)
        if i == 10:
            print(f"Step {i}: P1 reward={p1_reward}, P2 reward={p2_reward}")
    
    # Process observation
    player_1_obs, player_2_obs = agent.process_observation(obs, clear_stack=True)
    
    # Extract frames
    raw_frame = obs[0].numpy()
    p1_frame = player_1_obs[0].numpy()
    p2_frame = player_2_obs[0].numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Raw observation
    axes[0, 0].imshow(raw_frame, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Raw Observation (84x84)')
    axes[0, 0].set_xlabel(f'Unique values: {np.unique(raw_frame)}')
    
    # Player 1 view
    axes[0, 1].imshow(p1_frame, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Player 1 View (should see self on right)')
    axes[0, 1].set_xlabel(f'Sum: {p1_frame.sum():.0f}')
    
    # Player 2 view
    axes[1, 0].imshow(p2_frame, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Player 2 View (should see self on right after flip)')
    axes[1, 0].set_xlabel(f'Sum: {p2_frame.sum():.0f}')
    
    # Difference
    diff = np.abs(p1_frame - p2_frame)
    axes[1, 1].imshow(diff, cmap='hot', vmin=0, vmax=255)
    axes[1, 1].set_title('Absolute Difference |P1 - P2|')
    axes[1, 1].set_xlabel(f'Max diff: {diff.max():.0f}, Sum diff: {diff.sum():.0f}')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('observation_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to observation_comparison.png")
    
    # Print detailed analysis
    print("\n=== DETAILED ANALYSIS ===")
    
    # Check if observations are properly flipped
    print(f"\nAre P1 and P2 observations different? {not torch.equal(player_1_obs[0], player_2_obs[0])}")
    
    # Check edges more carefully
    left_strip = 10
    right_strip = 10
    
    print(f"\nEdge analysis (strip width = {left_strip}):")
    print(f"P1 left edge sum: {p1_frame[:, :left_strip].sum():.0f}")
    print(f"P1 right edge sum: {p1_frame[:, -right_strip:].sum():.0f}")
    print(f"P2 left edge sum: {p2_frame[:, :left_strip].sum():.0f}")
    print(f"P2 right edge sum: {p2_frame[:, -right_strip:].sum():.0f}")
    
    # Check if horizontal flip is correct by comparing pixels
    print("\nPixel correspondence check (for horizontal flip):")
    for i in range(0, 84, 20):
        print(f"Row {i}: P1[{i},0]={p1_frame[i,0]:.0f} should equal P2[{i},83]={p2_frame[i,83]:.0f}")
        print(f"Row {i}: P1[{i},83]={p1_frame[i,83]:.0f} should equal P2[{i},0]={p2_frame[i,0]:.0f}")
    
    # Check paddle positions
    print("\n=== GAME STATE ===")
    print(f"Ball position: x={env.ball.x:.0f}, y={env.ball.y:.0f}")
    print(f"Ball velocity: vx={env.ball.vx}, vy={env.ball.vy}")
    print(f"Player 1 (Green/Right) paddle: x={env.player_1_paddle.x:.0f}, y={env.player_1_paddle.y:.0f}")
    print(f"Player 2 (Purple/Left) paddle: x={env.player_2_paddle.x:.0f}, y={env.player_2_paddle.y:.0f}")
    
    # Find bright pixels (paddles and ball)
    bright_pixels_p1 = np.argwhere(p1_frame > 0)
    bright_pixels_p2 = np.argwhere(p2_frame > 0)
    
    if len(bright_pixels_p1) > 0:
        print(f"\nP1 view - Bright pixels found at columns: {np.unique(bright_pixels_p1[:, 1])}")
    if len(bright_pixels_p2) > 0:
        print(f"P2 view - Bright pixels found at columns: {np.unique(bright_pixels_p2[:, 1])}")

if __name__ == "__main__":
    visualize_observations()