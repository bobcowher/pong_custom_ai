import torch
import numpy as np
from game import Pong
from agent import Agent
import cv2
import matplotlib.pyplot as plt

def debug_observation_flipping():
    """Debug observation generation and flipping to ensure symmetry"""
    print("=== DEBUGGING OBSERVATION FLIPPING ===\n")
    
    # Create environment and agent
    env = Pong(player1="ai", player2="ai", render_mode="rgbarray")
    agent = Agent(eval=False)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Process observation
    player_1_obs, player_2_obs = agent.process_observation(obs, clear_stack=True)
    
    # Extract single frames for visualization
    p1_frame = player_1_obs[0].numpy()  # First frame from stack
    p2_frame = player_2_obs[0].numpy()  # First frame from stack
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show original observation
    axes[0].imshow(obs[0].numpy(), cmap='gray')
    axes[0].set_title('Original Observation')
    axes[0].axis('off')
    
    # Show player 1 view
    axes[1].imshow(p1_frame, cmap='gray')
    axes[1].set_title('Player 1 View (Right Side)')
    axes[1].axis('off')
    
    # Show player 2 view
    axes[2].imshow(p2_frame, cmap='gray')
    axes[2].set_title('Player 2 View (Should be flipped)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('observation_debug.png')
    print("Saved observation comparison to observation_debug.png")
    
    # Verify flip is correct
    print(f"\nPlayer 1 obs shape: {player_1_obs.shape}")
    print(f"Player 2 obs shape: {player_2_obs.shape}")
    print(f"Are observations different? {not torch.equal(player_1_obs, player_2_obs)}")
    
    # Check if flip is correct by comparing specific pixels
    # Player 1's paddle should be on the right for P1, left for P2
    p1_right_edge = p1_frame[:, -10:].sum()
    p1_left_edge = p1_frame[:, :10].sum()
    p2_right_edge = p2_frame[:, -10:].sum()
    p2_left_edge = p2_frame[:, :10].sum()
    
    print(f"\nEdge pixel sums:")
    print(f"P1 view - Right edge: {p1_right_edge}, Left edge: {p1_left_edge}")
    print(f"P2 view - Right edge: {p2_right_edge}, Left edge: {p2_left_edge}")


def debug_ball_spawn_pattern():
    """Debug ball spawn pattern to check for directional bias"""
    print("\n=== DEBUGGING BALL SPAWN PATTERN ===\n")
    
    env = Pong(player1="ai", player2="ai", render_mode="rgbarray")
    
    spawn_directions = []
    for i in range(20):
        env.reset()
        initial_vx = env.ball.vx
        spawn_directions.append("LEFT" if initial_vx < 0 else "RIGHT")
        
        # Force ball respawn
        env.ball.spawn()
        
    print("First 20 spawn directions:")
    for i, direction in enumerate(spawn_directions):
        print(f"Spawn {i+1}: {direction}")
    
    left_count = spawn_directions.count("LEFT")
    right_count = spawn_directions.count("RIGHT")
    print(f"\nTotal: {left_count} LEFT, {right_count} RIGHT")
    
    # Check first spawn
    print(f"\nIMPORTANT: First spawn direction is {spawn_directions[0]}")
    if spawn_directions[0] == "LEFT":
        print("This means Player 1 (right side) gets first defensive opportunity!")


def debug_training_sampling():
    """Debug the training sampling pattern"""
    print("\n=== DEBUGGING TRAINING SAMPLING PATTERN ===\n")
    
    print("Training samples from replay buffers alternately:")
    for step in range(10):
        if step % 2 == 0:
            print(f"Step {step}: Sampling from Player 1 buffer")
        else:
            print(f"Step {step}: Sampling from Player 2 buffer")
    
    print("\nThis alternating pattern could create training imbalance!")


def debug_reward_assignment():
    """Debug reward assignment logic"""
    print("\n=== DEBUGGING REWARD ASSIGNMENT ===\n")
    
    env = Pong(player1="ai", player2="ai", render_mode="rgbarray")
    
    # Simulate ball going off left edge
    env.ball.x = -10
    print("Ball position: x = -10 (off left edge)")
    print("Expected: Player 1 scores (+1 reward), Player 2 loses (-1 reward)")
    
    # Simulate ball going off right edge  
    env.ball.x = env.window_width + 10
    print("\nBall position: x = window_width + 10 (off right edge)")
    print("Expected: Player 2 scores (+1 reward), Player 1 loses (-1 reward)")
    
    print("\nPlayer positions:")
    print(f"Player 1 (Green) is on the RIGHT at x = {env.window_width - 2 * (env.window_width / 64)}")
    print(f"Player 2 (Purple) is on the LEFT at x = {env.window_width / 64}")


def debug_initial_positions():
    """Debug initial paddle and ball positions"""
    print("\n=== DEBUGGING INITIAL POSITIONS ===\n")
    
    env = Pong(player1="ai", player2="ai", render_mode="rgbarray")
    env.reset()
    
    print(f"Window dimensions: {env.window_width} x {env.window_height}")
    print(f"\nPlayer 1 paddle (RIGHT/Green):")
    print(f"  Position: x={env.player_1_paddle.x}, y={env.player_1_paddle.y}")
    print(f"  Color: {env.player_1_color}")
    
    print(f"\nPlayer 2 paddle (LEFT/Purple):")
    print(f"  Position: x={env.player_2_paddle.x}, y={env.player_2_paddle.y}")
    print(f"  Color: {env.player_2_color}")
    
    print(f"\nBall initial position: x={env.ball.x}, y={env.ball.y}")
    print(f"Ball initial velocity: vx={env.ball.vx}, vy={env.ball.vy}")
    

def debug_action_effects():
    """Debug how actions affect each player when observations are flipped"""
    print("\n=== DEBUGGING ACTION EFFECTS ===\n")
    
    env = Pong(player1="ai", player2="ai", render_mode="rgbarray")
    env.reset()
    
    print("Action mapping:")
    print("  0 = No movement")
    print("  1 = Move UP")
    print("  2 = Move DOWN")
    
    print("\nWhen Player 2 sees flipped observation:")
    print("  - Their paddle appears on the right (like Player 1)")
    print("  - But actions still control their actual paddle on the left")
    print("  - This should be correct behavior - no remapping needed")


def main():
    """Run all debug checks"""
    debug_observation_flipping()
    debug_ball_spawn_pattern()
    debug_training_sampling()
    debug_reward_assignment()
    debug_initial_positions()
    debug_action_effects()
    
    print("\n=== DIAGNOSIS SUMMARY ===\n")
    print("Most likely causes of Player 2 dominance:")
    print("1. Ball spawn pattern - First serve goes LEFT (toward Player 2)")
    print("   - This gives Player 1 the first defensive opportunity")
    print("   - But subsequent alternation should balance this out")
    print("\n2. Training data sampling - Alternates between P1 and P2 buffers")
    print("   - Players train on different distributions at each step")
    print("   - This could create subtle training imbalances")
    print("\nRecommended fixes:")
    print("1. Randomize first ball spawn direction instead of always going left")
    print("2. Sample from both buffers equally each training step")
    print("3. Or combine both buffers into one shared buffer")


if __name__ == "__main__":
    main()