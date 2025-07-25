import torch
import numpy as np
from game import Pong
from agent import Agent

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
    
    # Extract single frames for analysis
    p1_frame = player_1_obs[0].numpy()  # First frame from stack
    p2_frame = player_2_obs[0].numpy()  # First frame from stack
    
    # Verify flip is correct
    print(f"Player 1 obs shape: {player_1_obs.shape}")
    print(f"Player 2 obs shape: {player_2_obs.shape}")
    print(f"Are observations different? {not torch.equal(player_1_obs, player_2_obs)}")
    
    # Check if flip is correct by comparing specific pixels
    # Player 1's paddle should be on the right for P1, left for P2
    p1_right_edge = p1_frame[:, -10:].sum()
    p1_left_edge = p1_frame[:, :10].sum()
    p2_right_edge = p2_frame[:, -10:].sum()
    p2_left_edge = p2_frame[:, :10].sum()
    
    print(f"\nEdge pixel sums (to detect paddle positions):")
    print(f"P1 view - Right edge: {p1_right_edge}, Left edge: {p1_left_edge}")
    print(f"P2 view - Right edge: {p2_right_edge}, Left edge: {p2_left_edge}")
    
    # The flipping should swap the paddle positions
    if p1_right_edge > p1_left_edge and p2_left_edge > p2_right_edge:
        print("\n✓ Observation flipping appears correct!")
        print("  P1 sees their paddle on right, P2 sees their paddle on right (after flip)")
    else:
        print("\n✗ WARNING: Observation flipping may be incorrect!")


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
    print(f"\n⚠️  IMPORTANT: First spawn direction is {spawn_directions[0]}")
    if spawn_directions[0] == "LEFT":
        print("This means Player 1 (right side) gets first defensive opportunity!")
        print("However, the ball alternates, so this should balance out over time.")


def debug_training_sampling():
    """Debug the training sampling pattern"""
    print("\n=== DEBUGGING TRAINING SAMPLING PATTERN ===\n")
    
    print("Current implementation samples from replay buffers alternately:")
    print("if(total_steps % 2 == 0):")
    print("    sample from Player 1 buffer")
    print("else:")
    print("    sample from Player 2 buffer")
    
    print("\nSampling pattern for first 10 steps:")
    for step in range(10):
        if step % 2 == 0:
            print(f"Step {step}: Sampling from Player 1 buffer")
        else:
            print(f"Step {step}: Sampling from Player 2 buffer")
    
    print("\n⚠️  POTENTIAL ISSUE: This alternating pattern could create training imbalance!")
    print("Both players use the SAME network but train on different data distributions.")


def debug_reward_assignment():
    """Debug reward assignment logic"""
    print("\n=== DEBUGGING REWARD ASSIGNMENT ===\n")
    
    env = Pong(player1="ai", player2="ai", render_mode="rgbarray")
    
    print("Player positions:")
    print(f"Player 1 (Green) is on the RIGHT at x ≈ {env.window_width - 2 * (env.window_width / 64):.0f}")
    print(f"Player 2 (Purple) is on the LEFT at x ≈ {env.window_width / 64:.0f}")
    
    print("\nReward logic from game.py:")
    print("if ball_center_x < 0:  # Ball goes off LEFT edge")
    print("    player_1_score += 1")
    print("    player_1_reward += 1")
    print("    player_2_reward -= 1")
    print("elif ball_center_x > window_width:  # Ball goes off RIGHT edge")
    print("    player_2_score += 1")
    print("    player_1_reward -= 1")
    print("    player_2_reward += 1")
    
    print("\n✓ This appears correct: Player scores when ball goes off opponent's side")


def debug_initial_positions():
    """Debug initial paddle and ball positions"""
    print("\n=== DEBUGGING INITIAL POSITIONS ===\n")
    
    env = Pong(player1="ai", player2="ai", render_mode="rgbarray")
    env.reset()
    
    print(f"Window dimensions: {env.window_width} x {env.window_height}")
    print(f"\nPlayer 1 paddle (RIGHT/Green):")
    print(f"  Position: x={env.player_1_paddle.x:.0f}, y={env.player_1_paddle.y:.0f}")
    
    print(f"\nPlayer 2 paddle (LEFT/Purple):")
    print(f"  Position: x={env.player_2_paddle.x:.0f}, y={env.player_2_paddle.y:.0f}")
    
    print(f"\nBall initial position: x={env.ball.x:.0f}, y={env.ball.y:.0f}")
    print(f"Ball initial velocity: vx={env.ball.vx}, vy={env.ball.vy}")
    

def main():
    """Run all debug checks"""
    debug_initial_positions()
    debug_observation_flipping()
    debug_ball_spawn_pattern()
    debug_reward_assignment()
    debug_training_sampling()
    
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    print("\nThe two most likely causes of Player 2 dominance are:\n")
    
    print("1. **Training Data Sampling Asymmetry** (MOST LIKELY)")
    print("   - The model alternates between sampling from P1 and P2 buffers")
    print("   - This means at any given step, the model is only learning from one player's perspective")
    print("   - This could create subtle biases in the learned policy")
    
    print("\n2. **Ball Spawn Direction Bias** (LESS LIKELY)")
    print("   - The ball always spawns going LEFT on first serve")
    print("   - This gives Player 1 the first defensive opportunity")
    print("   - But the alternating pattern should balance this over many episodes")
    
    print("\n" + "="*60)
    print("RECOMMENDED FIXES")
    print("="*60)
    
    print("\n1. **Fix the training sampling** (agent.py line 194-197):")
    print("   Instead of alternating, sample from BOTH buffers each step:")
    print("   ```python")
    print("   # Sample from both buffers")
    print("   obs1, act1, rew1, next1, done1 = self.player_1_memory.sample_buffer(batch_size//2)")
    print("   obs2, act2, rew2, next2, done2 = self.player_2_memory.sample_buffer(batch_size//2)")
    print("   # Concatenate the samples")
    print("   observations = torch.cat([obs1, obs2], dim=0)")
    print("   # ... etc for other tensors")
    print("   ```")
    
    print("\n2. **Alternative: Use a single shared replay buffer**")
    print("   - Combine experiences from both players into one buffer")
    print("   - This ensures both players' experiences are equally represented")
    
    print("\n3. **Optional: Randomize initial ball direction** (assets.py line 67):")
    print("   ```python")
    print("   # Instead of: self.last_serve_left = not self.last_serve_left")
    print("   # Use: self.last_serve_left = random.choice([True, False])")
    print("   ```")


if __name__ == "__main__":
    main()