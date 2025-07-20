import cv2
import numpy as np
import torch

class ObservationDebugger:
    """
    Simple debugging tool to display player observations using OpenCV.
    Shows both player observations side by side for validation.
    """
    
    def __init__(self, scale_factor=6):
        """
        Initialize the observation debugger.
        
        Args:
            scale_factor: Factor to scale up the 84x84 observations for visibility
        """
        self.scale_factor = scale_factor
        self.window_name = "Pong Observations Debug"
        
        # Create the display window
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
    def show_observations(self, player_1_obs, player_2_obs):
        """
        Display both player observations side by side.
        
        Args:
            player_1_obs: Player 1 observation tensor (1, 84, 84)
            player_2_obs: Player 2 observation tensor (1, 84, 84)
        """
        # Convert tensors to numpy arrays
        p1_array = self._tensor_to_numpy(player_1_obs)
        p2_array = self._tensor_to_numpy(player_2_obs)
        
        # Scale up for visibility
        p1_scaled = cv2.resize(p1_array, 
                              (84 * self.scale_factor, 84 * self.scale_factor), 
                              interpolation=cv2.INTER_NEAREST)
        p2_scaled = cv2.resize(p2_array, 
                              (84 * self.scale_factor, 84 * self.scale_factor), 
                              interpolation=cv2.INTER_NEAREST)
        
        # Create combined image (side by side)
        combined = np.hstack([p1_scaled, p2_scaled])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Player 1", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Player 2", (84 * self.scale_factor + 10, 30), 
                   font, 1, (255, 255, 255), 2)
        
        # Add a dividing line
        line_x = 84 * self.scale_factor
        cv2.line(combined, (line_x, 0), (line_x, 84 * self.scale_factor), 
                (128, 128, 128), 2)
        
        # Display the image
        cv2.imshow(self.window_name, combined)
        
        # Allow OpenCV to process events (needed for display)
        cv2.waitKey(1)
    
    def _tensor_to_numpy(self, obs):
        """
        Convert observation tensor to numpy array suitable for display.
        
        Args:
            obs: Observation tensor (1, 84, 84)
            
        Returns:
            numpy array (84, 84) with values 0-255
        """
        if isinstance(obs, torch.Tensor):
            # Move to CPU and remove batch dimension
            array = obs.squeeze(0).cpu().numpy()
        else:
            array = obs.squeeze(0) if len(obs.shape) > 2 else obs
        
        # Convert to uint8 if needed
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        
        return array
    
    def close(self):
        """Close the debug window."""
        cv2.destroyWindow(self.window_name)
    
    def is_window_open(self):
        """Check if the debug window is still open."""
        try:
            # Check if window exists and is visible
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except:
            return False

def create_debug_main():
    """
    Create a simple main function that uses the observation debugger.
    """
    from game import Pong
    from agent import Agent
    
    # Initialize agent and game
    agent = Agent(eval=True)
    env = Pong(render_mode="human", player1="bot", player2="human", 
               bot_difficulty="hard", ai_agent=agent)
    
    # Create debugger
    debugger = ObservationDebugger(scale_factor=6)
    
    print("Observation Debugger Started!")
    print("- Main pygame window: The game")
    print("- OpenCV window: Player observations (Player 1 left, Player 2 right)")
    print("- Close the OpenCV window or press 'q' to stop debugging")
    print("\nControls:")
    print("Player 1: J/K keys (down/up)")
    print("Player 2: AI controlled")
    
    try:
        # Game loop
        while True:
            player_1_action = 0
            player_2_action = 0

            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            
            # Handle OpenCV events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or not debugger.is_window_open():
                break
            
            keys = pygame.key.get_pressed()

            # Handle player inputs
            if env.player1 == "human": 
                if keys[pygame.K_k]:
                    player_1_action = 1
                elif keys[pygame.K_j]:
                    player_1_action = 2
            
            if env.player2 == "human":
                if keys[pygame.K_w]:
                    player_2_action = 1
                elif keys[pygame.K_s]:
                    player_2_action = 2

            # Get bot/AI moves
            if env.player1 == "bot":
                player_1_action = env.get_bot_move(player=1)
            if env.player2 == "bot":
                player_2_action = env.get_bot_move(player=2)
            if env.player1 == "ai":
                player_1_action = env.get_ai_move(player=1)
            if env.player2 == "ai":
                player_2_action = env.get_ai_move(player=2)

            # Execute game step
            observation, player_1_reward, player_2_reward, done, truncated, info = env.step(
                player_1_action, player_2_action)
            
            # Process and display observations for debugging
            player_1_obs, player_2_obs = agent.process_observation(observation)
            debugger.show_observations(player_1_obs, player_2_obs)
            
            if done:
                print("Game Over!")
                break
                
    except KeyboardInterrupt:
        print("\nDebugging stopped by user")
    finally:
        debugger.close()
        pygame.quit()

if __name__ == "__main__":
    import pygame
    create_debug_main()