import torch
import numpy as np
import os

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions,
                 input_device, output_device='cpu', frame_stack=4):
        self.mem_size = max_size
        self.mem_ctr  = 0

        override = os.getenv("REPLAY_BUFFER_MEMORY")

        if override and override.lower() in ["cpu", "cuda:0", "cuda:1"]:
            self.input_device = override
        else:
            self.input_device  = input_device

        self.output_device = output_device

        # States (uint8 saves ~4× RAM vs float32)
        self.state_memory      = torch.zeros(
            (max_size, *input_shape), dtype=torch.uint8, device=input_device
        )
        self.next_state_memory = torch.zeros_like(self.state_memory)

        # Actions as scalar indices for torch.gather
        self.action_memory  = torch.zeros(max_size, dtype=torch.int64,
                                          device=input_device)
        self.reward_memory  = torch.zeros(max_size, dtype=torch.float32,
                                          device=input_device)
        self.terminal_memory = torch.zeros(max_size, dtype=torch.bool,
                                           device=input_device)

    # ------------------------------------------------------------------ #

    def can_sample(self, batch_size: int) -> bool:
        """Require at least 5×batch_size transitions before sampling."""
        return self.mem_ctr >= batch_size * 5

    # ------------------------------------------------------------------ #

    def store_transition(self, state, action, reward, next_state, done):
        """Write a transition in-place on `input_device`."""
        idx = self.mem_ctr % self.mem_size

        self.state_memory[idx]      = torch.as_tensor(
            state, dtype=torch.uint8, device=self.input_device)
        self.next_state_memory[idx] = torch.as_tensor(
            next_state, dtype=torch.uint8, device=self.input_device)

        self.action_memory[idx]   = int(action)
        self.reward_memory[idx]   = float(reward)
        self.terminal_memory[idx] = bool(done)

        self.mem_ctr += 1

    # ------------------------------------------------------------------ #

    def sample_buffer(self, batch_size):
        """Return tensors ready for training (on `output_device`)."""
        max_mem = min(self.mem_ctr, self.mem_size)
        batch   = torch.randint(0, max_mem, (batch_size,),
                                device=self.input_device, dtype=torch.int64)

        # Cast / move once, right here
        states      = self.state_memory[batch]     \
                        .to(self.output_device, dtype=torch.float32)
        next_states = self.next_state_memory[batch]\
                        .to(self.output_device, dtype=torch.float32)
        rewards     = self.reward_memory[batch].to(self.output_device)
        dones       = self.terminal_memory[batch].to(self.output_device)

        # **Return actions as 1-D (B,) LongTensor — caller will unsqueeze**
        actions     = self.action_memory[batch].to(self.output_device)

        return states, actions, rewards, next_states, dones
