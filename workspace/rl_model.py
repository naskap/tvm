import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TVMEnv(gym.Env):
    """Custom Environment that follows gym interface."""


    def __init__(self, builder, runner, sketches,buffers_per_store = 5, arith_intensity_curve_num_samples = 10):
        super().__init__()



        # Alternative for tiling: A continuous output + a 2^i FloorRound factor
        #    Postprocessing can find closest valid factor? 
        
        # Output for choosing storage granularities: 
        #                           Sample categorical is being output with respect to cooperative fetching (within annotating read reuse)
        #                                                                           and unroll_explicit
        #                           Candidates: [1, 2, 3, 4, 8, 16]
        # Output for choosing unroll factor: The default is [0, 16, 64, 512, 1024]


        self.action_space =  self.action_space = spaces.Dict({
                                "intrinsic_category": spaces.Discrete(3),  # wmma, mma, none
                                "transpose": spaces.Discrete(2),           # True, False
                                "tile_base": spaces.Discrete(15),          # 1-15 for base
                                "tile_multiplier": spaces.Discrete(8),     # 1-8 for multiplier
                                "tile_offset": spaces.Discrete(2),         # Binary offset for odd numbers
                                "storage_granularity": spaces.Discrete(6), # Default storage  [1,2,3,4,8,16]
                                "unroll_factor": spaces.Discrete(5)        # Default unroll factors: [0,16,64,512,1024]
                            })
        

        
            
        # Define constants for feature counts from the C++ code
        group1_feature_count = 57   #
        group2_feature_count = 18 * buffers_per_store
        group3_feature_count = arith_intensity_curve_num_samples # Corresponds to PerStoreFeature's arith_intensity_curve_num_samples param
        group4_feature_count = 4
        group5_feature_count = 3
        group6_feature_count = 8
        total_features = group1_feature_count + group2_feature_count + group3_feature_count + group4_feature_count + group5_feature_count + group6_feature_count
        
        self.obs_staging       = spaces.Box(low=-20.0, high=20.0, shape=(total_features,), dtype=np.float32)
        self.obs_to_stage_next = spaces.Box(low=-20.0, high=20.0, shape=(total_features,), dtype=np.float32)
        self.observation_space = spaces.Tuple(self.obs_staging, self.obs_to_stage_next)


    def step(self, action):
        ...
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        ...
        return observation, info

    def render(self):
        ...

    def close(self):
        ...