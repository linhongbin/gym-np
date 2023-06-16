## Idea on Paper Experiments

1. Baselines:
   
    - Dreamer with inputs of whole images (end-to-end MBRL method)
    - Dreamer with inputs of object poses (Pose-estimation MBRL method)
    - DDPG with inputs of object poses (Pose-estimation MFRL method)
    - Dreamer with positional inputs of features (Tracking-features MBRL method)
    - DDPG with inputs of object poses (Tracking-features MFRL method)

2. Investigating Quesions:
    
    - How well can methods adapts to unknown kinematic error

            1. Randomize a fixed transformation offset bewtween the actual and desired psm pose
   
        
    - How well can methods adapts to image noise or occlusion?
  
            1. add gaussian or other noise to original images
            2. partial visible of needle (In simulation, might make needle into several sections)
    
    - Ablation Study
  
        1. Is image preprocessing or behavior cloning necessary?
        2. How does they benefit the data efficiency and adapting the task variation such as scene background?   