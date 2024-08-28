# UnrealEngine-Renderer-Experimental
Experimental generative transformer model based renderer for Unreal Engine.

This renderer intitially prototyped in Javascript / Python uses a generative model to infer the next image to provide to the renderer (in this case, a player character animation) based upon the current game environment and controller input. The goal is to achieve an efficiency which eliminates the IK/FK solver overhead on the CPU and GPU, and the animation-related GPU overhead. Hopefully this optimization can be deployed in an Unreal Engine friendly, C++ compatible, .USD file format.

