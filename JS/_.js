let pixelArrayZ = [
    // 10x010 front
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
];

/**
 * Label the directions in the 3D viewport scene
 */
let _3DFaceLabelsMapStrZ = new Map();

_3DFaceLabelsZ.set('front', 0);
_3DFaceLabelsZ.set('back', 1);
_3DFaceLabelsZ.set('left', 2);
_3DFaceLabelsZ.set('right',3);
_3DFaceLabelsZ.set('bottom', 4);
_3DFaceLabelsZ.set('top', 5);

/**
 *  Simulate the allotted viewable angles and faces for the viewport
 */
let controllerViewportRangeObj = {
    min: 0,
    max: 5,
}

/**
 * Simulate the number of pixelArrayZ objects (ie. animation frames) to be presented onscreen
 */
let animationSequenceArrayZ = {
    start: 0,
    end: 1,
    step: 0.001,
}

/*
Scenarios

Depending on the current viewport perpective, the animation consists of 1000 frame flipbook, 
with each frame rendered,intended for rasterization display in pixelArrayZ!



Theory

There are a number of strategies to reach the rasterization image, 
intended to be displayed to the screen. 

1. Calculate IK/FK resolves for the actual 3D mesh within the environment, 
in realtime, while accounting for player / controller input (requires additional gpu/cpu)

2. Cache all animation perspectives for 3D animation sequence, 
then present the next frame based on controller-input 
and the environment (requires additional logic (cpu) + memory)

3. Use a generative transformer model to infer the next frame to present 
to the screen, based on controller-input and the 
environment (currently for your consideration!)

*/
