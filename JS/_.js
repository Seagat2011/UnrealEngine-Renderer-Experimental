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

// Function to handle gamepad connection
function handleGamepadConnected(event) {
    console.log("A gamepad connected:");
    console.log(event.gamepad);
}

// Function to handle gamepad disconnection
function handleGamepadDisconnected(event) {
    console.log("A gamepad disconnected:");
    console.log(event.gamepad);
}

// Function to process gamepad input
function processGamepadInput() {
    const gamepads = navigator.getGamepads();
    if (!gamepads) {
        return;
    }

    const gamepad = gamepads[0]; // Assuming we're using the first connected gamepad
    if (!gamepad) {
        return;
    }

    // Process button inputs
    for (let i = 0; i < gamepad.buttons.length; i++) {
        const button = gamepad.buttons[i];
        if (button.pressed) {
            console.log(`Button ${i} pressed`);
            // Add your button-specific logic here
        }
    }

    // Process analog stick inputs
    const leftStickX = gamepad.axes[0];
    const leftStickY = gamepad.axes[1];
    const rightStickX = gamepad.axes[2];
    const rightStickY = gamepad.axes[3];

    // Example: Check if left stick is pushed significantly to the right
    if (leftStickX > 0.5) {
        console.log("Left stick pushed right");
    }

    // Add more analog stick processing logic here

    // Request the next animation frame
    requestAnimationFrame(processGamepadInput);
}

// Set up event listeners
window.addEventListener("gamepadconnected", handleGamepadConnected);
window.addEventListener("gamepaddisconnected", handleGamepadDisconnected);

// Start processing gamepad input
requestAnimationFrame(processGamepadInput);

/**

# Long Short-Term Memory (LSTM) Architecture (as markdown)

graph LR
    X[Input] --> |x_t| C1[Forget Gate]
    X --> |x_t| C2[Input Gate]
    X --> |x_t| C3[Output Gate]
    H[Hidden State] --> |h_t-1| C1
    H --> |h_t-1| C2
    H --> |h_t-1| C3
    C1 --> D[Cell State]
    C2 --> D
    D --> E[New Cell State]
    E --> F[New Hidden State]
    C3 --> F
    F --> |h_t| Output
    E --> |c_t| CellStateOutput
    F --> |Feedback| H
    Output --> |Feedback| X

# LSTM Architecture Explanation

An LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) 
architecture designed to handle long-term dependencies in sequential data. 
Here's a breakdown of its key components and operations:

## 1. Input (x_t)
- The current input at time step t.

## 2. Previous Hidden State (h_t-1)
- The hidden state from the previous time step.

## 3. Cell State (c_t)
- The internal memory of the LSTM cell.

## 4. Gates
LSTM uses three gates to control the flow of information:

### a. Forget Gate
- Decides what information to discard from the cell state.
- σ(W_f · [h_t-1, x_t] + b_f)

### b. Input Gate
- Decides which new information to store in the cell state.
- σ(W_i · [h_t-1, x_t] + b_i)
- Creates new candidate values: tanh(W_c · [h_t-1, x_t] + b_c)

### c. Output Gate
- Decides what to output based on the cell state.
- σ(W_o · [h_t-1, x_t] + b_o)

## 5. Operations

1. Update the cell state:
   - c_t = (forget gate * c_t-1) + (input gate * new candidate values)

2. Generate the output (hidden state):
   - h_t = output gate * tanh(c_t)

## 6. Output
- The new hidden state (h_t) is used as output and passed to the next time step.

Note: σ represents the sigmoid function, and · represents matrix multiplication.

 */
