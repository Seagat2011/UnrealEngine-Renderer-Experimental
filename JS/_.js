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

class LSTM {
    constructor({ weights = [], biases = [] } = {}) {
        let self = this;
        let propertyNames = ['f','i','c','o'];

        if((weights.length > 0) 
                && (biases.length > 0)){
            weights.forEach((u,i,me) => self[`W${propertyNames[i]}`] = u);
            biases.forEach((u,i,me) => self[`b${propertyNames[i]}`] = u);
        } else {
            // Preconfigured weights and biases
            this.Wf = [[0.1, 0.2], [0.3, 0.4]];
            this.Wi = [[0.5, 0.6], [0.7, 0.8]];
            this.Wc = [[0.9, 1.0], [1.1, 1.2]];
            this.Wo = [[1.3, 1.4], [1.5, 1.6]];
            this.bf = [0.1, 0.2];
            this.bi = [0.3, 0.4];
            this.bc = [0.5, 0.6];
            this.bo = [0.7, 0.8];
        }
        // Initial hidden state and cell state
        this.h = [0, 0];
        this.c = [0, 0];
    }
  
    sigmoid(x) {
      return 1 / (1 + Math.exp(-x));
    }
  
    tanh(x) {
      return Math.tanh(x);
    }
  
    matrixMultiply(a, b) {
      return a.map((row, i) =>
        b[0].map((_, j) =>
          row.reduce((sum, elm, k) => sum + elm * b[k][j], 0)
        )
      );
    }
  
    vectorAdd(a, b) {
      return a.map((v, i) => v + b[i]);
    }
  
    step(x) {
      // Concatenate input and previous hidden state
      const xh = [x, ...this.h];
  
      // Forget gate
      const f = this.matrixMultiply([xh], this.Wf)[0];
      const f_bias = this.vectorAdd(f, this.bf);
      const f_activation = f_bias.map(this.sigmoid);
  
      // Input gate
      const i = this.matrixMultiply([xh], this.Wi)[0];
      const i_bias = this.vectorAdd(i, this.bi);
      const i_activation = i_bias.map(this.sigmoid);
  
      // Candidate values
      const c_tilde = this.matrixMultiply([xh], this.Wc)[0];
      const c_tilde_bias = this.vectorAdd(c_tilde, this.bc);
      const c_tilde_activation = c_tilde_bias.map(this.tanh);
  
      // Cell state update
      this.c = this.c.map((c_t, j) => 
        f_activation[j] * c_t + i_activation[j] * c_tilde_activation[j]
      );
  
      // Output gate
      const o = this.matrixMultiply([xh], this.Wo)[0];
      const o_bias = this.vectorAdd(o, this.bo);
      const o_activation = o_bias.map(this.sigmoid);
  
      // Hidden state update
      this.h = o_activation.map((o_t, j) => o_t * this.tanh(this.c[j]));
  
      return this.h[0]; // Return the first element of the hidden state as output
    }
  }
  
  // Example usage
  const lstm = new LSTM();
  const inputSeries = [0.5, 0.8, 0.2, 0.9, 0.1];
  const outputSeries = [];
  
  for (let input of inputSeries) {
    outputSeries.push(lstm.step(input));
  }
  
  console.log("Input series:", inputSeries);
  console.log("Output series:", outputSeries);

/** LSTM usage example */

const lstm = new LSTM();
const inputSeries = [0.5, 0.8, 0.2, 0.9, 0.1];

for (let i = 0; i < inputSeries.length; i++) {
  const input = inputSeries[i];
  const output = lstm.step(input);
  console.log(`Step ${i + 1}:`);
  console.log(`  Input: ${input}`);
  console.log(`  Output: ${output}`);
  console.log(`  Hidden State: ${lstm.h}`);
  console.log(`  Cell State: ${lstm.c}`);
  console.log('---');
}