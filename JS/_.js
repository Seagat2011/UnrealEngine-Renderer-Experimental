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

Depending on the current viewport perpective, 
the animation consists of 1000 frame 3D flipbook, 
with each frame rendered, intended for rasterization 
display in pixelArrayZ!



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

The initial strategy is to employ a combination 
Convolutional Neural Network (CNN)-
and Long Short-Term Memory (LSTM)- model to capture 
spatial- and temporal- data, respectively, 
from the game environment and player input-controller.

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

```mermaid
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
```

# LSTM Architecture Explanation

An LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) 
architecture designed to handle long-term dependencies or temporal dependancies
in sequential data.

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

/** 

# A Convolutional Neural Netwrork (CNNs) Architecture (as markdown)

```mermaid
graph LR
    A[Input Image] --> B[Conv Layer 1]
    B --> C[ReLU]
    C --> D[Max Pooling]
    D --> E[Conv Layer 2]
    E --> F[ReLU]
    F --> G[Max Pooling]
    G --> H[Flatten]
    H --> I[Fully Connected]
    I --> J[Softmax]
    J --> K[Output]
```

Convolutional Neural Netwrorks (CNNs) architectures learn 
spatially-based hierarchical features. The early layers 
detect simple features like edges and corners, while deeper 
layers combine these to recognize more complex patterns 
and eventually entire objects.

Here's a breakdown of its key components and operations:

1. Input Image: 
   The process starts with an input image, typically represented 
   as a 3D tensor (height x width x channels).

2. Convolutional Layer 1:
   This layer applies a set of learnable filters to the input image. 
   Each filter slides across the image, performing element-wise multiplication 
   and summing the results to produce a feature map.

   ```
   ┌─────┐
   │     │
   │  *  │  ->  Feature Map
   │     │
   └─────┘
    Filter
   ```

3. ReLU (Rectified Linear Unit):
   This activation function introduces non-linearity to the network. 
   It replaces all negative values in the feature maps with zero.

   ```
   f(x) = max(0, x)
   ```

4. Max Pooling:
   This layer reduces the spatial dimensions of the feature maps. 
   It divides the input into rectangular pooling regions and outputs 
   the maximum value for each region.

   ```
   2 | 4 | 3    
   ───┼───┼───  ->  4
   1 | 3 | 2    
   ```

5. Convolutional Layer 2:
   Similar to the first convolutional layer, 
   but it operates on the feature maps produced 
   by the previous layers. This allows the network 
   to learn more complex features.

6. ReLU and Max Pooling:
   These layers perform the same operations as described earlier.

7. Flatten:
   This layer reshapes the 3D output from the convolutional 
   and pooling layers into a 1D vector for input into 
   the fully connected layer.

   ```
   [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  ->  [1, 2, 3, 4, 5, 6, 7, 8]
   ```

8. Fully Connected Layer:
   This layer connects every neuron from the flattened input 
   to every neuron in the output. It combines the features 
   learned by the previous layers to make the final prediction.

9. Softmax:
   This final activation function converts the output of the 
   fully connected layer into a probability distribution over 
   the possible classes.

   ```
   softmax(x_i) = exp(x_i) / Σ exp(x_j)
   ```

10. Output:
    The final output is a vector of probabilities, 
    where each element represents the likelihood 
    of the input image belonging to a particular 
    class.

This CNN architecture learns hierarchical features. 
The early layers detect simple features like edges 
and corners, while deeper layers combine these 
to recognize more complex patterns and 
eventually entire objects.

*/

class CNN {
    constructor() {
      this.conv1 = this.createConvLayer(1, 8, 3); // 1 input channel, 8 filters, 3x3 kernel
      this.conv2 = this.createConvLayer(8, 16, 3); // 8 input channels, 16 filters, 3x3 kernel
      this.fc = this.createFCLayer(16 * 5 * 5, 10); // Assuming input size is 28x28, after two 2x2 max poolings: 16 * (28/2/2) * (28/2/2) = 16 * 5 * 5
    }
  
    createConvLayer(inChannels, outChannels, kernelSize) {
      return {
        weights: this.randomArray([outChannels, inChannels, kernelSize, kernelSize]),
        bias: this.randomArray([outChannels])
      };
    }
  
    createFCLayer(inFeatures, outFeatures) {
      return {
        weights: this.randomArray([outFeatures, inFeatures]),
        bias: this.randomArray([outFeatures])
      };
    }
  
    randomArray(shape) {
      const size = shape.reduce((a, b) => a * b);
      return Array.from({ length: size }, () => Math.random() - 0.5);
    }
  
    relu(x) {
      return Math.max(0, x);
    }
  
    conv2d(input, layer) {
      // Simplified 2D convolution
      const [outChannels, inChannels, kernelSize] = layer.weights.length;
      const output = Array(outChannels).fill().map(() => Array(input[0].length - kernelSize + 1).fill().map(() => Array(input[0][0].length - kernelSize + 1).fill(0)));
  
      for (let oc = 0; oc < outChannels; oc++) {
        for (let i = 0; i < output[0].length; i++) {
          for (let j = 0; j < output[0][0].length; j++) {
            let sum = 0;
            for (let ic = 0; ic < inChannels; ic++) {
              for (let ki = 0; ki < kernelSize; ki++) {
                for (let kj = 0; kj < kernelSize; kj++) {
                  sum += input[ic][i + ki][j + kj] * layer.weights[oc][ic][ki][kj];
                }
              }
            }
            output[oc][i][j] = this.relu(sum + layer.bias[oc]);
          }
        }
      }
      return output;
    }
  
    maxPool2d(input, poolSize = 2) {
      const output = Array(input.length).fill().map(() => 
        Array(Math.floor(input[0].length / poolSize)).fill().map(() => 
          Array(Math.floor(input[0][0].length / poolSize)).fill(0)
        )
      );
  
      for (let c = 0; c < input.length; c++) {
        for (let i = 0; i < output[0].length; i++) {
          for (let j = 0; j < output[0][0].length; j++) {
            let max = -Infinity;
            for (let pi = 0; pi < poolSize; pi++) {
              for (let pj = 0; pj < poolSize; pj++) {
                max = Math.max(max, input[c][i * poolSize + pi][j * poolSize + pj]);
              }
            }
            output[c][i][j] = max;
          }
        }
      }
      return output;
    }
  
    flatten(input) {
      return input.flat(2);
    }
  
    fullyConnected(input, layer) {
      return layer.weights.map((weights, i) => 
        this.relu(weights.reduce((sum, weight, j) => sum + weight * input[j], 0) + layer.bias[i])
      );
    }
  
    softmax(input) {
      const expValues = input.map(Math.exp);
      const sumExp = expValues.reduce((a, b) => a + b, 0);
      return expValues.map(exp => exp / sumExp);
    }
  
    forward(input) {
      let x = input;
      x = this.conv2d(x, this.conv1);
      x = this.maxPool2d(x);
      x = this.conv2d(x, this.conv2);
      x = this.maxPool2d(x);
      x = this.flatten(x);
      x = this.fullyConnected(x, this.fc);
      x = this.softmax(x);
      return x;
    }
  }
  
  // Usage example
  const cnn = new CNN();
  
  // Create a sample 28x28 grayscale image (1 channel)
  const sampleImage = Array(1).fill().map(() => 
    Array(28).fill().map(() => 
      Array(28).fill().map(() => Math.random())
    )
  );
  
  const output = cnn.forward(sampleImage);
  console.log("Output probabilities:", output);