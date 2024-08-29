

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
