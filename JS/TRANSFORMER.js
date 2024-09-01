/**

# Transformer Architecture (as markdown)

Brief

```mermaid
graph TD
    Input[Input] --> Embedding[Embedding Layer]
    Embedding --> PositionalEncoding[Positional Encoding]
    PositionalEncoding --> EncoderStack[Encoder Stack]
    EncoderStack --> |N x| MultiHeadAttention1[Multi-Head Attention]
    MultiHeadAttention1 --> AddNorm1[Add & Norm]
    AddNorm1 --> FeedForward1[Feed Forward]
    FeedForward1 --> AddNorm2[Add & Norm]
    AddNorm2 --> Output1[Encoder Output]
    
    Output1 --> DecoderStack[Decoder Stack]
    DecoderStack --> |N x| MultiHeadAttention2[Masked Multi-Head Attention]
    MultiHeadAttention2 --> AddNorm3[Add & Norm]
    AddNorm3 --> MultiHeadAttention3[Multi-Head Attention]
    MultiHeadAttention3 --> AddNorm4[Add & Norm]
    AddNorm4 --> FeedForward2[Feed Forward]
    FeedForward2 --> AddNorm5[Add & Norm]
    AddNorm5 --> LinearLayer[Linear Layer]
    LinearLayer --> Softmax[Softmax]
    Softmax --> Output[Output]
```

Detailed

```mermaid
graph TD
    Input[Input] --> |Split| Q[Query]
    Input --> |Split| K[Key]
    Input --> |Split| V[Value]
    Q --> |Linear| Q1[Q1]
    Q --> |Linear| Q2[Q2]
    Q --> |Linear| Q3[Q3]
    K --> |Linear| K1[K1]
    K --> |Linear| K2[K2]
    K --> |Linear| K3[K3]
    V --> |Linear| V1[V1]
    V --> |Linear| V2[V2]
    V --> |Linear| V3[V3]
    Q1 & K1 --> |MatMul, Scale| A1[Attention Scores 1]
    Q2 & K2 --> |MatMul, Scale| A2[Attention Scores 2]
    Q3 & K3 --> |MatMul, Scale| A3[Attention Scores 3]
    A1 --> |Softmax| W1[Attention Weights 1]
    A2 --> |Softmax| W2[Attention Weights 2]
    A3 --> |Softmax| W3[Attention Weights 3]
    W1 & V1 --> |MatMul| H1[Head 1]
    W2 & V2 --> |MatMul| H2[Head 2]
    W3 & V3 --> |MatMul| H3[Head 3]
    H1 & H2 & H3 --> |Concatenate| C[Concatenated Heads]
    C --> |Linear| O[Output]
```

# A Transformer Architecture Explanation

```mermaid
graph TD
    Input[Input] --> Embedding[Embedding Layer]
    Embedding --> PositionalEncoding[Positional Encoding]
    PositionalEncoding --> EncoderStack[Encoder Stack]
    EncoderStack --> |N x| MultiHeadAttention1[Multi-Head Attention]
    MultiHeadAttention1 --> AddNorm1[Add & Norm]
    AddNorm1 --> FeedForward1[Feed Forward]
    FeedForward1 --> AddNorm2[Add & Norm]
    AddNorm2 --> Output1[Encoder Output]
    
    Output1 --> DecoderStack[Decoder Stack]
    DecoderStack --> |N x| MultiHeadAttention2[Masked Multi-Head Attention]
    MultiHeadAttention2 --> AddNorm3[Add & Norm]
    AddNorm3 --> MultiHeadAttention3[Multi-Head Attention]
    MultiHeadAttention3 --> AddNorm4[Add & Norm]
    AddNorm4 --> FeedForward2[Feed Forward]
    FeedForward2 --> AddNorm5[Add & Norm]
    AddNorm5 --> LinearLayer[Linear Layer]
    LinearLayer --> Softmax[Softmax]
    Softmax --> Output[Output]

```

Now, let me explain the operation of this Transformer architecture:

1. Input and Embedding Layer:
   - The input sequence (e.g., words or tokens) is first passed through an embedding layer.
   - This layer converts each token into a dense vector representation.

2. Positional Encoding:
   - Since Transformers don't have a natural way to process sequential information, positional encodings are added to the embedded inputs.
   - These encodings provide information about the position of each token in the sequence.

3. Encoder Stack:
   - The encoder consists of multiple identical layers stacked on top of each other (usually 6 layers).
   - Each layer has two sub-layers:
     a. Multi-Head Attention:
        - This mechanism allows the model to focus on different parts of the input sequence when encoding each token.
        - "Multi-head" means this attention is performed multiple times in parallel, allowing the model to capture different types of relationships.
     b. Feed Forward Neural Network:
        - A simple fully connected neural network applied to each position separately and identically.
   - Each sub-layer is followed by Add & Norm operations:
     - Add: A residual connection that adds the input to the sub-layer output.
     - Norm: Layer normalization to stabilize the activations.

4. Decoder Stack:
   - Similar to the encoder, the decoder also consists of multiple identical layers.
   - Each layer has three sub-layers:
     a. Masked Multi-Head Attention:
        - This attention mechanism prevents the decoder from looking at future tokens during training.
     b. Multi-Head Attention:
        - This layer performs attention over the encoder's output.
     c. Feed Forward Neural Network:
        - Similar to the encoder's feed-forward layer.
   - Each sub-layer is also followed by Add & Norm operations.

5. Linear Layer and Softmax:
   - The output of the decoder stack is passed through a linear layer to project it to the vocabulary size.
   - A softmax function is applied to convert the output into probabilities for each token in the vocabulary.

Here's a more detailed explanation of the key components:

1. Multi-Head Attention:

```mermaid
graph TD
    Input[Input] --> |Split| Q[Query]
    Input --> |Split| K[Key]
    Input --> |Split| V[Value]
    Q --> |Linear| Q1[Q1]
    Q --> |Linear| Q2[Q2]
    Q --> |Linear| Q3[Q3]
    K --> |Linear| K1[K1]
    K --> |Linear| K2[K2]
    K --> |Linear| K3[K3]
    V --> |Linear| V1[V1]
    V --> |Linear| V2[V2]
    V --> |Linear| V3[V3]
    Q1 & K1 --> |MatMul, Scale| A1[Attention Scores 1]
    Q2 & K2 --> |MatMul, Scale| A2[Attention Scores 2]
    Q3 & K3 --> |MatMul, Scale| A3[Attention Scores 3]
    A1 --> |Softmax| W1[Attention Weights 1]
    A2 --> |Softmax| W2[Attention Weights 2]
    A3 --> |Softmax| W3[Attention Weights 3]
    W1 & V1 --> |MatMul| H1[Head 1]
    W2 & V2 --> |MatMul| H2[Head 2]
    W3 & V3 --> |MatMul| H3[Head 3]
    H1 & H2 & H3 --> |Concatenate| C[Concatenated Heads]
    C --> |Linear| O[Output]

```

   - The input is linearly projected into Query (Q), Key (K), and Value (V) representations.
   - For each attention head:
     a. Calculate attention scores by multiplying Q and K.
     b. Scale the scores and apply softmax to get attention weights.
     c. Multiply attention weights with V to get the output.
   - Concatenate the outputs from all heads and apply a final linear projection.

2. Positional Encoding:
   - Adds information about the position of each token in the sequence.
   - Usually implemented using sine and cosine functions of different frequencies.

3. Masked Multi-Head Attention:
   - Similar to regular Multi-Head Attention, but with a mask applied to the attention scores.
   - The mask prevents the decoder from attending to future positions during training.

4. Feed Forward Networks:
   - Typically consists of two linear transformations with a ReLU activation in between.
   - Applied to each position separately and identically.

The Transformer architecture has several advantages:
1. Parallelization: Unlike RNNs, Transformers can process all input tokens simultaneously.
2. Long-range dependencies: The attention mechanism allows the model to directly connect distant positions.
3. Flexibility: The same architecture can be used for various tasks like translation, summarization, and question-answering.

*/

// Simplified Transformer implementation in JavaScript

class Transformer {
  constructor({ inputSize, outputSize, numLayers = 2, numHeads = 4, hiddenSize = 64 } = {}) {
      let self = this;
      this.inputSize = inputSize;
      this.outputSize = outputSize;
      this.numLayers = numLayers;
      this.numHeads = numHeads;
      this.hiddenSize = hiddenSize;        
      // Initialize weights (simplified)
      this.weights = {
          embedding: this.randomMatrix(inputSize, hiddenSize),
          attention: {
            query: this.randomMatrix(hiddenSize, hiddenSize),
            key: this.randomMatrix(hiddenSize, hiddenSize),
            value: this.randomMatrix(hiddenSize, hiddenSize)
          },
          ffn: {
            w1: this.randomMatrix(hiddenSize, hiddenSize),
            w2: this.randomMatrix(hiddenSize, hiddenSize)
          },
          output: this.randomMatrix(hiddenSize, outputSize)
      };
  }

  randomMatrix(rows, cols) {
    const ret = Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => Math.random() - 0.5));
    return ret;
  }

  matrixMultiply(a, b) {
    const result = Array(a.length).fill().map(() => Array(b[0].length).fill(0));
    return result.map((row, i) => {
      return row.map((_, j) => {
        const ret = a[i].reduce((sum, elm, k) => sum + (elm * b[k][j]), 0);
        return ret;
      });
    });
  }

  softmax(arr) {
    const expValues = arr.map(Math.exp);
    const sumExpValues = expValues.reduce((a, b) => a + b);
    const ret = expValues.map(v => v / sumExpValues);
    return ret;
  }

  attention(query, key, value) {
    const scores = this.matrixMultiply(query, this.transpose(key));
    const scaledScores = scores.map(row => row.map(s => s / Math.sqrt(this.hiddenSize)));
    const weights = scaledScores.map(this.softmax);
    const ret = this.matrixMultiply(weights, value);
    return ret;
  }
  
  feedForward(x) {
    const hidden = this.matrixMultiply(x, this.weights.ffn.w1).map(row => 
      row.map(x => Math.max(0, x))  // ReLU activation
    );
    const ret = this.matrixMultiply(hidden, this.weights.ffn.w2);
    return ret;
  }

  transpose(matrix) {
    const ret = matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
    return ret;
  }

  forward(input) {
    // Embedding
    let output = this.matrixMultiply(input, this.weights.embedding);

    // Transformer layers
    for (let i = 0; i < this.numLayers; i++) {
      // Multi-head attention (simplified to single-head)
      const query = this.matrixMultiply(output, this.weights.attention.query);
      const key = this.matrixMultiply(output, this.weights.attention.key);
      const value = this.matrixMultiply(output, this.weights.attention.value);
      const attentionOutput = this.attention(query, key, value);

      // Add & Norm (simplified)
      output = output.map((row, i) => row.map((val, j) => val + attentionOutput[i][j]));

      // Feed-forward
      const ffnOutput = this.feedForward(output);

      // Add & Norm (simplified)
      output = output.map((row, i) => row.map((val, j) => val + ffnOutput[i][j]));
    }

    // Output layer
    const ret = this.matrixMultiply(output, this.weights.output);
    return ret;
  }
}

// Usage example
const inputSize = 10;
const outputSize = 5;
const transformer = 
  new Transformer({
      inputSize: inputSize, 
      outputSize: outputSize});

// Example input (batch size of 1, sequence length of 3)
const input = [
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
];

const output = transformer.forward(input);
console.log("Input:", input);
console.log("Output:", output);

/**

# Usage example explanation:

1. The `Transformer` class is defined with a constructor 
that initializes the model's parameters.

2. Helper functions like `randomMatrix`, `matrixMultiply`, 
`softmax`, and `transpose` are implemented to handle basic 
matrix operations.

3. The `attention` function implements the core attention mechanism.

4. The `feedForward` function represents the feed-forward 
neural network in each Transformer layer.

5. The main `forward` function implements the full 
Transformer forward pass:
 - It starts with an embedding layer.
 - It then applies multiple Transformer layers, each consisting of:
   - Multi-head attention (simplified to single-head in this implementation)
   - Add & Norm (simplified)
   - Feed-forward network
   - Another Add & Norm
 - Finally, it applies an output layer.

6. The usage example shows how to create a Transformer 
instance and run a forward pass with sample input.

To step through this code:

1. Start by creating a Transformer instance:
 ```javascript
 const transformer = new Transformer(inputSize, outputSize);
 ```
 This initializes all the weights randomly.

2. Prepare your input. In the example, we have a sequence of 
3 one-hot encoded vectors:
 ```javascript
 const input = [
   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
 ];
 ```

3. Call the `forward` method:
 ```javascript
 const output = transformer.forward(input);
 ```

4. Inside `forward`, you can add console.log statements to see 
the intermediate results:
 - After embedding
 - After each attention operation
 - After each feed-forward operation
 - Final output

Remember, this is a simplified implementation to help understand 
the core concepts. A production-ready Transformer would include 
more optimizations, proper layer normalization, and typically 
be implemented using a deep learning framework for efficiency 
and automatic differentiation.

Would you like me to explain any specific part of the code in more detail?

*/