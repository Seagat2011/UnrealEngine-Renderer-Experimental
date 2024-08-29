

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
