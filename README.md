# NGFuzz: Neural-Guided Fuzzing Framework

NGFuzz is an advanced fuzzing framework that combines traditional fuzzing techniques with neural network guidance to improve the efficiency and effectiveness of vulnerability discovery in software.

## Project Overview

NGFuzz builds upon the American Fuzzy Lop (AFL) fuzzer by incorporating neural network models to guide the fuzzing process. The framework uses machine learning to predict which inputs are more likely to trigger new code paths or uncover vulnerabilities, thereby optimizing the fuzzing process.

### Key Features

- Neural network-guided fuzzing for improved code coverage
- Gradient-based mutation strategies
- Adversarial example generation for discovering edge cases
- Real-time model retraining based on fuzzing results
- Compatible with AFL's instrumentation techniques

## Repository Structure

- **ngfuzz.c** - The core fuzzer implementation, extending AFL with neural guidance
- **ngfuzz.py** - Python implementation of the neural guidance module
- **ngfuzz_impGrad.py** - Implementation using importance-weighted gradients for mutation guidance
- **ngfuzz_py.ipynb** - Jupyter notebook for experimentation and visualization
- **AFL/** - Contains the original AFL fuzzer codebase and supporting files that NGFuzz builds upon

## Technical Approach

NGFuzz works by:

1. **Input Vectorization**: Converting binary inputs into numerical vectors suitable for neural network processing
2. **Path Analysis**: Analyzing execution paths to identify interesting code regions
3. **Neural Guidance**: Using neural networks to predict which mutations are most likely to discover new paths
4. **Gradient-Based Mutation**: Applying targeted mutations based on gradient information from the neural model
5. **Adversarial Attack Generation**: Leveraging adversarial machine learning techniques to craft inputs that target under-explored code paths:
   - Identifies under-explored paths through execution frequency analysis
   - Uses a specialized neural network to generate adversarial examples
   - Applies gradient descent optimization to find minimal but effective input perturbations
   - Employs L1/L2 regularization to balance exploration vs. exploitation
   - Dynamically adjusts targeting strategy based on fuzzing progress
6. **Continuous Learning**: Retraining the model as new inputs and execution paths are discovered

## Getting Started

### Prerequisites

- GCC compiler
- Python 3.6+
- TensorFlow/Keras
- Required Python packages: numpy, pandas, matplotlib, scipy, innvestigate

### Building the Fuzzer

```bash
# Compile the AFL core components
cd AFL
make clean all

# Return to main directory and compile NGFuzz
cd ..
gcc -o ngfuzz ngfuzz.c -lm -lpython3
```

### Running NGFuzz

```bash
# Basic usage
./ngfuzz -i input_dir -o output_dir /path/to/target_binary

# With neural guidance enabled
python ngfuzz_impGrad.py target_name output_folder &
./ngfuzz -i input_dir -o output_dir -N /path/to/target_binary
```

## Advanced Usage

NGFuzz supports various modes of operation:

- **Standard Fuzzing**: Similar to AFL but with neural network prioritization
- **Adversarial Mode**: Generates adversarial examples to target specific code paths
- **Importance-Weighted Gradients**: Focuses mutations on bytes with higher gradient values
- **Continuous Learning**: Automatically retrains the model based on new findings

### Adversarial Attack Component

One of the most innovative aspects of NGFuzz is its adversarial attack component, which leverages techniques from adversarial machine learning to discover new execution paths:

1. **Target Identification**: The system identifies under-explored code paths by analyzing execution frequency data
2. **Adversarial Example Generation**: Using a specialized neural network architecture, NGFuzz generates inputs that are likely to trigger these under-explored paths
3. **Gradient-Based Optimization**: The adversarial model uses gradient descent to find minimal perturbations that can change program behavior
4. **Regularization Controls**: L1/L2 regularization parameters allow fine-tuning the balance between exploration and exploitation
5. **Adaptive Targeting**: The system dynamically adjusts its targeting strategy based on fuzzing progress

This approach allows NGFuzz to intelligently craft inputs that explore hard-to-reach code regions, significantly improving upon traditional random mutation strategies. The adversarial component essentially "learns" how to break the target program by understanding which input modifications are most likely to trigger new behaviors.

## Acknowledgments

This project builds upon the American Fuzzy Lop (AFL) fuzzer by Michal Zalewski. The neural guidance components are implemented using TensorFlow/Keras and draw inspiration from various research papers in the field of neural-guided fuzzing.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file in the AFL directory for details.
