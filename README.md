# Edge AI Inference Engine

The project tackles the problem of deploying a functioning AI model 
on memory and compute-constrained devices. A neural network was built 
to classify handwritten digits, trained to 92.33% accuracy on 60,000 
MNIST images, then compressed using INT8 quantisation. The quantised 
model is 3.9× smaller with only 0.86% accuracy drop — fitting inside 
a 256KB microcontroller that the original model could not.

## What it implements

- train.py — defines SimpleNet (784→128→10 architecture), trains 
  on 60,000 MNIST images for 5 epochs, evaluates on 10,000 unseen 
  test images, saves weights as 399.9KB FP32 model.

- quantise.py— loads FP32 weights, converts all Linear layer 
  weights from 32-bit float to 8-bit integer using dynamic 
  quantisation, saves 103.6KB INT8 model and measures accuracy drop.

- benchmark.py — loads both models, runs 1,000 timed inference 
  passes on each, reports mean/median/p95/min latency. Demonstrates 
  hardware dependency of INT8 speedup.

## Key results

| Metric | FP32 | INT8 |
|---|---|---|
| Model size | 399.9 KB | 103.6 KB |
| Size reduction | 1× | 3.9× smaller |
| Test accuracy | 92.33% | 91.47% |
| Accuracy drop | — | 0.86% |
| Mean latency (CPU) | 0.161ms | 0.594ms |
| Fits in 256KB MCU? | No | Yes |

Note: INT8 is slower on a general-purpose CPU because FP32 arithmetic 
units are hardware-optimised. INT8 speedup is realised on ARM chips, 
TPUs, and purpose-built edge AI accelerators.

## Research connection

This project connects directly to the work of Prof. Anand Raghunathan 
(Purdue) whose research on energy-efficient ML hardware addresses 
exactly this tradeoff — accuracy vs model size vs inference speed. 
Prof. Miriam Leeser (Northeastern) designs FPGA-based accelerators 
where INT8 quantisation delivers its full speedup potential. The 
benchmark results demonstrate precisely why hardware-aware ML research 
exists — software optimisation alone is insufficient without the right 
silicon underneath.

## How to run

```bash
pip install torch torchvision numpy
git clone https://github.com/ReggieJOE/edge-ai-inference.git
cd edge-ai-inference
python train.py
python quantise.py
python benchmark.py
```

## Author

Reginald Jojo Gwira  
Kwame Nkrumah University of Science and Technology, Ghana  
GitHub: [ReggieJOE](https://github.com/ReggieJOE)