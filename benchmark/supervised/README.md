# Similarity Metric Benchmarking System

## Current results

## Running the benchmark

1. Generate initial training data with preprocessing: `python create_datasets.py -c config/benchmark_prod.json -f "cars.*lamb.*"`

2. Train and Hypertune the model parameters: `python hyper_parameter_search.py -c config/benchmark_prod.json -f "cars.*lamb.*"`

2. Evaluate performance:

3. Analyze results using notebook:
