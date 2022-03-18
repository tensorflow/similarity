# Similarity Metric Benchmarking System

## Current results
Apologies for the formatting - A more detailed table form of this json will be created once
more loss functions have been tested, but for now here are the raw json values.

One thing to note - There may be an error in the model config(not sure about loss) because the accuracy
and recall values are horrible. 

```json
{
    "loss": "circle_loss",
    "dataset_name": "cars196",
    "seen_queries": {
        "calibration": {
            "method": "optimal",
            "value": "0.019999999552965164",
            "distance": "0.8320743441581726",
            "precision": "0.010204081423580647",
            "recall": "1.0",
            "binary_accuracy": "0.010204081423580647",
            "f1": "0.020202020183205605"
        },
        "paper": {
            "R@1": "0.010204081",
            "R@2": "0.02244898",
            "R@4": "0.037755102",
            "R@8": "0.059183672"
        }
    },
    "unseen_queries": {
        "calibration": {
            "method": "optimal",
            "value": "0.019999999552965164",
            "distance": "0.8320743441581726",
            "precision": "0.010204081423580647",
            "recall": "1.0",
            "binary_accuracy": "0.010204081423580647",
            "f1": "0.020202020183205605"
        },
        "paper": {
            "R@1": "0.008163265",
            "R@2": "0.013265306",
            "R@4": "0.025510205",
            "R@8": "0.047959182"
        }
    }
}
```

```json
{
    "loss": "multi_similarity",
    "dataset_name": "cars196",
    "seen_queries": {
        "calibration": {
            "method": "optimal",
            "value": "0.009999999776482582",
            "distance": "0.9707562923431396",
            "precision": "0.006122448947280645",
            "recall": "1.0",
            "binary_accuracy": "0.006122448947280645",
            "f1": "0.012170384638011456"
        },
        "paper": {
            "R@1": "0.006122449",
            "R@2": "0.01122449",
            "R@4": "0.03265306",
            "R@8": "0.06122449"
        }
    },
    "unseen_queries": {
        "calibration": {
            "method": "optimal",
            "value": "0.009999999776482582",
            "distance": "0.9707562923431396",
            "precision": "0.006122448947280645",
            "recall": "1.0",
            "binary_accuracy": "0.006122448947280645",
            "f1": "0.012170384638011456"
        },
        "paper": {
            "R@1": "0.010204081",
            "R@2": "0.01632653",
            "R@4": "0.023469388",
            "R@8": "0.05612245"
        }
    }
}
```

```json
{
    "loss": "pn_loss_semi",
    "dataset_name": "cars196",
    "seen_queries": {
        "calibration": {
            "method": "optimal",
            "value": "0.019999999552965164",
            "distance": "0.7256540060043335",
            "precision": "0.009183673188090324",
            "recall": "1.0",
            "binary_accuracy": "0.009183673188090324",
            "f1": "0.01820020191371441"
        },
        "paper": {
            "R@1": "0.009183673",
            "R@2": "0.015306123",
            "R@4": "0.02755102",
            "R@8": "0.06122449"
        }
    },
    "unseen_queries": {
        "calibration": {
            "method": "optimal",
            "value": "0.019999999552965164",
            "distance": "0.7256540060043335",
            "precision": "0.009183673188090324",
            "recall": "1.0",
            "binary_accuracy": "0.009183673188090324",
            "f1": "0.01820020191371441"
        },
        "paper": {
            "R@1": "0.0051020407",
            "R@2": "0.01122449",
            "R@4": "0.024489796",
            "R@8": "0.06122449"
        }
    }
}

```

## Running the benchmark

1. Generate datasets: `python generate_datasets.py -c config/benchmark_prod.json`
datasets should be created under `datasets/__version__/__dataset_name__/`

2. Train models: `python train.py -c config/benchmark_prod.json`

3. Evaluate performance: `python evaluate.py -c config/benchmark_prod.json`

4. Analyze results using notebook: `JSON files containing the info can each be found at models/__version__/__dataset_name__ _ __loss__/`
