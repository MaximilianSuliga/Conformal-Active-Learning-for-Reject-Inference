# Conformal-Active-Learning-for-Reject-Inference
Master thesis of the full title "An Active Learning Approach for Reject Inference in Credit Scoring using Conformal Prediction Intervals on Real and Semi-Artificial Data".
Full text available as PDF.

**Author:** Maximilian Suliga, business administration student at Humboldt University of Berlin

**First Examinar:** Prof. Dr. Stefan Lessmann

**Second Examiner:** Prof. Dr. Benjamin Fabian

## Table of Content

- [Abstract](#Abstract)
- [Main Results](#Main-Results)
- [Dependencies](#Dependencies)
- [Reproducing results](#Reproducing-Results)
- [References](#References)


## Abstract
This master thesis investigates the suitability of Active Learning strategies based on conformal predictors for Reject Inference, with a focus on their performance, factors influencing their effectiveness, and recommendations for implementation in the industry. The study explores the role of nonconformity functions and their cost-sensitive recalibration in shaping the performance of conformal predictors. It is observed that a nonconformity function based on probabilistic error, while cost-insensitive, tends to select cheaper instances but may not always prioritize the most informative ones. In contrast, a cost-sensitive nonconformity function allows for the selection of both cost-efficient and informative applications, particularly in environments where opportunity cost is lower than the cost of default, and the number of repaying customers surpasses defaulting ones. The impact of class imbalance on the economic feasibility of AL strategies in Credit Scoring is identified as one of the significant factors. Active Learning demonstrates the highest relevance and superiority when applied to imbalanced datasets, where labeling the minor class incurs costs rather than the major class. This condition is well-met in conventional lending scenarios, while it does not hold in Peer-to-Peer lending with non-erroneous rejection assumptions. Based on these findings, various application areas are suggested for Active Learning in Credit Scoring.

## Main Results

![results](/result.png)

## Dependencies
* Python 3.8
*  Use of different Pandas versions: while the Data Preparation requires Pandas 1.5.0 or higher, the Main Experiment requires Pandas 1.3.4 or lower. Two different YAML files are provided, but they differ only in their Pandas version
*  Apart from standard Python libraries, this repository makes use of the nonconformist library (Linusson, 2017)

## Reproducing Results
pickle files, notebooks, prepared data, randomness in MC
* Folders contain python files, notebooks, pickle files and prepared data used in the main experiment
* Pickle Files generated in a Notebook are available in the folder of the respective notebook
* Since Monte Carlo simulations do not allow for a random seed, the prepared, partly simulated data is uploaded as well. Rerunning the Monte Carlo simulation is expected to yield at least slightly different simulations due to the underlying randomness.

## References
Linusson, H. (2017). An introduction to conformal prediction. 6th Symp. Conformal and Probabilistic Prediction Appl. Repository available at: https://github.com/donlnz/nonconformist
