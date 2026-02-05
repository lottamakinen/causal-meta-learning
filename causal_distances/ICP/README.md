# ICP method

## R setup: 

```
install.packages("glmnet", dependencies = TRUE)
install.packages("mboost", dependencies = TRUE)
install.packages("Matrix", dependencies = TRUE)
install.packages("parallel", dependencies = TRUE)
install.packages("stabs", dependencies = TRUE)
install.packages("InvariantCausalPrediction", dependencies = TRUE)
install.packages("tidyverse", dependencies = TRUE)
install.packages("optparse", dependencies = TRUE)
```

## Usage: 

1. The folder includes two different R scripts for ICP and hiddenICP method. ICP version is more reliable, and should be used instead of hiddenIPC. 

2. Make sure the global parameters at the top of the file are updated for your data. 

3. Run the R script by running the sbatch with a command `sbatch causal_distances/ICP/run_icp.sbatch` or directly in the terminal with the command `Rscript causal_distances/ICP/ICP.R.

## Notes on usage:

- The code uses a random allocation of experimental indices. Various approaches to select the experimental indices are possible.
- The time complexity of ICP method is exponential and it is computationally very time demanding when task number p > 10. To save computational resources with high task numbers you can use preselection by choosing the parameter `selection`. You can use different selection methods. Can use "all" for no pre-selection (which guarantees coverage but might take longer to compute), "boosting" for a boosting-type, "lasso" for Lasso cross-validated or "stability" for a stability-selection-type pre-selection. Default is "all" if p does not exceed 10 and "boosting" otherwise.
- You can select the number of tasks the preselection method outputs with the ICP parameter `maxNoVariables`. Also the maximum number of simultaneous variables in the experimental settings can be selected with the parameter `maxNoVariablesSimult`. Higher values may lead to better coverage, but make the method more computationally complex. For reference, running the method with `maxNoVariables = 16` and `maxNoVariablesSimult = 5` took us ~2h. 
