# Funk SVD
An attempt at implementing Funk SVD in Python. Using the MovieLens 100K dataset, a RMSE of 0.95 is found with a latent
dimension size of 30. The optimization process is vanilla SGD. Only library used is numpy.

## Train and test RMSE
Train and test RMSE were compared with different latent dimensions sizes.
Increased latent dimension sizes cause lower training RMSE with little benefit to the test RMSE. Appears indicative that
the model can easily overfit with too many dimensions.

![Test/train RMSE](test-vs-train.png "Test/train RMSE")
