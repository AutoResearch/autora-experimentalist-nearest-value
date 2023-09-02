# AutoRA Nearest Value Experimentalist

A experimentalist which returns the nearest values between the input samples and the allowed values, without replacement.

# Example Code

```
from autora.experimentalist.nearest_value import nearest_values_sampler
import numpy as np

#Meta-Setup
X_allowed = np.linspace(-3, 6, 10)
X = np.random.choice(X_allowed,10)
n = 5

#Sampler
X_new = nearest_values_sampler(X, X_allowed, n)
```
