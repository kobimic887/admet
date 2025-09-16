import warnings
import logging
import sys
import os
from contextlib import redirect_stderr

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Suppress import messages
with open(os.devnull, 'w') as devnull:
    with redirect_stderr(devnull):
        from admet_ai import ADMETModel
        model = ADMETModel()

# Now only your output will show
smiles = "O(c1ccc(cc1)CCOC)CC(O)CNC(C)C"
predictions = model.predict(smiles)
print("Predictions:")
print(predictions)