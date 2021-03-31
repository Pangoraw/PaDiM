import sys
import pickle

sys.path.append('./')
sys.path.append('./deep_svdd/src/')

from padim import PaDiM, PaDiMShared

PARAMS_PATH = sys.argv[1]

print(f"Converting {PARAMS_PATH}")

with open(PARAMS_PATH, 'rb') as f:
    params = pickle.load(f)

if 'shared' in PARAMS_PATH:
    padim = PaDiMShared.from_residuals(*params, device="cpu")

    padim.cov = padim.cov.permute(1, 0)
else:
    padim = PaDiM.from_residuals(*params, device="cpu")

    padim.covs = padim.covs.permute(2, 0, 1)
    padim.means = padim.means.permute(1, 0)


params = padim.get_residuals()
with open(PARAMS_PATH, 'wb') as f:
    pickle.dump(params, f)
