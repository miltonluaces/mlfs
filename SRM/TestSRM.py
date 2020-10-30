import numpy as np
from Srm import SRM
    
srm_model = SRM(neurons=3, threshold=1, t_current=0.3, t_membrane=20, eta_reset=5, verbose=True)

models = [srm_model]

for model in models:
    print("-"*10)
    if isinstance(model, SRM):
        print('Demonstration of the SRM Model')

    s = np.array([[0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    w = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    neurons, timesteps = s.shape

    for t in range(timesteps):
        total_current = model.check_spikes(s, w, t)
        print("Spiketrain:")
        print(s)

