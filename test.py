import torch
from neural_ssm import DeepSSM
import matplotlib.pyplot as plt

ssm = DeepSSM(d_input=1, d_state= 6, d_output=1, param='l2n', ff='LGLU', rho=.99,
              max_phase_b = None, rmin = .95,
        rmax = .99,
        gamma = 100,
              random_phase=True)

u = torch.zeros(1,300,1)
u[:,24,:] = 100

y,_ = ssm(u)

plt.figure()
plt.plot(y[0,:,0].detach().numpy())
plt.show()

print(y[0,-1,0])



