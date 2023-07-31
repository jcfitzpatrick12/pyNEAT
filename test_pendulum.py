from one_pendulum_model.test import pendulum_testing
from plotting.plot_neat import plotNEAT

#run the pendulum test

pendulum_testing().run_small_angle_test()
#pendulum_testing().varying_force_constant_test()
#load the most recent data!
plotNEAT().plot_pendulum_run()
