'''
In this script, we convert our original second order ODE describing the 1-pendulum with (horizontally) moving base through

theta'' + (g/l)sin(theta)+cos(theta)*X''/l = 0

where theta=theta(t) is the angle of the pendulum rod with the vertical, and X=X(t) is the base movement function which is prescribed.

We can convert this into a system of first order ODE's and use scipy to numerically evaluate the solutions

omega = theta'
omega' = -[(g/l)sin(theta) + cos(theta)*X''/l]

We implement this below.

rather than create our own numerical solver, just integrate over a short time then evaluate the new theta, theta dot ect. so that the NN can assess the current state!

rough plan:

solve from [t_n-1,t_n] (where t_n-1-t_n is small) save the solution for that time
extract the final theta, theta_dot input into NN, output X_dotdot (or related quantity)
then solve from t_n+1,t_n with the final theta, theta dot as initial conditions and the new X dot dot as the function.
repeat

theta,theta_dot ---> NN black box -----> X dot dot

So let's get some detail on whats needed for the NN

FIX THIS CODE

PLACE ALL CONSTANTS INTO sys_vars rather than packing them into the class
then rejig the code a bit for ease of use.




'''

'''
import packages and classes
'''

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
#from one_pendulum_funcs import one_pendulum_funcs
#from one_pendulum_plot import one_pendulum_plot
from sys_vars import sys_vars

raise SystemExit

##########################################
######## END OF USER INPUTS ##############
##########################################

#call the solution at 101 evenly spaced intervals for 10 seconds
t = np.linspace(0, sys_vars().T, sys_vars().n_t)

#sol holds the vector y(t) i.e. theta and omega at each t
#pend_funcs
sol = odeint(one_pendulum_funcs(sys_vars).pend_system, sys_vars().y0, t)

#defining the plot pendulum class
pend_plots=one_pendulum_plot(one_pendulum_funcs(),sol,t,sys_vars().y0)
#pend_plots.plot_pendulum_snapshots()
pend_plots.animate_pendulum()

#pend_plots.plot_sol()
pend_plots.check_small_angle_sol()





