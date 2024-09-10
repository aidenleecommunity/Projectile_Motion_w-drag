import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp

theta_high = 90* np.pi / 180
theta_low = 0* np.pi / 180
error = 100 #meters
error_bound = 0.05 #meters
n = 0
C = 0.02155 #D/m = pCA/2m (p: air density, C: drag coefficient, A: cross-sectional area of the ball(seen from front))
V = 15 #initial velocity m/s
x = 10 #get results from Yolov5 and Kalman Filter results (m)
y = 0 #target position (m)

def dSdt(t,S,C):
    px, vx, py, vy = S
    return [vx,
            -C*np.sqrt(vx**2+vy**2)*vx,
            vy,
            -9.8-C*np.sqrt(vx**2+vy**2)*vy]

while abs(error) > error_bound and n < 10:
    if error > 0:
        theta_high = (theta_high + theta_low) / 2
    else:
        theta_low = (theta_high + theta_low) / 2
    theta = (theta_high + theta_low) / 2 #firing angle
    
    #4 ODE solution
    sol = solve_ivp(dSdt, [0, 2], y0=[0,V*np.cos(theta),0,V*np.sin(theta)], t_eval=np.linspace(0,2,1000), args=(C,)) #atol=1e-7, rtol=1e-4
    #finding where projectile is close to target x
    x_near = np.where(abs(sol.y[0] - x) < 0.05, sol.y[0], 0)
    x_near = np.delete(x_near, np.where(x_near == 0))
    target_indices = np.where(np.isclose(sol.y[0], x_near[0]))
    #updating distance between projectile and real target position
    error = sol.y[2, target_indices] - y
    #how long it takes to reach the target
    reach_time = sol.t[target_indices]
    n=n+1
result_list = [n, theta* 180 / np.pi, error[0][0], reach_time[0]]
print(result_list)
if error[0][0] > 0:
    real_theta = theta*180/np.pi - 9.144*abs(error[0][0]) / (x*0.1595628)
else:
    real_theta = theta*180/np.pi + 9.144*abs(error[0][0]) / (x*0.1595628)

print(real_theta)