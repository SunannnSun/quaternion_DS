clc; clear
close all

rng('default')

dt = 0.1;

% Generate demonstrations as training data
N1 = 30;
N2 = 30;
N = N1 + N2;

var = 0.1; 

axis = [repmat([1,0,0], N1, 1);  repmat([0,0,1], N2, 1)];
ang_vel = [repmat(pi/3, N1, 1);  repmat(pi/3, N2, 1)];

R_train = SO3;
q_train = UnitQuaternion();
w_train = [0, 0, 0];

for n=1:N
    axis_n  = axis(n, :) + var.* randn(1,3);
    axis_n  = axis_n/norm(axis_n);
    ang_vel_n = (ang_vel(n) + var * randn()) * (N-n)/N; % Decaying velocity
    w_train(n, :) = ang_vel_n * axis_n;
   
    q = UnitQuaternion.angvec(ang_vel_n * dt, axis_n);
    q_train(n+1) = q * q_train(n);
    R_train(n+1) = q_train(n+1).R;
end



tranimate(R_train, 'fps', 1/dt/2, 'nsteps', N)


