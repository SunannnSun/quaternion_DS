clc; clear
close all

rng('default')

dt = 0.1;

% Generate demonstrations as training data
N = 50;
var = 0.1; 

axis = [1, 0, 0];
ang_vel = pi/6;

R_train = SO3;
q_train = UnitQuaternion();
w_train = [0, 0, 0];

for n=1:N
    axis_n  = axis + var.* randn(1,3);
    axis_n  = axis_n/norm(axis_n);
    ang_vel_n = (ang_vel + var * randn()) * (N-n)/N; % Decaying velocity
    w_train(n, :) = ang_vel_n * axis_n;
   
    q = UnitQuaternion.angvec(ang_vel_n * dt, axis_n);
    q_train(n+1) = q * q_train(n);
    R_train(n+1) = q_train(n+1).R;
end



% Construct and solve optimization problem
M = 3; 
A = sdpvar(M, M, 'symmetric', 'real');

Constraints = [A <= eye(M)];

Objective = 0;
for n=2:N
    q_diff = q_train(n) * q_train(end).conj();
    
    w_out = A * q_diff.v';

    Objective = Objective + norm(w_out - w_train(n,:)')^2;
end

sdp_options = sdpsettings('solver','sedumi','verbose', 1, 'debug', 1);
sol = optimize(Constraints, Objective, sdp_options)


disp('Optimal value:');
disp(value(Objective));
disp('Optimal solution (A):');
disp(value(A));



% Test the trained DS 
N_test = N +300;

A_train = value(A);
% q_test = q_train(2);
q_test = UnitQuaternion.Ry(0.5) * UnitQuaternion.Rz(0.5)
R_test = q_test.R;

for n=2:N_test
    
    q_diff = q_test(n-1) * q_train(end).conj();
    w_test = A_train * q_diff.v';

    axis_n  = w_test/norm(w_test);
    ang_vel_n = norm(w_test);
    
    q = UnitQuaternion.angvec(ang_vel_n * dt, axis_n);

    q_test(n) = q * q_test(n-1);
    R_test(:,:,n) = q_test(n).R;
end

while true
    tranimate(R_train, 'fps', 1/dt, 'nsteps', N)
    figure;
    tranimate(R_test, 'fps', 1/dt, 'nsteps', N_test)
end



