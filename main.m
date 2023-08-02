clc
clear 
close all

rng('default')

dt = 0.01;
N = 500;
v = [1, 0, 0];
theta = 0.2;

% R = angvec2tr(theta, v)

sigma = 0.3;
% v     = (v + sigma.*randn(1,3))
% v     = v/norm(v)
% theta =  sigma.* randn() + theta


% R = rotx(pi/2)
% [theta, v] = tr2angvec(R)
% trplot(R)


R_seq = eye(4);

for i=2:N
    theta_dist = theta + sigma.* randn();
    v_dist = (v + sigma.*randn(1,3));
    v_dist = v_dist/norm(v_dist);

    R = angvec2tr(theta_dist * dt, v_dist);

    w_seq(i-1, :) = v_dist * theta_dist;
    R_seq(:,:,i) = R * R_seq(:,:,i-1);
    q_seq(i,:) = rotm2quat(R_seq(1:3,1:3,i));
end


q_att = rotm2quat(R_seq(1:3,1:3,end))

M = 3; % Dimension of matrix
A = sdpvar(M, M, 'symmetric', 'real');

Constraints = [A < eye(M)]; % X must be positive semidefinite


Objective = 0;
w_opt = sdpvar(M, N, 'full');
w_total_error = sdpvar(1,1);


for n=2:N
    q_curr = q_seq(n, :)
    
    vec_n = -q_curr(1) .* q_att(2:4) + q_att(1) .* q_curr(2:4) + cross(q_curr(2:4), q_att(2:4))

    w_n = A * vec_n'

    Objective = Objective + norm(w_n - w_seq(n-1,:)')^2
end

sdp_options = sdpsettings('solver','sedumi','verbose', 1, 'debug', 1);
sol = optimize(Constraints, Objective, sdp_options)


disp('Optimal value:');
disp(value(Objective));
disp('Optimal solution (A):');
disp(value(A));

A_opt = value(A)
tranimate(R_seq, 'fps', 1/dt, 'nsteps', N)
% % tranimate(inv(R))

% test

q_curr = q_seq(1, :)
vec_part = -q_curr(1) .* q_att(2:4) + q_att(1) .* q_curr(2:4) + cross(q_curr(2:4), q_att(2:4))

w_test = A_opt * vec_part'


q_curr = q_seq(2, :)
R_test_seq = R_seq(:,:,1)

for i=2:N+100

    vec_part = -q_curr(1) .* q_att(2:4) + q_att(1) .* q_curr(2:4) + cross(q_curr(2:4), q_att(2:4))
    w_test = A_opt * vec_part' * dt; % multiplied by the time difference
    
%     R_test = axang2rotm(w_test)


    R_test = angvec2tr(norm(w_test), w_test/norm(w_test));


    R_test_seq(:,:,i) = R_test * R_test_seq(:,:,i-1);
   

end
tranimate(R_test_seq, 'fps', 1/dt, 'nsteps', N)

q_curr = q_seq(N-1, :)
vec_par_att = -q_curr(1) .* q_att(2:4) + q_att(1) .* q_curr(2:4) + cross(q_curr(2:4), q_att(2:4))

w_att = A_opt * vec_par_att'

