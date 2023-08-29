clc; clear
close all

%%% Verify some properties of Riemannian manifold, specifically Sphere-3 %%


% p = UnitQuaternion.Rx(0.1)
% 
% q = UnitQuaternion.Rx(0.5)
% 
% g = q * p
% 
% 
% vec = g.log
% 
% vec.exp



%%% Verify Rotation Difference between 2 unit quaternions %%

p = UnitQuaternion.Rx(0.1)

g = UnitQuaternion.Rx(0.5)

q_diff = p * g.conj

theta = log(q_diff)



