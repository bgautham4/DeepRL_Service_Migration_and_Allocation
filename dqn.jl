#contains code for the algorithm
#Here we make use of target network for added stability during training.
module DQN 

#Bring the buffer code here
include("buffer.jl")
#Make use of ReplayBuffer and its associated methods
using .Buffer: ReplayBuffer, store_transition!, sample_buffer

#Using environment, get the valid space of actions.
include("environment_dqn.jl")
using .Env_DQN: ACTION_SPACE

using Flux, StatsBase

mutable struct Model
	Q
	Q_target
	memory::ReplayBuffer
	opt_state
	training_mode::Bool
	batch_size::Int
	step_cnt::Int
	update_freq::Int
	γ::Real
	ϵ::Real
end

function Model(Q, opt, memory::ReplayBuffer, batch_size::Int, update_freq = 10, γ = 0.98, ϵ = 0.99)
	Q_target = Flux.deepcopy(Q)
	opt_state = Flux.setup(opt, Q)
	Model(
		Q,
		Q_target,
		memory,
		opt_state,
		true,
		batch_size,
		0,
		update_freq,
		γ,
		ϵ
		)
end

function choose_action(model::Model, obs)
	obs = vec(obs)
	tot_obs  = hcat((vcat(obs,a) for a in ACTION_SPACE)...)
	Q_vals = model.Q(tot_obs)
	_,i = findmax(Q_vals)
	i = LinearIndices(Q_vals)[i]
	action = ACTION_SPACE[i]
	if model.training_mode
		action = rand() < model.ϵ ? rand(ACTION_SPACE) : action
	end
	return action
end

function learn!(model::Model)
	γ = model.γ
	s,a,r,t,s′ = sample_buffer(model.memory, model.batch_size)
	model.training_mode = false		
	a′ = map(1:model.batch_size) do x
		choose_action(model,s[:,x])
	end
	a′ = Flux.unsqueeze(a′,1)
	model.training_mode = true
	#Bootstrap target
	y = Flux.unsqueeze(r,1) .+ γ .* (1 .- Flux.unsqueeze(t,1)) .* model.Q_target(vcat(s′,a′))
	
	#Optimize the Q network using bootstrap target:
	∇Q = Flux.gradient(model.Q) do Q
		loss = mean((y .- Q(vcat(s,a))).^2)
	end
	Flux.update!(model.opt_state, model.Q, ∇Q[1])
	
	#Update target network:
	if (model.step_cnt - model.batch_size) % model.update_freq == 0
		for (θ,θ′) in zip(Flux.params(model.Q),Flux.params(model.Q_target))
			θ′ .= θ 
		end
	end		
end

export Model, ReplayBuffer, store_transition!, choose_action, learn! #Export out these useful functions to calling scope

end

