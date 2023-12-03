module DDPG #contains code for buffer and agent definition

module Buffer

export ReplayBuffer, store_transition!, sample_buffer
using StatsBase

mutable struct ReplayBuffer
	mem_size::Int
	mem_cntr::Integer
	state_mem
	action_mem
	reward_mem
	terminated_mem
	next_state_mem
end
function ReplayBuffer(;mem_size, state_size, action_size)
	state_mem = zeros(Float32,state_size,mem_size)
	action_mem = zeros(Float32,action_size,mem_size)
	reward_mem = zeros(Float32,mem_size)
	terminated_mem = zeros(Bool,mem_size)
	next_state_mem = zeros(Float32,state_size,mem_size)
	ReplayBuffer(mem_size,
		0,
		state_mem,
		action_mem,
		reward_mem,
		terminated_mem,
		next_state_mem
	)
end
function store_transition!(mem::ReplayBuffer,state, action, reward, terminate, next_state)
	indx = (mem.mem_cntr % mem.mem_size) + 1
	mem.state_mem[:,indx] = state
	mem.action_mem[:,indx] = action
	mem.reward_mem[indx] = reward
	mem.terminated_mem[indx] = terminate
	mem.next_state_mem[:,indx] = next_state
	mem.mem_cntr += 1
end
##!!! Only sample batch_size after you have stored atleast a batch_size number of transitions
function sample_buffer(mem::ReplayBuffer, batch_size::Int)
	max_mem = min(mem.mem_cntr,mem.mem_size) 
	indices = StatsBase.sample(1:max_mem, batch_size, replace = false)
	states = mem.state_mem[:, indices]
	actions = mem.action_mem[:, indices]
	rewards = mem.reward_mem[indices]
	terminated = mem.terminated_mem[indices]
	next_states = mem.next_state_mem[:,indices]
	(states, actions, rewards, terminated, next_states)
end


end

module AgentDefinition
export Agent, learn!, choose_action
using Flux, StatsBase
using ..Buffer #Get definitions for buffer type and functions
mutable struct Agent
	actor
	critic
	target_actor
	target_critic
	memory::ReplayBuffer
	actor_opt_state
	critic_opt_state
	training_mode::Bool
	batch_size::Int
	step_cnt::Int
	γ::Real
	τ::Real
	ϵ::Real
end
function Agent(actor, critic, actor_optim, critic_optim, memory::ReplayBuffer , batch_size::Int ,γ = 0.99 ,τ = 0.01 ,ϵ = 0.99)
	target_actor = Flux.deepcopy(actor)
	target_critic = Flux.deepcopy(critic)
	actor_opt_state = Flux.setup(actor_optim,actor)
	critic_opt_state = Flux.setup(critic_optim,critic)
	Agent(
		actor,
		critic,
		target_actor,
		target_critic,
		memory,
		actor_opt_state,
		critic_opt_state,
		true,
		batch_size,
		0,
		γ,
		τ,
		ϵ
	)
end
function choose_action(agent::Agent, obs)
	if agent.training_mode
		action = Flux.unsqueeze(obs,2) |> agent.actor
		δ = 0.01
		Δ = 0.15
		noised_action = clamp.(action .+ Δ .* randn(length(action)), δ,1)
		#epsilon greedy method for selecting actions
		rand() > agent.ϵ ? (return action) : (return noised_action)
		return noised_action
	else
		return Flux.unsqueeze(obs,2) |> agent.actor
	end
end
function learn!(agent::Agent)
	s,a,r,t,s′ = sample_buffer(agent.memory, agent.batch_size)
	γ = agent.γ
	τ = agent.τ		
	#Bootstrap target
	y = Flux.unsqueeze(r,1) .+ γ .* (1 .- Flux.unsqueeze(t,1)) .* agent.target_critic(vcat(s′,agent.target_actor(s′)))
	#Optimize the critic network using bootstrap target:
	∇Q = Flux.gradient(agent.critic) do Q
		loss = mean((y .- Q(vcat(s,a))).^2)
	end
	Flux.update!(agent.critic_opt_state, agent.critic, ∇Q[1])
	#Optimize the actor network by maximizing value (minimizing -value):
	∇μ = Flux.gradient(agent.actor) do μ
		a_mu = μ(s)
		loss = -mean(agent.critic(vcat(s,a_mu)))
	end
	Flux.update!(agent.actor_opt_state, agent.actor, ∇μ[1])
	#Update target networks by using polyak averaging:
	for (θ,θ′) in zip(Flux.params(agent.actor),Flux.params(agent.target_actor))
		θ′ .= τ .* (θ) .+ (1-τ) .* (θ′)
	end		
	for (θ,θ′) in zip(Flux.params(agent.critic),Flux.params(agent.target_critic))
		θ′ .= τ .* (θ) .+ (1-τ) .* (θ′)
	end
end

end

#Export the functions out
using .Buffer
export ReplayBuffer, store_transition!, sample_buffer

using .AgentDefinition
export Agent, choose_action, learn!
end
