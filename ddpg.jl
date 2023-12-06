module DDPG #contains code algorithm definition

#Bring the buffer code here
include("buffer.jl")
#Make use of the ReplayBuffer struct and its methods
using .Buffer: ReplayBuffer, store_transition!, sample_buffer

export Agent, learn!, choose_action, ReplayBuffer, store_transition! #Export these useful functions to calling scope 

using Flux, StatsBase

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
