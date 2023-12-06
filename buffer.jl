module Buffer #Code for the replay memory implemntation.

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