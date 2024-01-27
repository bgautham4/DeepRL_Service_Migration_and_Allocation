module Env_DQN #Defines types, attributes and useful functions for environment
include("constants_and_utils.jl")
using .UtilsAndConstants:transmit_rate
#Bring the common stuff such as Services types into this scope
include("environment.jl") 
using .Env: N_RSU, Service, SERVICE_SET, N_SERVICE, B̄, C̄, SLOT_DUR, RSU_R

const ACTION_SPACE = [false, true]
#The state for this environment is of form..
#[#Vehicles_under_RSU, #Vehicles_under_server, #Allocations_made_by_next_server, #Service_id]
#Action is of the form..
#[Migrate_service::Bool]

mutable struct User
	service::Service
	BRB::Int
	CRB::Int
	migrate_service::Bool
end

function migrate_service!(user::User)
	user.server_id == 0 ? user.server_id = 1 : user.server_id = 0
end

function reward(user::User)
	service = user.service
	bv = service.data_size * 1e6
	Cv = service.crb_needed
	τ = service.thresh
	τₘ = service.max_thresh
	R = transmit_rate(200,user.BRB)
	if user.BRB == 0 || user.CRB == 0
		u = 0 # utility for failing to allocate resources
	else
		del = bv/R + (Cv/user.CRB)
		u = del < τ ? 1.0 : del > τₘ ? 0.0 : -(del - τₘ)/(τₘ - τ) 
	end
	return u
end

export ACTION_SPACE, Service, SERVICE_SET, N_RSU, N_SERVICE, User, B̄, C̄, reward

end