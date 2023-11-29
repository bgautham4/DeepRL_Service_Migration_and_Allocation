let
	using Pkg
	Pkg.activate(".")
end

using JSON, Plots, StatsBase, Flux, JLD2

include("constants_and_utils.jl")
include("ddpg.jl")

using .UtilsAndConstants
using .DDPG

#Define "User"(i.e: Vehicle) type and its associated functions:
begin
	mutable struct User
		rsu_id::Int
		service::Service
		BRB::Int
		CRB::Int
		mig_service::Bool
	end

	function action_mapper(x) #Map output from network to an appropriate workable form
		@assert length(x) == (N_SERVICE*N_RSU + N_SERVICE + 2)
        action = []
        for rsu_indx in 1:N_RSU
            rsu_allocs = x[(rsu_indx-1)*N_SERVICE + 1: (rsu_indx-1)*N_SERVICE + N_SERVICE]
            #Normalize the allocations to sum to 1
            rsu_allocs .= rsu_allocs ./ sum(rsu_allocs)
            push!(action,rsu_allocs)
        end

        crb_allocs = x[N_RSU*N_SERVICE + 1 : N_RSU*N_SERVICE + N_SERVICE + 1]
        #Normalize the CRB allocations to sum to 1
        crb_allocs .= crb_allocs ./ sum(crb_allocs)
        push!(action,crb_allocs)

        mig_fraction = x[end]
    	push!(action, mig_fraction)
    end	

    
    #generating a snapshot for further analysis
    function snapshot(state, action)
    	users = []
    	users_per_service = zeros(Int,N_SERVICE)
    	#Allocate B/W
		for rsu_indx in 1:N_RSU
			#Convert the fractional allocations to integers
			rsu_allocs = (action[rsu_indx] .* B̄) .|> x -> floor(Int,x)
			for (app_indx,app) in enumerate(SERVICE_SET)
				veh_cnt = state[(rsu_indx-1)*N_SERVICE + app_indx] #Get count of vehicles
				users_per_service[app_indx] += veh_cnt
				for _ in 1:veh_cnt
					BRB = rsu_allocs[app_indx] ÷ veh_cnt
					push!(users, User(rsu_indx,app,BRB,0,false))
				end
			end
		end
		#Sample some users for migration
		#And give them their CRB which is allocated equally
		W = state[N_RSU*N_SERVICE + 1]
		mig_cnt = floor(Int, W * action[end])
		#mig_descision = action[end] > 0.5 ? true : false
		mig_users = []
		if mig_cnt>0 && W>0
			Y = state[end] #Number of allocations made by your neighbour
			mig_candidates = filter(users) do user
				user.rsu_id == 5 || user.rsu_id == 4
			end
			n_mig_candidates = length(mig_candidates)
			#println("$(mig_cnt), $(n_mig_candidates), $(W), $(sum(state[12:15]))")
			mig_users = StatsBase.sample(mig_candidates, mig_cnt, replace = false)
			map(mig_users) do user
				#Due to migrations we have now freed up some CRBs
				indx = user.service.service_indx
				users_per_service[indx] -= 1 
				user.CRB = Y ÷ length(mig_users)
				user.mig_service = true
			end
		end
		#Allocate the CRBs to the remaining vehicles
		crb_allocs = (action[N_RSU + 1] .* C̄) .|> x->floor(Int,x)
		for (app_indx,app) in enumerate(SERVICE_SET)
			rem_users = filter(users) do user
				user.mig_service == false && user.service == app
			end
			map(rem_users) do user
				V = users_per_service[app_indx]
				#y = (y>0)*y + (y==0||y<0)*1
				user.CRB = crb_allocs[app_indx] ÷ V
			end
		end

		return users 
	end

end

#State and action sizes
begin
	const N_STATE = N_SERVICE * N_RSU + 3
	const N_ACTION = N_SERVICE*N_RSU + N_SERVICE + 2
end

STATE_ORDER = vcat(["($(i), $(j))" for i=1:5 for j=1:3],["rem_time","migs","allocs"]) #tells us the order of the state vector.

#Evaluating the models performance based on various metrics:
begin
	test_agent = Agent(
		Chain(Dense(N_STATE => 100,tanh),Dense(100 => N_ACTION, σ)),
		Chain(Dense(N_STATE+N_ACTION => 100,tanh),Dense(100 => 1,relu)),
		Adam(0.01),
		Adam(0.01),
		ReplayBuffer(;mem_size = 1, state_size = N_STATE, action_size = N_ACTION),
		120,
		0.98,
		0.01,
		1
	)
	#Load the params for the networks...
	let
		str  = "2991"
		actor_state = JLD2.load("actor_params/actor_$(str).jld2", "model_state")
		critic_state = JLD2.load("critic_params/critic_$(str).jld2", "model_state")
		target_actor_state = JLD2.load("target_actor_params/target_actor_$(str).jld2", "model_state")
		target_critic_state = JLD2.load("target_critic_params/target_critic_$(str).jld2", "model_state")
		Flux.loadmodel!(test_agent.actor, actor_state)
		Flux.loadmodel!(test_agent.critic, critic_state)
		Flux.loadmodel!(test_agent.target_actor, target_actor_state)
		Flux.loadmodel!(test_agent.target_critic, target_critic_state)
	end
	test_agent.training_mode = false
end


#Load the test data set
begin
	noised_time_test = let
	json_str = open("../time_series/noised_series_7_5_20_2_test.json","r") do file
		read(file,String)
	end
	JSON.parse(json_str)
	end
end

#Let us compute the average service delay, and the number of allocations per service
begin
	service_dels = zeros(length(SERVICE_SET)) #Mean service delay over all t and all episodes
	vehicle_counts = zeros(length(SERVICE_SET)) #Averaged counts of vehicles running service Sᵢ.
	allocs = zeros(length(SERVICE_SET)) #Averaged allocations
	non_allocs = zeros(length(SERVICE_SET))
	violations  = zeros(length(SERVICE_SET))
	rsu_allocs = [zeros(length(SERVICE_SET)) for _ in 1:N_RSU]
	migs = zero(Float64)
	for episode in 1:50 #We have 50 test episodes
		Delays = [zeros(40) for _ in SERVICE_SET]
		Counts = [zeros(40) for _ in SERVICE_SET]
		Allocs = [zeros(40) for _ in SERVICE_SET]
		Non_allocs = [zeros(40) for _ in SERVICE_SET]
		Violations = [zeros(40) for _ in SERVICE_SET]
		Rsu_allocs = [[zeros(40) for _ in SERVICE_SET] for _ in 1:N_RSU]
		Migrations = zeros(40)
		Possible_migs = zeros(40)
		for t in 1:40 # 40 time slots long
			st = [noised_time_test[key][episode][t] for key in STATE_ORDER]
			s = Flux.normalize(st) .|> Float32
			a = choose_action(test_agent, s)
			users = snapshot(st,action_mapper(a)) #Obtain the snapshot of the users
			#Count number of migrations
			Migrations[t] = count(user->user.mig_service, users)
			for (app_indx,app) in enumerate(SERVICE_SET)
				service_users = filter(users) do user
					user.service == app
				end 

				Allocs[app_indx][t] = length(service_users) > 1 ? service_users[1].CRB * length(service_users) : 0.0
				
				Counts[app_indx][t] += length(service_users)

				delays = map(service_users) do user
					Rsu_allocs[user.rsu_id][app_indx][t] += user.BRB
					bv = app.data_size * 1e6
					Cv = app.crb_needed
					R = transmit_rate(200,user.BRB)
					if user.BRB == 0 || user.CRB == 0
						del = 0
						Non_allocs[app_indx][t] += 1
						Violations[app_indx][t] += 1
					else
						del = bv/R + (Cv/user.CRB)
						del > app.max_thresh ? Violations[app_indx][t] += 1 : nothing
					end
					del
				end
				delays = length(delays) > 0 ? delays : [0.0]
				push!(Delays[app_indx], mean(delays)) #mean(delays) is the mean service delay at time t
			end
		end
		#Compute the mean quatity over all time slots t
		mean_delay = Delays .|> mean
		mean_counts = Counts .|> mean
		mean_allocs = Allocs .|> mean
		mean_non_allocs = Non_allocs .|> mean
		mean_violations = Violations .|> mean
		mean_migrations = mean(Migrations)
		mean_rsu_allocs = map(Rsu_allocs) do service_allocs
								map(service_allocs) do service_alloc
									mean(service_alloc)
								end
							end
		#add this to the episodic mean quantity
		service_dels .= service_dels .+ mean_delay
		vehicle_counts .= vehicle_counts .+ mean_counts
		allocs .= allocs .+ mean_allocs
		non_allocs .= non_allocs .+ mean_non_allocs
		violations .= violations .+ mean_violations
		global migs += mean_migrations
		rsu_allocs .= rsu_allocs .+ mean_rsu_allocs
	end
	service_dels .= service_dels ./ 50 #Average over the 50 episodes
	vehicle_counts .= vehicle_counts ./ 50
	allocs .= allocs ./ 50
	non_allocs .= non_allocs ./ 50
	violations .= violations ./ 50
	migs /= 50
	rsu_allocs .= rsu_allocs ./ 50
	possible_migs = [mean(noised_time_test["rem_time"][ep]) for ep in 1:50] |> mean


	Info_dict = Dict()
	Info_dict["service_dels"] = service_dels
	Info_dict["vehicle_counts"] = vehicle_counts
	Info_dict["allocs"] = allocs
	Info_dict["non_allocs"] = non_allocs
	Info_dict["violations"] = violations
	Info_dict["migs"] = migs
	Info_dict["rsu_allocs"] = rsu_allocs
	Info_dict["possible_migs"] = possible_migs
	for (key,val) in Info_dict
		println("$(key) ----> $(val)")
	end
	save("sim_7_5_20_2_test.jld2", Info_dict)
end