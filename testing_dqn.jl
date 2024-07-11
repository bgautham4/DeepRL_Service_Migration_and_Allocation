let
	using Pkg
	Pkg.activate(".")
end

using JSON, Plots, StatsBase, Flux, JLD2, CSV

include("constants_and_utils.jl")
include("dqn.jl")
include("environment_dqn.jl")

using .UtilsAndConstants
using .DQN
using .Env_DQN

#State and action sizes
begin
	const N_STATE = 4
	const N_ACTION = 1
	const N_ACTION_SPACE = 2
end

#Evaluating the models performance based on various metrics:
begin
	model = Model(
		Chain(Dense(N_STATE + N_ACTION => 100,tanh),Dense(100 => 1, relu)),
		Adam(0.01),
		ReplayBuffer(;mem_size = 1, state_size = N_STATE, action_size = N_ACTION),
		200,
		10,
		0.98,
		0.99
	)
	#Load the params for the networks...
	let
		str  = "2000"
		Q_nw_state = JLD2.load("dqn_params/$(str).jld2", "model_state")
		Flux.loadmodel!(model.Q, Q_nw_state)
	end
	model.training_mode = false
end

#Load the test data set
begin
	noised_time_test = let
	json_str = open("path_to_time_series_file","r") do file
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
	N_episodes = 50
	for episode in 1:N_episodes #We have N_episode number test episodes
		Delays = [zeros(40) for _ in SERVICE_SET]
		Counts = [zeros(40) for _ in SERVICE_SET]
		Allocs = [zeros(40) for _ in SERVICE_SET]
		Non_allocs = [zeros(40) for _ in SERVICE_SET]
		Violations = [zeros(40) for _ in SERVICE_SET]
		Migrations = zeros(40)
		Possible_migs = zeros(40)
		for t in 1:40 # 40 time slots long
			rsu_loads = map(1:5) do rsu_id
				Tuple(noised_time_test["($(rsu_id), $(i))"][episode][t] for i in 1:N_SERVICE)
			end
			server_load = sum.(rsu_loads) |> sum
			next_server_allocs = noised_time_test["allocs_2"][episode][t]
			users = []
			for rsu_id in 1:N_RSU
				rsu_state = rsu_loads[rsu_id]
				rsu_load = sum(rsu_state)
				for service_id in 1:N_SERVICE
					vehs = [User(SERVICE_SET[service_id], B̄ ÷ rsu_load, 0, false) for _ in 1:rsu_state[service_id]]
					map(vehs) do veh
						s = [rsu_load,
						server_load,
						next_server_allocs,
						service_id
						]
						a = choose_action(model,s)
						veh.migrate_service = a
					end
					push!(users, vehs...)					
				end
			end
			N_migs = count(user->user.migrate_service, users)
			CRB_per_veh = C̄ ÷ ((server_load - N_migs)*((server_load - N_migs)!=0) + (server_load - N_migs + 1)*((server_load - N_migs)==0))  
			CRB_per_veh_mig = next_server_allocs ÷ (N_migs*(N_migs>0) + 1*(N_migs == 0))
			map(users) do user
				user.CRB = user.migrate_service ? CRB_per_veh_mig : CRB_per_veh
			end
			#Count number of migrations
			Migrations[t] = N_migs
			for (app_indx,app) in enumerate(SERVICE_SET)
				service_users = filter(users) do user
					user.service == app
				end 

				Allocs[app_indx][t] = length(service_users) > 1 ? service_users[1].CRB * length(service_users) : 0.0
				
				Counts[app_indx][t] += length(service_users)

				delays = map(service_users) do user
					bv = app.data_size * 1e6
					Cv = app.crb_needed
					R = transmit_rate(200,user.BRB)
					if user.BRB == 0 || user.CRB == 0
						del = app.max_thresh
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
		#add this to the episodic mean quantity
		service_dels .= service_dels .+ mean_delay
		vehicle_counts .= vehicle_counts .+ mean_counts
		allocs .= allocs .+ mean_allocs
		non_allocs .= non_allocs .+ mean_non_allocs
		violations .= violations .+ mean_violations
		global migs += mean_migrations
	end
    #Ensemble average over the N_episodes episodes
	service_dels .= service_dels ./ N_episodes 
	vehicle_counts .= vehicle_counts ./ N_episodes
	allocs .= allocs ./ N_episodes
	non_allocs .= non_allocs ./ N_episodes
	violations .= violations ./ N_episodes
	migs /= N_episodes


	Info_dict = Dict()
	Info_dict["service_dels"] = service_dels
	Info_dict["vehicle_counts"] = vehicle_counts
	Info_dict["allocs"] = allocs
	Info_dict["non_allocs"] = non_allocs
	Info_dict["violations"] = violations
	Info_dict["migs"] = migs
	for (key,val) in Info_dict
		println("$(key) ----> $(val)")
	end
	#save("veh_init_test.jld2", Info_dict)
	CSV.write("veh_init.csv", Info_dict)
end
