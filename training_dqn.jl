let
	using Pkg
	Pkg.activate(".")
end

using JSON, Plots, StatsBase, Flux, JLD2

include("constants_and_utils.jl")
include("dqn.jl")
include("environment_dqn.jl")

using .UtilsAndConstants
using .DQN
using .Env_DQN

#Load the data set
begin
	noised_time_train = let
	json_str = open("../time_series/noised_series_7_train.json","r") do file
		read(file,String)
	end
	JSON.parse(json_str)
	end

	noised_time_test = let
	json_str = open("../time_series/noised_series_7_test.json","r") do file
		read(file,String)
	end
	JSON.parse(json_str)
	end
end

#State and action sizes
begin
	const N_STATE = 4
	const N_ACTION = 1
	const N_ACTION_SPACE = 2
end

#Train and eval loop
begin
	model = Model(
		Chain(Dense(N_STATE + N_ACTION => 100,tanh),Dense(100 => 1, relu)),
		Adam(0.01),
		ReplayBuffer(;mem_size = 2000, state_size = N_STATE, action_size = N_ACTION),
		200,
		10,
		0.98,
		0.99
	)

	γ = model.γ
	G_train_episodes = [] #Averaged Return for training samples
	G_test_episodes = [] #Averaged Return for testing samples
	G_episodes = []
for episode in 1:2000
	#Evaluate performance every 10 training rounds:
	if (episode % 10) == 0
		#Put model into eval mode
		model.training_mode = false
		G_train_avg = 0
		G_test_avg = 0
		for ep in 1:50 #We have 50 training and testing episodes
			G_ep = 0
			for t in 1:40 #Each 40 slots long
				rsu_loads = map(1:5) do rsu_id
				Tuple(noised_time_train["($(rsu_id), $(i))"][ep][t] for i in 1:N_SERVICE)
				end
				server_load = sum.(rsu_loads) |> sum
				next_server_allocs = noised_time_train["allocs_2"][ep][t]
				if t<40
					rsu_loads_t′ = map(1:5) do rsu_id
						Tuple(noised_time_train["($(rsu_id), $(i))"][ep][t+1] for i in 1:N_SERVICE)
					end
					server_load_t′ = sum.(rsu_loads_t′) |> sum
					next_server_allocs_t′ = noised_time_train["allocs_2"][ep][t+1]
					terminated = false
				else
					rsu_loads_t′ = rsu_loads
					server_load_t′ = server_load
					next_server_allocs_t′ = next_server_allocs
					terminated = true
				end
				users = []
				for rsu_id in 1:N_RSU
					rsu_state = rsu_loads[rsu_id]
					rsu_load = sum(rsu_state)
					rsu_state_t′ = rsu_loads_t′[rsu_id]
					rsu_load_t′ = sum(rsu_state_t′)
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
				CRB_per_veh_mig = next_server_allocs ÷ (N_migs*(N_migs>0) + 1*(N_migs==0))
				R_t = 0
				map(users) do user
					user.CRB = user.migrate_service ? CRB_per_veh_mig : CRB_per_veh
					r = reward(user)
					R_t += r
				end
				
				G_ep += R_t * γ ^(t-1) 
			end
			G_train_avg += G_ep/50
		end
		push!(G_train_episodes, G_train_avg)
		for ep in 1:50 #Evaluate over test data
			G_ep = 0
			for t in 1:40 #Each 40 slots long
				rsu_loads = map(1:5) do rsu_id
				Tuple(noised_time_test["($(rsu_id), $(i))"][ep][t] for i in 1:N_SERVICE)
				end
				server_load = sum.(rsu_loads) |> sum
				next_server_allocs = noised_time_test["allocs_2"][ep][t]
				if t<40
					rsu_loads_t′ = map(1:5) do rsu_id
						Tuple(noised_time_test["($(rsu_id), $(i))"][ep][t+1] for i in 1:N_SERVICE)
					end
					server_load_t′ = sum.(rsu_loads_t′) |> sum
					next_server_allocs_t′ = noised_time_test["allocs_2"][ep][t+1]
					terminated = false
				else
					rsu_loads_t′ = rsu_loads
					server_load_t′ = server_load
					next_server_allocs_t′ = next_server_allocs
					terminated = true
				end
				users = []
				for rsu_id in 1:N_RSU
					rsu_state = rsu_loads[rsu_id]
					rsu_load = sum(rsu_state)
					rsu_state_t′ = rsu_loads_t′[rsu_id]
					rsu_load_t′ = sum(rsu_state_t′)
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
				CRB_per_veh_mig = next_server_allocs ÷ (N_migs*(N_migs>0) + 1*(N_migs==0))
				R_t = 0
				map(users) do user
					user.CRB = user.migrate_service ? CRB_per_veh_mig : CRB_per_veh
					r = reward(user)
					R_t += r
				end
				
				G_ep += R_t * γ ^(t-1) 
			end
			G_test_avg += G_ep/50
		end
		push!(G_test_episodes,G_test_avg)
		println("Evaluation results for eval - round $(episode / 10).......\n Training  : $(G_train_avg), Testing : $(G_test_avg)")
		#Checkpoint!!
		#Let us save the model parameters here....
		#Save the model as the following $(episode).jld2
		jldsave("dqn_params/$(episode).jld2", model_state = Flux.state(model.Q))
		#Evaluation ends here resume training
		model.training_mode = true
	end
	G_ep = 0
	for t in 1:40 
			#State is as follows : [#Vehs under RSU, #Vehs under server 1, #Allocs made by server 2, User's service type]
			rsu_loads = map(1:5) do rsu_id
				Tuple(noised_time_train["($(rsu_id), $(i))"][episode%50 + 1][t] for i in 1:N_SERVICE)
			end
			server_load = sum.(rsu_loads) |> sum
			next_server_allocs = noised_time_train["allocs_2"][episode%50 + 1][t]
			if t<40
				rsu_loads_t′ = map(1:5) do rsu_id
					Tuple(noised_time_train["($(rsu_id), $(i))"][episode%50 + 1][t+1] for i in 1:N_SERVICE)
				end
				server_load_t′ = sum.(rsu_loads_t′) |> sum
				next_server_allocs_t′ = noised_time_train["allocs_2"][episode%50 + 1][t+1]
				terminated = false
			else
				rsu_loads_t′ = rsu_loads
				server_load_t′ = server_load
				next_server_allocs_t′ = next_server_allocs
				terminated = true
			end
			users = []
			user_states = []
			user_next_states = []
			for rsu_id in 1:N_RSU
				rsu_state = rsu_loads[rsu_id]
				rsu_load = sum(rsu_state)
				rsu_state_t′ = rsu_loads_t′[rsu_id]
				rsu_load_t′ = sum(rsu_state_t′)
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
						s′ = [rsu_load_t′,
						server_load_t′,
						next_server_allocs_t′,
						service_id
						]
						push!(user_states, s)
						push!(user_next_states, s′)
					end
					push!(users, vehs...)					
				end
			end
			N_migs = count(user->user.migrate_service, users)
			CRB_per_veh = C̄ ÷ ((server_load - N_migs)*((server_load - N_migs)!=0) + (server_load - N_migs + 1)*((server_load - N_migs)==0))  
			CRB_per_veh_mig = next_server_allocs ÷ (N_migs*(N_migs>0) + 1*(N_migs==0))
			R_t = 0
			map(users, user_states, user_next_states) do user, s, s′
				user.CRB = user.migrate_service ? CRB_per_veh_mig : CRB_per_veh
				r = reward(user)
				store_transition!(model.memory, s, [user.migrate_service], r, terminated, s′)
				model.step_cnt += 1
				R_t += r
			end
			G_ep += R_t*γ^(t-1)
			#If we have fewer samples than the batch size, do not train network
			if model.step_cnt > model.batch_size
				learn!(model)
			end
	end
	model.ϵ -= 1/2000
	push!(G_episodes,G_ep)
end
    #Plot stuff
	p = plot(G_train_episodes,xlabel = "Training round", ylabel = "Averaged Return",label = "train_data")
	plot!(p,G_test_episodes,label = "test_data")
	q = plot(G_episodes, xlabel = "Training round", ylabel = "Return")
	Plots.pdf(p,"rewards.pdf")
	Plots.pdf(q,"rewards2.pdf")
end
