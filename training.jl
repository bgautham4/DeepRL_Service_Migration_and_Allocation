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

	function utility(user::User)
		reward_mul = [0.5,0.75,1]
		service = user.service
		bv = service.data_size * 1e6
		Cv = service.crb_needed
		τ = service.thresh
		τₘ = service.max_thresh
		R = transmit_rate(200,user.BRB)
		if user.BRB == 0 || user.CRB == 0
			u = -0.2 # utility for failing to allocate resources
		else
			del = bv/R + (Cv/user.CRB)
			u = del < τ ? 1.0 : del > τₘ ? 0.0 : -(del - τₘ)/(τₘ - τ)
			u = u*reward_mul[service.service_indx] + 0.2 #A small bonus for allocating resources
		end
		return u
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

    #Compute the reward given a state and an action,
    #While also generating a snapshot for further analysis
    function reward(state, action)
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
			#println("$(n_mig_candidates), $(sum(state[13:15])), $(W), $(mig_cnt)")
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

		#Calculate reward
		R = users .|> utility |> sum
		return R 
	end

end

#Load the data set
begin
	noised_time_train = let
	json_str = open("../time_series/noised_series_7_5_20_2_train.json","r") do file
		read(file,String)
	end
	JSON.parse(json_str)
	end

	noised_time_test = let
	json_str = open("../time_series/noised_series_7_5_20_2_test.json","r") do file
		read(file,String)
	end
	JSON.parse(json_str)
	end
end

#State and action sizes
begin
	const N_STATE = N_SERVICE * N_RSU + 3
	const N_ACTION = N_SERVICE*N_RSU + N_SERVICE + 2
end

STATE_ORDER = vcat(["($(i), $(j))" for i=1:5 for j=1:3],["rem_time","migs","allocs"]) #tells us the order of the state vector.

#A sample training and evaluation loop:
begin
	test_agent = Agent(
		Chain(Dense(N_STATE => 100,tanh),Dense(100 => N_ACTION, σ)),
		Chain(Dense(N_STATE+N_ACTION => 100,tanh),Dense(100 => 1,relu)),
		Adam(0.01),
		Adam(0.01),
		ReplayBuffer(;mem_size = 2000, state_size = N_STATE, action_size = N_ACTION),
		120,
		0.98,
		0.01,
		1
	)
	G_avg_train = [] #Averaged Return for training samples
	G_avg_test = [] #Averaged Return for testing samples
	γ = test_agent.γ

	for episode in 1:3000
		#Evaluation occurs here
		if (episode-1) % 10 == 0
			println("*******episode $(episode) starting*******")
			println("######Evaluation $(episode÷10) started #########")
			G_train = 0
			G_test = 0
			test_agent.training_mode = false
			for ep in 1:50
				g_ep = 0
				for t in 1:40
					st = [noised_time_train[key][ep][t] for key in STATE_ORDER]
					s = Flux.normalize(st) .|> Float32
					a = choose_action(test_agent, s)
					r = reward(st, action_mapper(a))
					g_ep += r*(γ^(t-1))
				end
				G_train += g_ep
			end
			push!(G_avg_train,G_train/50)
			for ep in 1:50
				g_ep = 0
				for t in 1:40
					st = [noised_time_test[key][ep][t] for key in STATE_ORDER]
					s = Flux.normalize(st) .|> Float32
					a = choose_action(test_agent, s)
					r = reward(st, action_mapper(a))
					g_ep += r*(γ^(t-1))
				end
				G_test += g_ep
			end
			push!(G_avg_test,G_test/50)
			println("The averaged return for the train data is $(G_train/50)")
			println("The averaged return for the test data is $(G_test/50)")
			#Checkpoint!!
			#Let us save the model parameters here....
			#Save the model as the following model_$(episode).jld2
			jldsave("actor_params/actor_$(episode).jld2", model_state = Flux.state(test_agent.actor))
			jldsave("critic_params/critic_$(episode).jld2", model_state = Flux.state(test_agent.critic))
			jldsave("target_actor_params/target_actor_$(episode).jld2", model_state = Flux.state(test_agent.target_actor))
			jldsave("target_critic_params/target_critic_$(episode).jld2", model_state = Flux.state(test_agent.target_critic))
		end
		#Evaluation ends here. Resume training

		test_agent.training_mode = true
		for t in 1:40 #Each episode is 40 slots long
			#Generate the state vectors s and s′
			st = [noised_time_train[key][episode%50 + 1][t] for key in STATE_ORDER]
			s = Flux.normalize(st) .|> Float32
			if t!=40
				s′ = [noised_time_train[key][episode%50 + 1][t+1] for key in STATE_ORDER]
				s′ = Flux.normalize(s′) .|> Float32
				t = false
			else
				s′ = s #No s′ when episode terminates. I just store 's' as a placeholder in the buffer.
				t = true
			end
			a = choose_action(test_agent, s) #Generate an action.
			r = reward(st, action_mapper(a)) #Obtain reward.
			#Store the transitions into replay buffer.
			store_transition!(test_agent.memory, s,a,r,t,s′)
			test_agent.step_cnt += 1

			#If we have fewer samples than the batch size, do not train network
			if test_agent.step_cnt > test_agent.batch_size
				learn!(test_agent)
			end
		end
		#test_agent.ϵ -= 0.001
	end

	#Plot some stuff
	p = plot(G_avg_train,xlabel = "Training round", ylabel = "Averaged Return",label = "train_data")
	plot!(p,G_avg_test,label = "test_data")
	Plots.pdf(p,"rewards.pdf")
end
