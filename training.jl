let
	using Pkg
	Pkg.activate(".")
end

using JSON, Plots, StatsBase, Flux, JLD2

include("constants_and_utils.jl")
include("ddpg.jl")
include("environment.jl")

using .UtilsAndConstants
using .DDPG
using .Env


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
