### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 161af454-2de6-11ee-05de-c9432d420a24
let
	using Pkg
	Pkg.activate(".")
end

# ╔═╡ 235616e2-3377-4a03-bd2e-86514d73b262
using Plots

# ╔═╡ 0d50b122-7614-4acf-8294-8f02c660dc25
using ReinforcementLearning, Distributions, StatsBase

# ╔═╡ 04fe6e61-bd64-4204-9104-76e39d8a6c46
begin
	const GRID_SIZE = 5 # in km
	const SLOT_DUR = 1.5 # in minutes
	const RSU_R = 0.5 # coverage radius of RSU in km 
end

# ╔═╡ 98cffce9-5036-4fea-88e8-fe82463c1e47
begin
	
	@enum Service service1 = 1 service2 = 2 service3 = 3

	mutable struct Path
		start::Real
		fin::Real
		v_intr::Real
		eta::Real  
		#n_vehs::Integer
	end

	abstract type AbstractVehicle end

	mutable struct Vehicle <: AbstractVehicle
		veh_id::Integer
		service::Service
		position::Real
		velocity::Real
		rsu_id::Integer
		#prev_tslot_rsu_id::Integer
		#old_rsu_id::Integer
		edge_id::Integer
		allocated_BRB::Integer
		allocated_CRB::Integer
	end

	mutable struct DummyVehicle <: AbstractVehicle
		position::Real
		velocity::Real
	end
	
	mutable struct RSU
		rsu_id::Integer
		position::Real
		vehicles::Vector{Vehicle}
		vaccant_BRB::Integer
	end


	mutable struct EdgeServer
		edge_id::Integer
		#path::Path
		#rsus::Vector{RSU}
		vehicles::Vector{Vehicle}
		buff::Vector{Vehicle}
		#buff_alloc::Integer
		vaccant_CRB::Integer
	end

end

# ╔═╡ 17968306-6e98-453d-b087-618efedd3e55
begin
	const SERVICE_SET = (service1,service2,service3)
	#const MAX_TSLOTS = 5
end

# ╔═╡ 53ad2d23-2e07-47b1-b7f7-394c69218929
begin
	#Basic utility functions
	function distance(p1,p2)
		abs(p1-p2)
	end
end

# ╔═╡ c1bb1b4c-fc8e-4371-866c-40bbb6c7746f
begin
	# Functions for running simulation
	"""
	function incr_vehicle_count!(path::Path)
		path.n_vehs += 1
	end

	function decr_vehicle_count!(path::Path,rev_flag::Bool)
		path.n_vehs -= 1
	end
	"""

	function in_coverage(vehicle::Vehicle,rsu::RSU) :: Bool
		distance(rsu.position, vehicle.position) <= RSU_R
	end

	
	function update_vehicle_position!(vehicle::AbstractVehicle)
		vehicle.position += vehicle.velocity * (SLOT_DUR / 60)
	end


	function gen_snr(l_v::Real, BRB::Integer)
		N0=-174 # Noise spectral density (dbm/Hz)
		B= BRB * 180e3 # Bandwidth
		f=3.5e9 # Carrier frequency (Hz)
		d0=100 # Reference distance (m)
		#d=500 # Distance of vehicle to BS (m)
		gam=3.75 # Pathloss exponent (outdoor)
		c=3e8 # Speed of light (m/s)
		shw=8 # standard deviation of log-normal shadowing (dB)

		N=N0+10*log10(B) # Noise power (dBm)
		Pt=200e-3 # Transmit power of vehicle (W)
		Ptm=10*log10(Pt)+30 #Transmit power of vehicle (dBm)
		K=-20*log10(4*pi*d0*f/c) # Free space pathloss (dB)
		PL=K-10*gam*log10(l_v/d0)+shw*randn() # Simplified path loss with shadowing (dB)
		Pr=Ptm+PL # Received power at BS (dBm)
		snr=Pr-N-30 # SNR (dB)

		return 10^(snr/10)
	end

	function transmit_rate(l_v::Real, BRB::Integer)
		SNR = gen_snr(l_v, BRB)
		#SNR == Inf ? SNR = 1 : nothing
		R = BRB * 180e3 * log2(1+SNR)
		return R
	end

	function app_aggr_rsu(rsu::RSU) :: Dict{Service, Vector{Vehicle}}
		veh_app_info = Dict()
		for service in SERVICE_SET
			veh_app_info[service] = filter(vehicle -> vehicle.service == service, rsu.vehicles)
		end
		veh_app_info
	end

	function app_aggr_edge(edge::EdgeServer) :: Dict{Service, Vector{Vehicle}}
		veh_app_info = Dict()
		for service in SERVICE_SET
			veh_app_info[service] = filter(vehicle -> vehicle.service == service, edge.vehicles)
		end
		veh_app_info
	end

	function find_all_vehs!(rsu::RSU, vehicles::Vector{Vehicle})
		indx = findall(vehicle->in_coverage(vehicle,rsu), vehicles)
		rsu.vehicles = vehicles[indx]
		map(vehicles[indx]) do vehicle
			vehicle.rsu_id = rsu.rsu_id
		end
		#rsu.vaccant_BRB = 50
	end

	function find_all_vehs_edge!(server::EdgeServer, vehicles::Vector{Vehicle})
		indx = findall(vehicle->vehicle.edge_id == 1, vehicles)
		server.vehicles = vehicles[indx]
	end
	
end

# ╔═╡ 5d72579f-160c-420f-bc37-6405b37512e3
values(Dict(1=>2,2=>3)) |> collect 

# ╔═╡ 3e3071b2-8daa-4168-9c21-043e792e4e2c
Integer(service1)

# ╔═╡ 3011b8b9-bb32-4c4b-a718-59ca147b0e7f
begin

	mutable struct ServiceMigrationEnv <: RL.AbstractEnv
		path::Path
		rsus::Vector{RSU}
		server::EdgeServer
		vehicles::Vector{Vehicle}
		#n_migs::Integer
		#n_allocs::Integer
		n_vehs::Integer
		#continuous::Bool
		step_no::Integer
		max_steps::Integer
		reward::Real
	end

	function RLBase.reset!(env::ServiceMigrationEnv)
		env.rsus = [RSU(i,0.5+i-1,[],50) for i in 1:5]
		v_distr = truncated(Normal(32,4);lower = 24,upper = 40)
		env.vehicles = [Vehicle(i,rand(SERVICE_SET),rand(0:0.125:5),rand(v_distr),0,1,0,0) for i in 1:40]
		map(env.vehicles) do vehicle
			rsu_indx = findfirst(rsu -> in_coverage(vehicle,rsu),env.rsus)
			push!(env.rsus[rsu_indx].vehicles,vehicle)
			vehicle.rsu_id = rsu_indx
		end
		env.server = EdgeServer(1,copy(env.vehicles),[],100)
		env.path = Path(0,5,32.5,0.018)
		env.step_no = 0
		env.max_steps = 60
		env.reward = 0
		env.n_vehs = 40
	end
	
	RLBase.is_terminated(env::ServiceMigrationEnv) = env.step_no >= env.max_steps 

 	function RLBase.state(env::ServiceMigrationEnv)
		rsu_groups = map(env.rsus) do rsu
			app_infos = app_aggr_rsu(rsu)
			[length(app_info) for app_info in values(app_infos)]
		end

		#edge_groups = [length(app_info) for app_info in (env.server |> app_aggr_edge |> values)]
		
		vcat(
			rsu_groups...,
			#edge_groups,
			length(env.server.buff), #Migrations from prev edge server
			#env.n_allocs #Previous allocations made by next edge server
			)	
	end

	function RLBase.state_space(env::ServiceMigrationEnv)
		Space(
				vcat(
					[0:50 for _ in env.rsus for _ in SERVICE_SET],
					[0:20],
					#[0:10],
					)
			)
	end

	function RLBase.action_space(env::ServiceMigrationEnv)
		rsu_groups = map(env.rsus) do rsu
			app_infos = app_aggr_rsu(rsu)
			[length(app_info) for app_info in values(app_infos)]
		end
		edge_dict = app_aggr_edge(env.server)
		edge_groups = map(SERVICE_SET) do service
			length(edge_dict[service])
		end
		Space(
			vcat(
				[rsu_app_cnt:rsu_app_cnt + rsu.vaccant_BRB ÷ length(SERVICE_SET) for (rsu_group,rsu) in zip(rsu_groups,env.rsus) for rsu_app_cnt in rsu_group], #RSU app groups BRB allocation
				[edge_groups[i]:edge_groups[i] + env.server.vaccant_CRB ÷ (length(SERVICE_SET)+1) for i in 1:length(SERVICE_SET)], #Edge app group CRB allocation
				[length(env.server.buff):length(env.server.buff) + env.server.vaccant_CRB ÷ (length(SERVICE_SET)+1)], #Allocations for migrations by prev edge server
				[0:rsu_app_num for rsu_group in rsu_groups for rsu_app_num in rsu_group] #Per app migrations from each RSU of edge server
			)
		)
			
		
	end

	function _migrate_vehicles!(env::ServiceMigrationEnv, action)
		n_rsus = length(env.rsus)
		n_services = length(SERVICE_SET)
		mig_counts = action[n_rsus*n_services+n_services+2:end]
		for (rsu_indx,rsu) in enumerate(env.rsus)
			for (app_indx,app) in enumerate(SERVICE_SET)
				n_migs = mig_counts[(rsu_indx-1)*n_services + app_indx]
				vehs = filter(vehicle -> vehicle.service == app && vehicle.edge_id == 1, rsu.vehicles)
				if n_migs >= length(vehs)
					map(vehs) do veh
						veh.edge_id = 0
						#env.server.vaccant_CRB += veh.allocated_CRB
					end
				else
					map(StatsBase.sample(vehs,n_migs,replace=false)) do veh
						veh.edge_id = 0
						#env.server.vaccant_CRB += veh.allocated_CRB
					end
				end
			end
		end
	end

	function _allocate_BRB!(env::ServiceMigrationEnv, action)
		n_rsus = length(env.rsus)
		n_services = length(SERVICE_SET)
		allocs = action[1:n_rsus*n_services]
		for(rsu_indx,rsu) in enumerate(env.rsus)
			rsu.vaccant_BRB = 50
			for (app_indx,app) in enumerate(SERVICE_SET)
				alloc_app = allocs[(rsu_indx-1)*n_services + app_indx]
				vehs = filter(vehicle -> vehicle.service == app, rsu.vehicles)
				N_vehs = length(vehs)
				map(vehs) do veh
					veh.allocated_BRB = alloc_app ÷ N_vehs
					rsu.vaccant_BRB -= veh.allocated_BRB
				end
			end
		end
	end

	function _allocate_CRB!(env::ServiceMigrationEnv, action)
		n_rsus = length(env.rsus)
		n_services = length(SERVICE_SET)
		env.server.vaccant_CRB = 100
		allocs = action[n_rsus*n_services + 1:n_rsus*n_services + n_services]
		#env.server.vehicles = vcat([rsu.vehicles for rsu in env.rsus]...) |> filter(vehicle -> vehicle.edge_id == 1)
		env.server.vehicles = filter(veh->veh.edge_id==1, env.server.vehicles)
		app_dict = app_aggr_edge(env.server) 
		N_vehs_apps = map(SERVICE_SET) do service
			length(app_dict[service])
		end #No of users in each app type
		for vehicle in env.server.vehicles
			app_indx = Int(vehicle.service)
			vehicle.allocated_CRB = allocs[app_indx] ÷ N_vehs_apps[app_indx]
			env.server.vaccant_CRB -= vehicle.allocated_CRB
		end

		alloc_prev_server = action[n_rsus*n_services + n_services + 1]
		N_migs = length(env.server.buff)
		for vehicle in env.server.buff
			vehicle.allocated_CRB = alloc_prev_server ÷ N_migs
			env.server.vaccant_CRB -= vehicle.allocated_CRB
		end
	end

	function _step!(env::ServiceMigrationEnv)
		INJECTION_RATE = 7
		Distr = Poisson(INJECTION_RATE)
		v_distr = truncated(Normal(32,4);lower = 24,upper = 40)
		N_inject = rand(Distr)
		new_vehs = [Vehicle(i, rand(SERVICE_SET), rand(0:0.125:0.5), rand(v_distr), 1, 1, 0, 0) for i in 1:N_inject]
		push!(env.server.vehicles, env.server.buff...) #Move previous migrations to be under edge server
		env.server.buff = new_vehs
		map(env.vehicles) do vehicle
			update_vehicle_position!(vehicle)
		end

		to_delete = map(vehicle->vehicle.position > 5, env.vehicles)
		deleteat!(env.vehicles, to_delete)
		find_all_vehs_edge!(env.server, env.vehicles)
		push!(env.vehicles, new_vehs...)
		map(env.rsus) do rsu
			find_all_vehs!(rsu,env.vehicles)
		end
		env.n_vehs = length(env.vehicles)
		env.step_no += 1 #Step through time
	end
		
		
		

	function(env::ServiceMigrationEnv)(action)
		#@assert action in RLBase.action_space(env)
		_migrate_vehicles!(env,action)
		_allocate_BRB!(env,action)
		_allocate_CRB!(env,action)
		#Compute reward
		R = 0
		for vehicle in env.vehicles
			x_v = vehicle.position
			x_r = env.rsus[vehicle.rsu_id].position
			l_v = distance(x_v,x_r)*1000
			l_v = l_v + 10*(l_v < 10)
			tx_rate = transmit_rate(l_v, vehicle.allocated_BRB + 1*(vehicle.allocated_BRB==0))
			tx_time = 100 / tx_rate
			comp_time = 0.5 / (vehicle.allocated_CRB)
			#println(comp_time)
			r_v = exp(-(comp_time+tx_time))
			#println("$(r_v),$(tx_rate), $(comp_time), $(tx_time)")
			R += r_v
		end
		env.reward = R / length(env.vehicles)
		_step!(env)
	end

	
	function RLBase.reward(env::ServiceMigrationEnv)
		env.reward
	end	
	

	function ServiceMigrationEnv()
		env = ServiceMigrationEnv(
			Path(0,0,0,0),
			[RSU(0,0,[],0),],
			EdgeServer(0,[],[],0),
			[],
			0,
			0,
			0,
			0
		)
		reset!(env)
		env
	end
	"""
	function inject_vehicles!()
		nothing
	end


	"""
	function RLBase.ChanceStyle(::ServiceMigrationEnv)
		RLBase.DETERMINISTIC
	end
	
end

# ╔═╡ c4b2eea4-c984-4591-8e62-841ef654a677
begin
	test_env = ServiceMigrationEnv()
end

# ╔═╡ 2c8406e2-2928-4d14-a669-3c01ae46f111
RLBase.test_runnable!(test_env)

# ╔═╡ 422746a8-4131-4463-b466-9118fcaa7587
res = run(RandomPolicy(action_space(test_env)), test_env, StopAfterEpisode(1_00),TotalRewardPerEpisode())

# ╔═╡ f32ca1c1-30d6-40b4-8f7d-4290fed932de
plot(res.rewards)

# ╔═╡ 0d7b668c-6df9-4372-895b-17ea13f2e829


# ╔═╡ 580aed47-941e-4ef7-bdbe-2fbf5332dcd8
?ActionTransformedEnv

# ╔═╡ cffa1013-ee3a-48c6-a395-301ffada1109
?DDPGPolicy

# ╔═╡ e91d1f09-9633-46f8-9734-f0aa83312507
?legal

# ╔═╡ Cell order:
# ╠═161af454-2de6-11ee-05de-c9432d420a24
# ╠═235616e2-3377-4a03-bd2e-86514d73b262
# ╠═04fe6e61-bd64-4204-9104-76e39d8a6c46
# ╠═98cffce9-5036-4fea-88e8-fe82463c1e47
# ╠═17968306-6e98-453d-b087-618efedd3e55
# ╠═53ad2d23-2e07-47b1-b7f7-394c69218929
# ╠═c1bb1b4c-fc8e-4371-866c-40bbb6c7746f
# ╠═5d72579f-160c-420f-bc37-6405b37512e3
# ╠═3e3071b2-8daa-4168-9c21-043e792e4e2c
# ╠═0d50b122-7614-4acf-8294-8f02c660dc25
# ╠═3011b8b9-bb32-4c4b-a718-59ca147b0e7f
# ╠═c4b2eea4-c984-4591-8e62-841ef654a677
# ╠═2c8406e2-2928-4d14-a669-3c01ae46f111
# ╠═422746a8-4131-4463-b466-9118fcaa7587
# ╠═f32ca1c1-30d6-40b4-8f7d-4290fed932de
# ╠═0d7b668c-6df9-4372-895b-17ea13f2e829
# ╠═580aed47-941e-4ef7-bdbe-2fbf5332dcd8
# ╠═cffa1013-ee3a-48c6-a395-301ffada1109
# ╠═e91d1f09-9633-46f8-9734-f0aa83312507
