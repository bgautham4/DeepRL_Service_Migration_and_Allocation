### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ f89a11ba-8382-11ee-36db-6b03dfe5d226
let
	using Pkg
	Pkg.activate(".")
end

# ╔═╡ 66d1583e-a6f1-49ae-9abe-6cce7de78cc1
using Distributions, StatsBase, Plots, JSON

# ╔═╡ 9c93f8d0-06ad-4084-bfcd-64ea95208c11
begin

        struct Service
			service_indx::Int
			data_size::Int #In MB
			crb_needed::Int
			thresh::Real
			max_thresh::Real
		end

        mutable struct Path
                start::Real
                fin::Real
                v_intr::Real
                eta::Real  
                #n_vehs::Integer
        end

        mutable struct Vehicle 
                veh_id::Integer
                service::Service
                position::Real
                velocity::Real
                rsu_id::Integer
                edge_id::Integer
        end

        mutable struct RSU
                rsu_id::Integer
                position::Real
                vehicles::Vector{Vehicle}
        end


        mutable struct EdgeServer
                edge_id::Integer
                vehicles::Vector{Vehicle}
                buff::Vector{Vehicle}
        end

end

# ╔═╡ a131b99c-b100-44df-a80e-0660f1205d01
begin
		const service1 = Service(
			1,
			40,
			2,
			0.2,
			0.35
		)

		const service2 = Service(
			2,
			40,
			2,
			0.2,
			0.35
		)
		const service3 = Service(
			3,
			40,
			2,
			0.2,
			0.35
		)
        const SERVICE_SET = (service1, service2, service3)
        #const DWELL_TIME_SET = (2,4,6,8,10,12,14) #Quantize vehicles based on remaining time
        #const MAX_TSLOTS = 5
end

# ╔═╡ 137d6874-12f4-4366-8095-60ecbe30ee39
begin
        #Basic utility functions
        function distance(p1,p2)
                abs(p1-p2)
        end
end

# ╔═╡ b0075d3a-99ae-4ed6-95ed-81a1e7184754
begin
	const SLOT_DUR = 1.5 #in mins
	const RSU_R = 0.5 #In km
	const MAX_STEPS =  floor(Int, 60 / SLOT_DUR) 
	const V_DISTR = Poisson(7)
	const VEL_DISTR = truncated(Normal(32,4);lower = 24, upper = 40)
	const ALLOC_DISTR = Poisson(20)
	const MIG_DISTR = Poisson(5)
end

# ╔═╡ 4f956cb9-85f4-48df-8749-5d60ddd64257
begin
       
        function in_coverage(vehicle::Vehicle,rsu::RSU) :: Bool
                distance(rsu.position, vehicle.position) <= RSU_R
        end


        function update_vehicle_position!(vehicle::Vehicle)
                vehicle.position += vehicle.velocity * (SLOT_DUR / 60)
        end


        #Aggregate vehicles under a RSU based on the service being used by the vehicle.
        function app_aggr_rsu(rsu::RSU) :: Dict{Service, Vector{Vehicle}}
                veh_app_info = Dict()
                for service in SERVICE_SET
                        veh_app_info[service] = filter(vehicle -> vehicle.service == service, rsu.vehicles)
                end
                veh_app_info
        end

        #Aggregate all users in a server based on the service being used.
        function app_aggr_edge(edge::EdgeServer) :: Dict{Service, Vector{Vehicle}}
                veh_app_info = Dict()
                for service in SERVICE_SET
                        veh_app_info[service] = filter(vehicle -> vehicle.service == service, edge.vehicles)
                end
                veh_app_info
        end

        #Aggregate users based on their remain time under edge server.
        function rem_time(edge::EdgeServer, duration::Int) :: Vector{Vehicle}
            vehs = filter(edge.vehicles) do vehicle
				rem_time = (5 - vehicle.position) / (vehicle.velocity)
				rem_time = (rem_time * 60) / SLOT_DUR
				rem_time < duration
			end
			vehs
        end

        #Given a set of vehicles, find all vehicles that are under the coverage of RSU.
        function find_all_vehs!(rsu::RSU, vehicles::Vector{Vehicle})
                indx = findall(vehicle->in_coverage(vehicle,rsu), vehicles)
                rsu.vehicles = vehicles[indx]
                map(vehicles[indx]) do vehicle
                        vehicle.rsu_id = rsu.rsu_id
                end
                #rsu.vaccant_BRB = 50
        end

        #Given a set of vehicles, find all vehicles that have their service hosted on the given edge server.
        function find_all_vehs_edge!(server::EdgeServer, vehicles::Vector{Vehicle})
                indx = findall(vehicle->vehicle.edge_id == 1, vehicles)
                server.vehicles = vehicles[indx]
        end

end

# ╔═╡ 1d5e4419-0970-4ff3-8554-c59c7972ca8e
begin
	mutable struct Env
		rsus::Vector{RSU}
		server::EdgeServer
		vehicles::Vector{Vehicle}
		step_no::Integer
	end

	function Env(;n_rsus = 5, n_vehs = 40)
		rsus = [RSU(i, 0.5+(i-1),[]) for i in 1:n_rsus]
		server = EdgeServer(1,[],[])
		vehicles = [Vehicle(i, rand(SERVICE_SET), rand(0:0.25:5),rand(VEL_DISTR),0, 1) for i in 1:n_vehs]
		map(rsus) do rsu
			find_all_vehs!(rsu,vehicles)
		end
		server.vehicles = copy(vehicles)
		Env(rsus,server,vehicles,0)
	end
end	

# ╔═╡ b8de13fa-eafd-444b-b4cf-4ff80df4605f
begin
	function gen_vehs()
		N_inject = rand(V_DISTR)
        new_vehs = [Vehicle(i,rand(SERVICE_SET),rand(0:0.125:1),rand(VEL_DISTR), 1, 1) for i in 1:N_inject]
	end

	function _update_env!(env::Env,new_vehs)
        #push!(env.server.vehicles, env.server.buff...) #Move previous migrations to be under edge server
        #env.server.buff = new_vehs
		#push!(env.server.vehicles, new_vehs)		
		#Update vehicle positions
	
        map(env.vehicles) do vehicle
            update_vehicle_position!(vehicle)
        end

        to_delete = map(vehicle->vehicle.position > 5, env.vehicles)
        deleteat!(env.vehicles, to_delete)
		push!(env.vehicles, new_vehs...)
        find_all_vehs_edge!(env.server, env.vehicles)
        map(env.rsus) do rsu
            find_all_vehs!(rsu,env.vehicles)
        end
        #env.n_vehs = length(env.vehicles)
        env.step_no += 1 #Step through time
	end

	function reset!(env::Env)
		#env = Env()
		n_rsus = length(env.rsus)
		n_vehs = 42
		rsus = [RSU(i, 0.5+(i-1),[]) for i in 1:n_rsus]
		server = EdgeServer(1,[],[])
		vehicles = [Vehicle(i, rand(SERVICE_SET), rand(0:0.25:5),rand(VEL_DISTR),0, 1) for i in 1:n_vehs]
		map(rsus) do rsu
			find_all_vehs!(rsu,vehicles)
		end
		server.vehicles = copy(vehicles)
		env.rsus = rsus
		env.server = server
		env.vehicles = vehicles
		env.step_no = 0
	end
end

# ╔═╡ 59d612f7-cb91-45a5-bd29-1a5c6f7ee46b
test_env = Env()

# ╔═╡ 24f565f3-695c-4012-9d4e-7f93075cf217
# ╠═╡ disabled = true
#=╠═╡
begin
	reset!(test_env)
	
	info_dict = Dict{Any,Vector{Vector{Int}}}()
	info_dict["migs"] = []
	info_dict["allocs"] = []
	info_dict["rem_time"] = []
	for (rsu_indx,_) in enumerate(test_env.rsus)
		for (app_indx,_) in enumerate(SERVICE_SET)
			info_dict[(rsu_indx,app_indx)] = [Vector{Int}() for _ in 1:100]
		end
	end
	for ep in 1:100 #100 episodes
	T_SLOTS = 40
	
	migs = []
	allocs = []
	r_time = []
	for t_slot in 1:T_SLOTS
		for (rsu_indx,rsu) in enumerate(test_env.rsus)
			rsu_vehs = app_aggr_rsu(rsu)
			vehs = []
			for (app_indx,app) in enumerate(SERVICE_SET)
				"""
				if haskey(info_dict,(rsu_indx,app_indx))
					push!(info_dict[(rsu_indx,app_indx)],length(rsu_vehs[app]))
				else
					info_dict[(rsu_indx,app_indx)] = [length(rsu_vehs[app]),]
				end
				"""
				push!(info_dict[(rsu_indx,app_indx)][ep], length(rsu_vehs[app]))
			end
		end
		new_vehs = gen_vehs()
		W = rem_time(test_env.server,1) |> length
		"""
		push!(info_dict["migs"],length(new_vehs))
		push!(info_dict["allocs"],rand(ALLOC_DISTR))
		push!(info_dict["rem_time"],W)
		"""
		push!(migs,length(new_vehs))
		push!(allocs,rand(ALLOC_DISTR))
		push!(r_time,W)
		_update_env!(test_env,new_vehs)
	end
	push!(info_dict["migs"],migs)
	push!(info_dict["allocs"],allocs)
	push!(info_dict["rem_time"],r_time)
	end
end
  ╠═╡ =#

# ╔═╡ 1a6eea70-0a32-4626-a38b-e1fd003597b3
begin
	reset!(test_env)
	
	info_dict = Dict{Any,Vector{Int}}()
	info_dict["migs"] = []
	info_dict["allocs"] = []
	info_dict["rem_time"] = []
	T_SLOTS = 40
	for t_slot in 1:T_SLOTS
		for (rsu_indx,rsu) in enumerate(test_env.rsus)
			rsu_vehs = app_aggr_rsu(rsu)
			for (app_indx,app) in enumerate(SERVICE_SET)
				if haskey(info_dict,(rsu_indx,app_indx))
					push!(info_dict[(rsu_indx,app_indx)],length(rsu_vehs[app]))
				else
					info_dict[(rsu_indx,app_indx)] = [length(rsu_vehs[app]),]
				end
			end
		end
		new_vehs = gen_vehs()
		W = rem_time(test_env.server,2) |> length
		push!(info_dict["migs"],rand(MIG_DISTR))
		push!(info_dict["allocs"],rand(ALLOC_DISTR))
		push!(info_dict["rem_time"],W)
		
		_update_env!(test_env,new_vehs)
	end
end

# ╔═╡ 6bb3bb6c-3894-4fc8-a3cf-7f59b7bf8408
info_dict

# ╔═╡ a13e33e3-93b6-440c-90de-2c70695b8b86
json_str = JSON.json(info_dict)

# ╔═╡ fd61cad0-ec73-4d69-b02f-40d1c3fbca33
# ╠═╡ disabled = true
#=╠═╡
open("../time_series/base_series_7_5_20_2.json","w") do file
	write(file,json_str)
end
  ╠═╡ =#

# ╔═╡ bf7144e0-4757-48a9-b32c-5d72ca79d935
# ╠═╡ disabled = true
#=╠═╡
noised_series = open("../time_series/noised_series_5_5_15_1_train.json") do file
	json_str = read(file,String)
	JSON.parse(json_str)
end
  ╠═╡ =#

# ╔═╡ 38d1dbeb-5368-491e-b6d3-ee3dca3578e7
# ╠═╡ disabled = true
#=╠═╡
x = sum((noised_series["(5, $(i))"] for i in 1:3)) .- noised_series["rem_time"]
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═f89a11ba-8382-11ee-36db-6b03dfe5d226
# ╠═66d1583e-a6f1-49ae-9abe-6cce7de78cc1
# ╠═9c93f8d0-06ad-4084-bfcd-64ea95208c11
# ╠═a131b99c-b100-44df-a80e-0660f1205d01
# ╠═137d6874-12f4-4366-8095-60ecbe30ee39
# ╠═4f956cb9-85f4-48df-8749-5d60ddd64257
# ╠═b0075d3a-99ae-4ed6-95ed-81a1e7184754
# ╠═1d5e4419-0970-4ff3-8554-c59c7972ca8e
# ╠═b8de13fa-eafd-444b-b4cf-4ff80df4605f
# ╠═59d612f7-cb91-45a5-bd29-1a5c6f7ee46b
# ╠═24f565f3-695c-4012-9d4e-7f93075cf217
# ╠═1a6eea70-0a32-4626-a38b-e1fd003597b3
# ╠═6bb3bb6c-3894-4fc8-a3cf-7f59b7bf8408
# ╠═a13e33e3-93b6-440c-90de-2c70695b8b86
# ╠═fd61cad0-ec73-4d69-b02f-40d1c3fbca33
# ╠═bf7144e0-4757-48a9-b32c-5d72ca79d935
# ╠═38d1dbeb-5368-491e-b6d3-ee3dca3578e7
