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
using ReinforcementLearning

# ╔═╡ 04fe6e61-bd64-4204-9104-76e39d8a6c46
begin
	const GRID_SIZE = 5 # in km
	const SLOT_DUR = 1 # in minutes
	const RSU_R = 0.5 # coverage radius of RSU in km 
end

# ╔═╡ 98cffce9-5036-4fea-88e8-fe82463c1e47
begin
	
	@enum Service service1 = 1 service2 = 2

	mutable struct Path
		path_no::Integer
		start::Tuple{<:Real,<:Real}
		fin::Tuple{<:Real,<:Real}
		n_vehs::Integer
		n_vehs_rev::Integer
	end

	mutable struct Course
		path::Path
		rev_flag::Bool
		next_seg::Union{Course,Nothing}
	end

	mutable struct Vehicle
		veh_id::Integer
		service::Service
		path_list::Union{Course,Nothing}
		position::Tuple{<:Real,<:Real}
		velocity::Real
		rsu_id::Integer
		prev_tslot_rsu_id::Integer
		edge_id::Integer
		#old_edge_id
		allocated_BRB::Integer
		allocated_CRB::Integer
	end
	
	mutable struct RSU
		rsu_id::Integer
		path::Path
		position::Tuple{<:Real,<:Real}
		vehicles::Vector{Vehicle}
		vaccant_BRB::Integer
	end


	mutable struct EdgeServer
		edge_id::Integer
		path::Path
		rsus::Vector{RSU}
		vehicles::Vector{Vehicle}
		vaccant_CRB::Integer
		nbr_mig_prev_slot::Integer
		nbr_alloc_prev_slot::Integer
	end

	function RSU(id::Integer, path::Path, position::Tuple{<:Real,<:Real})
		RSU(id,path,position,[],50)
	end
end

# ╔═╡ 17968306-6e98-453d-b087-618efedd3e55
begin
	const SERVICE_SET = (service1,service2)
	const MAX_TSLOTS = 5
end

# ╔═╡ 53ad2d23-2e07-47b1-b7f7-394c69218929
begin
	#Basic utility functions
	function euclidian_distance(x1,y1,x2,y2)
		sqrt((x1-x2)^2 + (y1-y2)^2)
	end

	function r_along_line(x0,y0,x1,y1,x2,y2,r) # Finds the coordinates x,y by travelling a distance of r along the line segement (x1,y1),(x2,y2) when initial position was x0,y0
		m = abs((y2 - y1)/(x2 - x1))
		theta = atan(m)
		x_dir = -1*(x2<x1) + 1*(x2>x1)
		y_dir = -1*(y2<y1) + 1*(y2>y1)
		x = x0 + x_dir*r*cos(theta)
		y = y0 + y_dir*r*sin(theta)
		(x,y)
	end

	function ls_hop_count(conn_status::Matrix{Bool})
		M,N = size(conn_status)
		@assert M==N
		hop_cnt = map(conn_status) do x
			x==1 ? 1 : 50
		end
		for node in 1:N
			N_pr = [node,]
			node_table = view(hop_cnt,node,:)
			while length(N_pr) != N
				#Find w such that it has min cost and is not in N_pr
				w = 0
				cost = 50
				for (i,j) in enumerate(node_table)
					if i in N_pr
						continue
					end
					j < cost ? (cost = j ; w = i) : nothing
				end
				push!(N_pr,w)
				w_table = view(hop_cnt,w,:)
				for v in 1:N
					if v in N_pr
						continue
					end
					node_table[v] = min(node_table[v], node_table[w] + w_table[v])
				end
			end
		end
	hop_cnt			
	end
end

# ╔═╡ 8d2b20e3-f2d3-4877-a7b9-371c9b6ebcef
ls_hop_count([true false true;false true true;true true true])

# ╔═╡ 54678886-aaa7-489d-b497-2f175457b103
r_along_line(0,0,1,1,0,0,1)

# ╔═╡ c1bb1b4c-fc8e-4371-866c-40bbb6c7746f
begin
	# Functions for running simulation
	function incr_vehicle_count!(path::Path,rev_flag::Bool)
		rev_flag ? path.n_vehs_rev+=1 : path.n_vehs += 1 
	end

	function decr_vehicle_count!(path::Path,rev_flag::Bool)
		rev_flag ? path.n_vehs_rev+=1 : path.n_vehs -= 1
	end

	function in_coverage(vehicle::Vehicle,rsu::RSU) :: Bool
		vehicle_path::Path = vehicle.path_list.path
		rsus_path::Path = rsu.path
		
		if vehicle_path != rsus_path
			return false
		end
		
		x_rsu,y_rsu = rsu.position
		x_veh,y_veh = vehicle.position
		
		if (x_veh-x_rsu)^2 + (y_veh-y_rsu)^2  <= RSU_R ^2
			return true
		end
		
		return false	
	end

	function set_direction(path::Path,rev_flag::Bool)
		if rev_flag
			x_i,y_i = path.fin
			x_e,y_e = path.start
		else
			x_i,y_i = path.start
			x_e,y_e = path.fin
		end
		(x_i,x_e,y_i,y_e)
	end
	
	function update_vehicle_position!(vehicle::Vehicle,d::Real)
		x_prev,y_prev = vehicle.position
		path::Path = vehicle.path_list.path
		rev_flag::Bool = vehicle.path_list.rev_flag
		x_i,x_e,y_i,y_e = set_direction(path,rev_flag)
		#v = vehicle.velocity
		"""
		m = abs((y_e - y_i)/(x_e - x_i))
		theta = atan(m)
		x_dir = -1*(x_e<x_i) + 1*(x_e>x_i)
		y_dir = -1*(y_e<y_i) + 1*(y_e>y_i)
		x = x_prev + x_dir*d*cos(theta)
		y = y_prev + y_dir*d*sin(theta)
		"""
		vehicle.position = r_along_line(x_prev,y_prev,x_i,y_i,x_e,y_e,d)
	end

	function update_vehicle_path!(vehicle::Vehicle)
		x,y = vehicle.position
		current_path::Path = vehicle.path_list.path
		rev_flag::Bool = vehicle.path_list.rev_flag 
		x_i,x_e,y_i,y_e = set_direction(current_path,rev_flag)
		x_dir = (x_e - x_i)
		y_dir = (y_e - y_i)
		x_r = (x_e-x)*(x_dir<0 && x<x_e) + (x-x_e)*(x_dir>0 && x>x_e)  #Residual distance in x
		y_r = (y_e-y)*(y_dir<0 && y<y_e) + (y-y_e)*(y_dir>0 && y>y_e)  #Residual distance in y
		r = sqrt(x_r^2 + y_r^2)
		
		if r==0
			return
		end

		next_path_list::Union{Course,Nothing} = vehicle.path_list.next_seg
		vehicle.path_list = next_path_list
		decr_vehicle_count!(current_path,rev_flag)
		
		if next_path_list == nothing
			return
		end

		next_path::Path = next_path_list.path
		next_rev_flag = next_path_list.rev_flag
		vehicle.position = (x_e,y_e)
		incr_vehicle_count!(next_path,next_rev_flag)
		update_vehicle_position!(vehicle,r)
		#update_vehicle_path!(vehicle) #Call function again to verify that vehicle has not left the new path aswell.
	end
		
	function associate_vehicle_rsu!(vehicle::Vehicle,rsu::RSU)
		vehicle.rsu_id = rsu.rsu_id
		#associated_vehicles::Vector{Vehicle} = rsu.vehicles
		push!(rsu.vehicles,vehicle)
	end

	function dissociate_vehicle_rsu!(vehicle::Vehicle,rsu::RSU)
		allocated_BRB = vehicle.allocated_BRB
		associated_vehicles::Vector{Vehicle} = rsu.vehicles
		indx = findfirst(x->x==vehicle,associated_vehicles)
		popat!(associated_vehicles,indx)
		rsu.vaccant_BRB += allocated_BRB
	end

	function tslot_aggr_rsu(rsu::RSU, vehicles::Vector{Vehicle})  #Function to aggregate the users based on time slots left under RSU 
		eps = 0.0001
		x_ri,y_ri = rsu.position
		x_i,y_i = rsu.path.start
		x_e,y_e = rsu.path.fin
		x_re,y_re = r_along_line(x_ri,y_ri,x_i,y_i,x_e,y_e,RSU_R)
		aggr_tslot = [Vector{Vehicle}() for _ in 1:MAX_TSLOTS]	
		for vehicle in vehicles
			v = vehicle.velocity
			x,y = vehicle.position
			d = euclidian_distance(x,y,x_re,y_re)
			t = d/v
			t_slots = ceil(Int,60*t/SLOT_DUR + eps)
			t_slots < 5 ? push!(aggr_tslot[t_slots],vehicle) : push!(aggr_tslot[5],vehicle)
		end
		aggr_tslot
	end

	function app_aggr_rsu(rsu::RSU)
		app_aggr = Vector{Vector{Vector{Vehicle}}}(undef,length(SERVICE_SET))
		for (i,service) in enumerate(SERVICE_SET)
			subset = filter(vehicle -> vehicle.service == service, rsu.vehicles)
			app_aggr[i] = tslot_aggr_rsu(rsu,subset)
		end
		app_aggr
	end

	function associate_vehicle_server!(vehicle::Vehicle, server::EdgeServer)
		vehicle.edge_id = server.edge_id
		push!(server.vehicles,vehicle)
	end

	function dissociate_vehicle_server!(vehicle::Vehicle, server::EdgeServer)
		allocated_CRB = vehicle.allocated_CRB
		indx = findfirst(x->x==vehicle,server.vehicles)
		popat!(server.vehicles,indx)
		server.vaccant_CRB += allocated_CRB
	end

	function app_aggr_server(server::EdgeServer)
		app_aggr = Vector{Vector{Vehicle}}(undef,length(SERVICE_SET))
		for (i,service) in enumerate(SERVICE_SET)
			veh_service = filter(vehicle -> vehicle.service == service, server.vehicles)
			app_aggr[i] = veh_service
		end
		app_aggr
	end

	function hop_cnt(vehicle::Vehicle, hop_cnt_matrix)
		vehicle_path_no = vehicle.path_list.path.path_no
		vehicle_edge_no = vehicle.edge_id
		hop_cnt_matrix[vehicle_path_no,vehicle_edge_no]
	end
	
	function remove_vehicles!(vehicles::Vector{Vehicle}, rsus::Vector{RSU}, servers::Vector{EdgeServer})
		to_delete = map(vehicles) do vehicle
			vehicle.path_list==nothing
		end
		#First dissociate all to be removed vehicles from their RSUs
		map(vehicles[to_delete]) do vehicle
			dissociate_vehicle_rsu!(vehicle,rsus[vehicle.rsu_id])
		end

		#Then dissociate all to be removed vehicles from their edge servers
		map(vehicles[to_delete]) do vehicle
			dissociate_vehicle_server!(vehicle,servers[vehicle.edge_id])
		end
		#Remove vehicles from simulation
		deleteat!(vehicles,to_delete)
	end
end

# ╔═╡ a957d99b-fc8f-414c-abca-568a48ef6786
begin
	path = Path(1, (0.0, 1), (5,1), 0,0)
	path2 = Path(2,(5,1),(10,1),0,0)
	rsu0 = RSU(0,path,(-10,-10))
	rsu1 = RSU(1,path,(0.5,1))
	rsu2 = RSU(2,path,(1.5,1))
	rsu3 = RSU(3,path,(2.5,1))
	rsu4 = RSU(4,path,(3.5,1))
	rsu5 = RSU(5,path,(4.5,1))
	rsu6 = RSU(6,path2,(5.5,1))
	rsu7 = RSU(7,path2,(6.5,1))
	rsu8 = RSU(8,path2,(7.5,1))
	rsu9 = RSU(9,path2,(8.5,1))
	rsu10 = RSU(10,path2,(9.5,1))
	rsus = Vector{RSU}(undef,10)
	rsus1 = Vector{RSU}(undef,5)
	rsus2 = Vector{RSU}(undef,5)
	for i in 1:10
		rsus[i] = eval(Symbol("rsu",i))
		i<=5 ? rsus1[i] = eval(Symbol("rsu",i)) : rsus2[i-5] = eval(Symbol("rsu",i))
	end
	server1 = EdgeServer(1,path,rsus1,[],0,0,0)
	server2 = EdgeServer(2,path2,rsus2,[],0,0,0)
	servers = [server1,server2]
	course = Course(path,false,nothing)
	velocities = [10,15,20,25,30] 
	vehicles = Vector{Vehicle}(undef,10)
	for i in 1:10
		v = rand(velocities)
		s = rand(SERVICE_SET)
		vehicle = Vehicle(i,s,course,course.path.start,v,1,0,1,0,0)
		vehicles[i] = vehicle
		associate_vehicle_rsu!(vehicle,rsu1)
		associate_vehicle_server!(vehicle,server1)
	end
end

# ╔═╡ 3e3071b2-8daa-4168-9c21-043e792e4e2c
Integer(service1)

# ╔═╡ 3011b8b9-bb32-4c4b-a718-59ca147b0e7f
begin

	mutable struct ServiceMigrationEnv <: RL.AbstractEnv
		paths::Vector{Path}
		rsus::Vector{RSU}
		server::EdgeServer
		#hop_counts::Matrix{T} where T<:Integer
		vehicles::Vector{Vehicle}
		#continuous::Bool
		step_no::Integer
		max_steps::Integer
	end

	function RLBase.reset!(env::ServiceMigrationEnv)
		nothing
	end
	
	RLBase.is_terminated(env::ServiceMigrationEnv) = env.step_no >= env.max_steps ? true : false

 	function RLBase.state(env::ServiceMigrationEnv)
		vcat(
			[length(service_tslot) for rsu in env.server.rsus for service in app_aggr_rsu(rsu) for service_tslot in service],
			[length(service) for service in app_aggr_server(env.server)],
			length(filter(vehicle -> (vehicle.path_list != nothing) && (vehicle.path_list.path != env.server.path), env.server.vehicles)),
			env.server.nbr_mig_prev_slot,
			env.server.nbr_alloc_prev_slot
			)	
	end

	function RLBase.state_space(env::ServiceMigrationEnv)
		Space(
				vcat(
					[0:50 for _ in env.server.rsus for _ in SERVICE_SET for _ in 1:MAX_TSLOTS],
					[0:50 for _ in SERVICE_SET],
					[0:50],
					[0:100],
					[0:100]
					)
			)
	end

	function RLBase.action_space(env::ServiceMigrationEnv)
		Space(

			vcat(
				[Base.OneTo(5) for _ in env.server.rsus for _ in SERVICE_SET for _ in 1:MAX_TSLOTS],
				[Base.OneTo(100÷length(SERVICE_SET)) for _ in SERVICE_SET],
				[0:10],
				#length(filter(vehicle -> vehicle.path_list.path != env.server.path, env.server.vehicles))],
				[0:50],
				[0:40]
				)
			
			)
			
		
	end

	function _allocate_BRB!(env::ServiceMigrationEnv, action) #Given the action vector, assign the BRB to each vehicle under the server's RSUs
		for (rsu_index,rsu) in enumerate(env.server.rsus)
			allocs = action[(rsu_index-1)*length(SERVICE_SET)*MAX_TSLOTS + 1: (rsu_index-1)*length(SERVICE_SET)*MAX_TSLOTS + length(SERVICE_SET)*MAX_TSLOTS]
			#println(allocs)
			#println(action)
			vehicle_groups = app_aggr_rsu(rsu)
			for (service_indx,_) in enumerate(SERVICE_SET)
				for time_index in 1:MAX_TSLOTS
					vehicle_group = vehicle_groups[service_indx][time_index]
					map(vehicle_group) do vehicle
						vehicle.allocated_BRB = allocs[(service_indx-1)*MAX_TSLOTS + time_index] ÷ length(vehicle_group)
						rsu.vaccant_BRB -= allocs[(service_indx-1)*MAX_TSLOTS + time_index] ÷ length(vehicle_group)
					end
				end
			end
		end
	end

	function _allocate_CRB!(env::ServiceMigrationEnv,action) #Given the action vector, assign the BRB to each vehicle under the server
		J = length(env.server.rsus)
		allocs_CRB = action[J*length(SERVICE_SET)*MAX_TSLOTS + 1: J*length(SERVICE_SET)*MAX_TSLOTS + length(SERVICE_SET)]
		vehicle_app_groups = app_aggr_server(env.server)
		for (indx,vehicle_app_group) in enumerate(vehicle_app_groups)
			map(vehicle_app_group) do vehicle
				vehicle.allocated_CRB = allocs_CRB[indx] ÷ length(vehicle_app_group)
				env.server.vaccant_CRB -= allocs_CRB[indx] ÷ length(vehicle_app_group)
				end
		end
	end

	function inject_vehicles!()
		nothing
	end

	function update_path_velocity!()
		nothing
	end

	function _step!(env::ServiceMigrationEnv)
		inject_vehicles!() #Define functionality later
		update_path_velocity!() #Define later
		
		to_delete = map(env.vehicles) do vehicle
			update_vehicle_position!(vehicle,vehicle.velocity*(SLOT_DUR/60))
			update_vehicle_path!(vehicle)
			if vehicle.path_list != nothing
				indx = findfirst(rsu->in_coverage(vehicle,rsu),rsus)
				prev_tslot_rsu_id = vehicle.rsu_id
				vehicle.prev_tslot_rsu_id = prev_tslot_rsu_id
				if indx != nothing && prev_tslot_rsu_id != indx
					prev_tslot_rsu_id !=0 ? dissociate_vehicle_rsu!(vehicle,rsus[prev_tslot_rsu_id]) : nothing
					associate_vehicle_rsu!(vehicle,rsus[indx])
				end
				false #This vehicle must not be removed 
			else
				vehicle.rsu_id != 0 ? dissociate_vehicle_rsu!(vehicle,env.rsus[vehicle.rsu_id]) : nothing
				vehicle.edge_id == 2 ? dissociate_vehicle_server!(vehicle,env.server) : nothing
				true #This vehicle must be removed before the next slot
			end
		end

		deleteat!(env.vehicles,to_delete)
		env.step_no += 1
			
	end

	
	function(env::ServiceMigrationEnv)(action)
		@assert action in RLBase.action_space(env)
		_allocate_BRB!(env,action)
		_allocate_CRB!(env,action)
		_step!(env)
	end


	function RLBase.reward(env::ServiceMigrationEnv)
		nothing
	end
	
	
end

# ╔═╡ c4b2eea4-c984-4591-8e62-841ef654a677
begin
	test_env = ServiceMigrationEnv(
				[path,path2],
				[rsu1,rsu2,rsu3,rsu4,rsu5],
				server1,
				vehicles,
				0,
				10
				)
	#state_space(test_env)
end

# ╔═╡ 5946c928-6cd4-42a3-abc8-8cf3fc853b51
begin
	action_space(test_env) |> rand |> test_env
	state(test_env),test_env.vehicles
end

# ╔═╡ 30ebfca9-6312-428b-8347-c604b433fce1
isempty([])

# ╔═╡ f6a43370-9a75-4fa8-ab85-c9509654b4b8
app_aggr_rsu(rsu1)

# ╔═╡ bb6fbe72-6716-494a-bd3b-fa02c969f422
ActionTransformedEnv

# ╔═╡ Cell order:
# ╠═161af454-2de6-11ee-05de-c9432d420a24
# ╠═235616e2-3377-4a03-bd2e-86514d73b262
# ╠═04fe6e61-bd64-4204-9104-76e39d8a6c46
# ╠═98cffce9-5036-4fea-88e8-fe82463c1e47
# ╠═17968306-6e98-453d-b087-618efedd3e55
# ╠═53ad2d23-2e07-47b1-b7f7-394c69218929
# ╠═8d2b20e3-f2d3-4877-a7b9-371c9b6ebcef
# ╠═54678886-aaa7-489d-b497-2f175457b103
# ╠═c1bb1b4c-fc8e-4371-866c-40bbb6c7746f
# ╠═a957d99b-fc8f-414c-abca-568a48ef6786
# ╠═3e3071b2-8daa-4168-9c21-043e792e4e2c
# ╠═bc84f8a6-526a-4be9-9d1a-ef202b3e6700
# ╠═0d50b122-7614-4acf-8294-8f02c660dc25
# ╠═3011b8b9-bb32-4c4b-a718-59ca147b0e7f
# ╠═c4b2eea4-c984-4591-8e62-841ef654a677
# ╠═5946c928-6cd4-42a3-abc8-8cf3fc853b51
# ╠═30ebfca9-6312-428b-8347-c604b433fce1
# ╠═f6a43370-9a75-4fa8-ab85-c9509654b4b8
# ╠═bb6fbe72-6716-494a-bd3b-fa02c969f422
