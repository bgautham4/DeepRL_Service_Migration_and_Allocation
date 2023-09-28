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

# ╔═╡ 04fe6e61-bd64-4204-9104-76e39d8a6c46
begin
	const GRID_SIZE = 5 # in km
	const SLOT_DUR = 1 # in minutes
	const RSU_R = 0.5 # coverage radius of RSU in km 
end

# ╔═╡ 98cffce9-5036-4fea-88e8-fe82463c1e47
begin

	mutable struct Path
		path_no::Integer
		start::Tuple{<:Real,<:Real}
		fin::Tuple{<:Real,<:Real}
		n_vehs::Integer
	end

	mutable struct Course
		path::Path
		next_seg::Union{Course,Nothing}
	end

	mutable struct Vehicle
		veh_id::Integer
		path_list::Union{Course,Nothing}
		position::Tuple{<:Real,<:Real}
		velocity::Real
		rsu_id::Integer
		allocated_BRB::Integer
	end
	
	mutable struct RSU
		rsu_id::Integer
		path::Path
		position::Tuple{<:Real,<:Real}
		vehicles::Vector{Vehicle}
		vaccant_BRB::Integer
	end
	
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
end

# ╔═╡ 54678886-aaa7-489d-b497-2f175457b103
r_along_line(0,0,1,1,0,0,1)

# ╔═╡ c1bb1b4c-fc8e-4371-866c-40bbb6c7746f
begin
	# Functions for running simulation
	function incr_vehicle_count!(path::Path)
		path.n_vehs += 1 
	end

	function decr_vehicle_count!(path::Path)
		path.n_vehs -= 1
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
	
	function update_vehicle_position!(vehicle::Vehicle,d::Real)
		x_prev,y_prev = vehicle.position
		x_i,y_i = vehicle.path_list.path.start
		x_e,y_e = vehicle.path_list.path.fin
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
		x_i,y_i = current_path.start
		x_e,y_e = current_path.fin
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
		decr_vehicle_count!(current_path)
		
		if next_path_list == nothing
			return
		end

		next_path::Path = next_path_list.path
		vehicle.position = (x_e,y_e)
		incr_vehicle_count!(next_path)
		update_vehicle_position!(vehicle,r)
		#update_vehicle_path!(vehicle) #Call function again to verify that vehicle has not left the new path aswell.
	end
		
	function associate_vehicle_rsu!(vehicle::Vehicle,rsu::RSU)
		vehicle.rsu_id = rsu.rsu_id
		associated_vehicles::Vector{Vehicle} = rsu.vehicles
		push!(associated_vehicles,vehicle)
	end

	function dissociate_vehicle_rsu!(vehicle::Vehicle,rsu::RSU)
		allocated_BRB = vehicle.allocated_BRB
		associated_vehicles::Vector{Vehicle} = rsu.vehicles
		indx = findfirst(x->x==vehicle,associated_vehicles)
		popat!(associated_vehicles,indx)
		rsu.vaccant_BRB += allocated_BRB
	end

	function tslot_aggr_rsu(rsu::RSU) :: Dict{Int,Vector{Vehicle}} #Function to aggregate the users based on time slots left under RSU
		x_ri,y_ri = rsu.position
		x_i,y_i = rsu.path.start
		x_e,y_e = rsu.path.fin
		x_re,y_re = r_along_line(x_ri,y_ri,x_i,y_i,x_e,y_e,RSU_R)
		aggr_dict = Dict{Int,Vector{Vehicle}}()
		for vehicle in rsu.vehicles
			v = vehicle.velocity
			x,y = vehicle.position
			d = euclidian_distance(x,y,x_re,y_re)
			t = d/v
			t_slots = 1 + ceil(Int,60*t/SLOT_DUR)
			haskey(aggr_dict,t_slots) ? push!(aggr_dict[t_slots],vehicle) : aggr_dict[t_slots] = [vehicle,]
		end
		aggr_dict
	end
		
	function remove_vehicles!(vehicles::Vector{Vehicle},rsus::Vector{RSU})
		to_delete = map(vehicles) do vehicle
			vehicle.path_list==nothing
		end
		#First dissociate all to be removed vehicles from their RSUs
		map(vehicles[to_delete]) do vehicle
			dissociate_vehicle_rsu!(vehicle,rsus[vehicle.rsu_id])
		end
		#Remove vehicles from simulation
		deleteat!(vehicles,to_delete)
	end
end

# ╔═╡ a957d99b-fc8f-414c-abca-568a48ef6786
begin
	path = Path(1, (0.0, 1), (5,1), 0)
	rsu_0 = RSU(0,path,(-1,-1),Vector{Vehicle}[],50)
	rsu1 = RSU(1,path,(0.5,1),Vector{Vehicle}[],50)
	rsu2 = RSU(2,path,(1.5,1),Vector{Vehicle}[],50)
	rsu3 = RSU(3,path,(2.5,1),Vector{Vehicle}[],50)
	rsu4 = RSU(4,path,(3.5,1),Vector{Vehicle}[],50)
	rsu5 = RSU(5,path,(4.5,1),Vector{Vehicle}[],50)
	rsus = Vector{RSU}(undef,5)
	for i in 1:5
		rsus[i] = eval(Symbol("rsu",i))
	end
	rsus
	course = Course(path,nothing)
	velocities = [10,15,20,25,30] 
	vehicles = Vector{Vehicle}(undef,10)
	for i in 1:10
		v = rand(velocities)
		vehicle = Vehicle(i,course,course.path.start,v,1,0)
		vehicles[i] = vehicle
		associate_vehicle_rsu!(vehicle,rsu1)
	end
	vehicles
end

# ╔═╡ 8c9c6615-8c80-4f25-b095-506fb2fa9acc
rsus

# ╔═╡ bc84f8a6-526a-4be9-9d1a-ef202b3e6700
begin
	"""plt = plot(legend=false, xlim=(0, GRID_SIZE), ylim=(0, GRID_SIZE), aspect_ratio=:equal)
	plot!(plt,[path.start[1], path.fin[1]], [path.start[2], path.fin[2]], color=:blue, lw=2)"""
	
	map(vehicles) do vehicle
		update_vehicle_position!(vehicle,vehicle.velocity*(SLOT_DUR/60))
		update_vehicle_path!(vehicle)
		if vehicle.path_list != nothing
			indx = findfirst(rsu->in_coverage(vehicle,rsu),rsus)
			old_rsu_id = vehicle.rsu_id
			println("$(old_rsu_id),$(indx),coords = $(vehicle.position)")
			if old_rsu_id != indx
				dissociate_vehicle_rsu!(vehicle,rsus[old_rsu_id])
				associate_vehicle_rsu!(vehicle,rsus[indx])
			end
		end
		
	end
	remove_vehicles!(vehicles,rsus)
	
	data = map(rsus) do rsu
		tslot_aggr_rsu(rsu)
	end

	plt = bar()
	
	t_slots = collect(keys(data[1]))
	counts = map(values(data[1])) do v
			length(v)
		end
	bar!(plt,t_slots,counts,bar_width = 1)
	plt
	
end

# ╔═╡ Cell order:
# ╠═161af454-2de6-11ee-05de-c9432d420a24
# ╠═235616e2-3377-4a03-bd2e-86514d73b262
# ╠═04fe6e61-bd64-4204-9104-76e39d8a6c46
# ╠═98cffce9-5036-4fea-88e8-fe82463c1e47
# ╠═53ad2d23-2e07-47b1-b7f7-394c69218929
# ╠═54678886-aaa7-489d-b497-2f175457b103
# ╠═c1bb1b4c-fc8e-4371-866c-40bbb6c7746f
# ╠═a957d99b-fc8f-414c-abca-568a48ef6786
# ╠═8c9c6615-8c80-4f25-b095-506fb2fa9acc
# ╠═bc84f8a6-526a-4be9-9d1a-ef202b3e6700
