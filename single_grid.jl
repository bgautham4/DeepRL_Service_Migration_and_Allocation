### A Pluto.jl notebook ###
# v0.19.26

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
	const GRID_SIZE = 2 # in km
	const SLOT_DUR = 1 # in minutes
	const RSU_R = 0.25 # coverage radius of RSU in km 
end

# ╔═╡ 98cffce9-5036-4fea-88e8-fe82463c1e47
begin
	
	mutable struct Grid
		grid_no::Integer
		n_vehs::Integer
	end

	mutable struct Path
		path_no::Integer
		grid::Grid
		start::Tuple{<:Real,<:Real}
		fin::Tuple{<:Real,<:Real}
	end

	mutable struct RSU
		rsu_id::Integer
		grid::Grid
		position::Tuple{<:Real,<:Real}
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
		associated_rsu::RSU
	end
end

# ╔═╡ c1bb1b4c-fc8e-4371-866c-40bbb6c7746f
begin
	
	function incr_vehicle_count!(grid::Grid)
		grid.n_vehs += 1 
	end

	function decr_vehicle_count!(grid::Grid)
		grid.n_vehs -= 1
	end

	function in_coverage(vehicle::Vehicle,rsu::RSU) :: Bool
		vehicle_grid::Grid = vehicle.path_list.path.grid
		rsu_grid::Grid = rsu.grid
		
		if vehicle_grid != rsu_grid
			return false
		end
		
		x_rsu,y_rsu = rsu.position
		x_veh,y_veh = vehicle.position
		
		if ((x_veh-x_rsu)^2 + (y_veh-y_rsu)^2) <= RSU_R
			return true
		end
		
		return false	
	end
	
	function update_vehicle_position!(vehicle::Vehicle,d::Real)
		x_prev,y_prev = vehicle.position
		x_i,y_i = vehicle.path_list.path.start
		x_e,y_e = vehicle.path_list.path.fin
		#v = vehicle.velocity
		m = (y_e - y_i)/(x_e - x_i)
		theta = m >= 0 ? atan(m) : π - atan(-m)
		x = x_prev + d*cos(theta)
		y = y_prev + d*sin(theta)
		vehicle.position = (x,y)
	end

	function update_vehicle_grid!(vehicle::Vehicle)
		x,y = vehicle.position
		x_r = (-x)*(x<0) + (x-GRID_SIZE)*(x>GRID_SIZE)  #Residual distance in x
		y_r = (-y)*(y<0) + (y-GRID_SIZE)*(y>GRID_SIZE)  #Residual distance in y
		r = sqrt(x_r^2 + y_r^2)
		
		if r==0
			return
		end

		current_path::Path = vehicle.path_list.path
		next_path_list::Union{Course,Nothing} = vehicle.path_list.next_seg
		vehicle.path_list = next_path_list
		decr_vehicle_count!(current_path.grid)
		
		if next_path_list == nothing
			return
		end

		next_path::Path = next_path_list.path
		vehicle.position = (x,y)
		incr_vehicle_count!(next_path.grid)
		update_vehicle_position!(vehicle,r)
	end
		
	function associate_vehicle_rsu!(vehicle::Vehicle,rsu::RSU)
		old_rsu::RSU = vehicle.associated_rsu
		old_rsu.n_vehs -= 1
		vehicle.associated_rsu = rsu
		rsu.n_vehs += 1
	end

	function remove_vehicles!(vehicles::Vector{Vehicle})
		to_delete = map(vehicles) do vehicle
			vehicle.path_list==nothing ? true : false
		end
		deleteat!(vehicles,to_delete)
	end
end

# ╔═╡ 3e51ae9e-1a43-4cc0-8184-a652e3675c98
begin
	grid1 = Grid(1,0)
	grid2 = Grid(2,0)
	path1 = Path(1,grid1,(0,1),(2,1))
	path2 = Path(1,grid1,(1,0),(1,2))
	path3 = Path(1,grid1,(2,0),(0,2))
	path4 = Path(1,grid1,(0,0),(2,2))
	
	course1 = Course(path1,nothing)
	course2 = Course(path2,nothing)
	course3 = Course(path3,nothing)
	course4 = Course(path4,nothing)
	rsu_0 = RSU(0,Grid(0,0),(-1,-1),0)
	vehicle1 = Vehicle(1,course1,course1.path.start,10,rsu_0)
	vehicle2 = Vehicle(2,course2,course2.path.start,10,rsu_0)
	vehicle3 = Vehicle(3,course3,course3.path.start,10,rsu_0)
	vehicle4 = Vehicle(4,course4,course4.path.start,10,rsu_0)
	vehicles = [vehicle1,vehicle2,vehicle3,vehicle4]
end

# ╔═╡ a02014d5-699e-455b-b9d1-855b22b2a1c0
begin
	p = scatter()
	x = range(0,2,50)
	p1 = map(x->x,x)
	p2 = map(x->-x+2,x)
	p3 = map(x->1,x)
	map(vehicles) do vehicle
		update_vehicle_position!(vehicle,vehicle.velocity*(SLOT_DUR/60))
		update_vehicle_grid!(vehicle)
	end
	remove_vehicles!(vehicles)
	[scatter!(p,[vehicle.position[1]],[vehicle.position[2]]) for vehicle in vehicles]
	scatter!(p,xlim = (0,2), ylim= (0,2))
	plot!(p,x,p1)
	plot!(p,x,p2)
	plot!(p,x,p3)
	plot!(p,p3,x)
end

# ╔═╡ Cell order:
# ╠═161af454-2de6-11ee-05de-c9432d420a24
# ╠═235616e2-3377-4a03-bd2e-86514d73b262
# ╠═04fe6e61-bd64-4204-9104-76e39d8a6c46
# ╠═98cffce9-5036-4fea-88e8-fe82463c1e47
# ╠═c1bb1b4c-fc8e-4371-866c-40bbb6c7746f
# ╠═3e51ae9e-1a43-4cc0-8184-a652e3675c98
# ╠═a02014d5-699e-455b-b9d1-855b22b2a1c0
