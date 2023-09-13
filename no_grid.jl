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

# ╔═╡ a7b0c4de-aa7d-4201-97a2-d2df8cfa902c
using DataFrames

# ╔═╡ d8ba55ec-e66c-44e7-ac63-d2948264acaf
using Pluto

# ╔═╡ 04fe6e61-bd64-4204-9104-76e39d8a6c46
begin
	const GRID_SIZE = 15 # in km
	const SLOT_DUR = 1 # in minutes
	const RSU_R = 0.25 # coverage radius of RSU in km 
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

# ╔═╡ c1bb1b4c-fc8e-4371-866c-40bbb6c7746f
begin
	
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
		m = abs((y_e - y_i)/(x_e - x_i))
		theta = atan(m)
		x_dir = -1*(x_e<x_i) + 1*(x_e>x_i)
		y_dir = -1*(y_e<y_i) + 1*(y_e>y_i)
		x = x_prev + x_dir*d*cos(theta)
		y = y_prev + y_dir*d*sin(theta)
		vehicle.position = (x,y)
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
		update_vehicle_path!(vehicle) #Call function again to verify that vehicle has not left the new path aswell.
	end
		
	function associate_vehicle_rsu!(vehicle::Vehicle,rsu::RSU)
		vehicle.associated_rsu_id = rsu.rsu_id
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
		
	function remove_vehicles!(vehicles::Vector{Vehicle})
		to_delete = map(vehicles) do vehicle
			vehicle.path_list==nothing
		end
		deleteat!(vehicles,to_delete)
	end
end

# ╔═╡ 3e51ae9e-1a43-4cc0-8184-a652e3675c98
# ╠═╡ disabled = true
#=╠═╡
begin
	path1 = Path(1,(0,1),(1,1),0)
	path2 = Path(2,(1,1),(2,1),0)
	path3 = Path(3,(1,0),(1,1),0)
	path4 = Path(4,(1,1),(1,2),0)
	path5 = Path(5,(2,2),(1,1),0)
	path6 = Path(6,(1,1),(0,2),0)
		
	course1 = Course(path1,Course(path2,nothing))
	course2 = Course(path1,Course(path4,nothing))
	course3 = Course(path3,Course(path2,nothing))
	course4 = Course(path3,Course(path4,nothing))
	course5 = Course(path5,Course(path6,nothing))
	
	rsu_0 = RSU(0,path1,(-1,-1),Vector{Vehicle}[],50)
	vehicle1 = Vehicle(1,course1,course1.path.start,10,0,0)
	vehicle2 = Vehicle(2,course2,course2.path.start,10,0,0)
	vehicle3 = Vehicle(3,course3,course3.path.start,10,0,0)
	vehicle4 = Vehicle(4,course4,course4.path.start,10,0,0)
	vehicle5 = Vehicle(5,course5,course5.path.start,10,0,0)
	vehicles = [vehicle1,vehicle2,vehicle3,vehicle4,vehicle5]
	
end
  ╠═╡ =#

# ╔═╡ 6c7038b4-7055-4b80-935f-d933235307ba
# Define paths of different lengths that connect to each other to span the grid
begin
# Create some paths within the valid map size
# Create some simplified paths
path1 = Path(1, (1.13, 0.0), (1.13, 15), 0)
path2 = Path(2, (1.13, 0.0), (13.3, 2), 0)
path3 = Path(3, (13.3, 2), (12.8, 9.3), 0)
path4 = Path(4, (12.8, 9.3), (11.1, 11.1), 0)
path5 = Path(5, (11.1, 11.1), (10.9, 15), 0)
path6 = Path(6, (6.5, 0.0), (6.2, 0.8), 0)
path7 = Path(7, (6.2, 0.8), (5.8, 15), 0)
path8 = Path(8, (5.9, 9), (1.13, 15), 0)
path9 = Path(9, (5, 0), (4.7, 10.3), 0)
path10 = Path(10, (5.9,9), (11.1,11.1), 0)
path11 = Path(11, (5.9,9), (13,6.2),0)

paths = Vector{Path}(undef,11)
for i in 1:11
	paths[i] = eval(Symbol("path",i))
end
paths
end


# ╔═╡ 28008d6b-c034-46fb-b0b2-4e54f4f969d5
begin
	course1 = Course(path1,nothing)
	course2 = Course(path2,Course(path3,Course(path4,Course(path5,nothing))))
	course3 = Course(path6,Course(path7,nothing))
	course_list = [course1,course2,course3]
end

# ╔═╡ 9b536663-99bc-4da6-b34a-56260e054ff9
begin
	rsu_0 = RSU(0,path1,(-1,-1),Vector{Vehicle}[],50)
	velocities = [20,25,30,35]
	vehicles = Vector{Vehicle}(undef,10)
	for i in 1:10
		course = rand(course_list)
		v = rand(velocities)
		vehicles[i] = Vehicle(i,course,course.path.start,v,0,0)
	end
	vehicles
end

# ╔═╡ dd72eaf3-586f-4384-a934-89fbed70df2b
df = DataFrame(veh_id = Int[], 
               position_x = Float64[], 
               position_y = Float64[], 
               velocity = Float64[], 
               rsu_id = Integer[], 
               allocated_BRB = Float64[])

# ╔═╡ 2f106f40-f67f-47f9-ac45-f036cfbc73b9
for vehicle in vehicles
	push!(df, (vehicle.veh_id, 
               vehicle.position[1], 
               vehicle.position[2], 
               vehicle.velocity, 
               vehicle.rsu_id, 
               vehicle.allocated_BRB))
end

# ╔═╡ c7616c2b-85b9-4176-b67e-a078e625dc95
df

# ╔═╡ a383b0ac-d6e6-462a-9abc-486aed845511
begin
#@gif for t in 1:60	
start_points = [(path.start[1], path.start[2]) for path in paths]
end_points = [(path.fin[1], path.fin[2]) for path in paths]

# Create a plot
plt = plot(legend=false, xlim=(0, GRID_SIZE), ylim=(0, GRID_SIZE), aspect_ratio=:equal)

# Plot the paths as lines
for path in paths
    plot!(plt,[path.start[1], path.fin[1]], [path.start[2], path.fin[2]], color=:blue, lw=2)
end

# Plot start and end points as dots
scatter!(plt,start_points, color=:green, markersize=5, label="Start")
scatter!(plt,end_points, color=:red, markersize=5, label="End")

# Show the plot
	map(vehicles) do vehicle
		update_vehicle_position!(vehicle,vehicle.velocity*(SLOT_DUR/60))
		update_vehicle_path!(vehicle)
	end
	remove_vehicles!(vehicles)
	[scatter!(plt,[vehicle.position[1]+0.0125*rand()],[vehicle.position[2]+0.0125*rand()],marker=:star4,markersize=10) for vehicle in vehicles]
	#scatter!(p,xlim = (0,15), ylim= (0,15))
	plt
#end
end

# ╔═╡ Cell order:
# ╠═161af454-2de6-11ee-05de-c9432d420a24
# ╠═235616e2-3377-4a03-bd2e-86514d73b262
# ╠═04fe6e61-bd64-4204-9104-76e39d8a6c46
# ╠═98cffce9-5036-4fea-88e8-fe82463c1e47
# ╠═c1bb1b4c-fc8e-4371-866c-40bbb6c7746f
# ╠═3e51ae9e-1a43-4cc0-8184-a652e3675c98
# ╠═6c7038b4-7055-4b80-935f-d933235307ba
# ╠═28008d6b-c034-46fb-b0b2-4e54f4f969d5
# ╠═9b536663-99bc-4da6-b34a-56260e054ff9
# ╠═a7b0c4de-aa7d-4201-97a2-d2df8cfa902c
# ╠═dd72eaf3-586f-4384-a934-89fbed70df2b
# ╠═2f106f40-f67f-47f9-ac45-f036cfbc73b9
# ╠═c7616c2b-85b9-4176-b67e-a078e625dc95
# ╠═d8ba55ec-e66c-44e7-ac63-d2948264acaf
# ╠═a383b0ac-d6e6-462a-9abc-486aed845511
