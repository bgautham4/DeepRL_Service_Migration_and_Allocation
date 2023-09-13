### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 6c25b640-20a3-11ee-0011-45a111457fa3
let
	using Pkg
	Pkg.activate(".")
end

# ╔═╡ 5a2aa0d0-3b25-49a9-b05c-c40126d6c64a
using Plots

# ╔═╡ c78cc19f-d0d4-40c9-84f4-3453f1b26de8
const MAP_SIZE = 5 #Defines the total map size as MAP_SIZE x MAP_SIZE Grids  

# ╔═╡ 2ee4f4aa-704f-4204-83c5-410b54856701
const GRID_SIZE = 2 #Defines the size of each Grid in kilometers

# ╔═╡ 3c7a89fe-2887-460d-97a9-d2c1c791cf68
const SLOT_DUR = 2 #Defines the length of each time slot in minutes.

# ╔═╡ c34e7117-78fe-4fe3-85e2-93203a120830
md""" 
Based on the above definitions we can find some derived parameters:
* Assuming that the vehicle can only traverse atmost one grid between durations, the max allowable speed is then (60 * GRID\_SIZE / SLOT\_DUR) km/hr

Here it is 60km/hr OR 1 km/min OR 1 slots/grid

Given a speed of x km/min we would get 1/x slots/grid
"""

# ╔═╡ fab8fd33-8861-4353-80a8-e2b936b34c19
begin
	@enum LinkStatus invalid=0 valid=1
	
	mutable struct Grid
		grid_no::Integer
		flow_rates::Vector{Tuple{LinkStatus,Real}}
		n_vehs::Integer
	end
	
	Grid(grid_no::Integer) = Grid(grid_no,fill((invalid,0),MAP_SIZE*MAP_SIZE),0)
	
	mutable struct Vehicle
		current_grid::Integer
		path::Vector{Integer}
		velocity::Real
		slots_to_next_grid::Integer
	end

	function Vehicle(path::Vector{<:Integer},grids::Vector{Grid})
		vehicle = Vehicle(path[1],path,0,0)
		vehicle.velocity = grids[path[1]].flow_rates[path[2]][2]
		vehicle.slots_to_next_grid = ceil(Integer,1/vehicle.velocity)
		return vehicle
	end
end

# ╔═╡ 76dcaae7-da58-45d0-b1f7-418fcd2f95a1
begin
	grids = [Grid(i) for i in 1:25]
	for grid in grids
		grid_no = grid.grid_no
		if (grid_no%5)-1==0
			grid.flow_rates[grid_no+1] = (valid,rand([1/3,2/3,1]))
		
		elseif (grid_no%5) == 0
			grid.flow_rates[grid_no-1] = (valid,rand([1/3,2/3,1]))

		else
			grid.flow_rates[grid_no+1] = (valid,rand([1/3,2/3,1]))
			grid.flow_rates[grid_no-1] = (valid,rand([1/3,2/3,1]))

		end
		
		if grid_no > 5
			grid.flow_rates[grid_no-5] = (valid,rand([1/3,2/3,1]))
		end

		if grid_no < 21
			grid.flow_rates[grid_no+5] = (valid,rand([1/3,2/3,1]))
		end
	end
	grids
end

# ╔═╡ a414381e-e2f5-4913-87be-3e23c9d9c9ce
begin
	function incr_vehicle_count!(grid::Grid)
		grid.n_vehs += 1
	end

	function decr_vehicle_count!(grid::Grid)
		grid.n_vehs -= 1
	end
	
	function in_terminal_grid(vehicle::Vehicle)
		current_grid = vehicle.current_grid
		path_len = length(vehicle.path)
		current_grid_index = findfirst(x->x==current_grid,vehicle.path)
		
		if current_grid_index == path_len
			return true
		end
		
		return false
	end
	
	function update_state!(vehicle::Vehicle,grids::Vector{Grid})
		if vehicle.slots_to_next_grid != 0
			vehicle.slots_to_next_grid -= 1
			return
		end
		
		current_grid = vehicle.current_grid
		current_grid_index = findfirst(x->x==current_grid,vehicle.path)
		vehicle.current_grid = vehicle.path[current_grid_index+1]
		decr_vehicle_count!(grids[current_grid])
		incr_vehicle_count!(grids[vehicle.current_grid])
		
		if in_terminal_grid(vehicle)
			return
		end
		
		next_grid = vehicle.path[current_grid_index+2]
		_,vehicle.velocity = grids[vehicle.current_grid].flow_rates[next_grid]
		vehicle.slots_to_next_grid = ceil(Integer,1/vehicle.velocity)
		
		return
	end
	
	function inject_vehicles!(vehicles::Vector{Vehicle},grids::Vector{Grid})
		#paths = [[1,2,3,8,13,14,15,20,25],[4,9,14,19,20],[11,16,17,18,19,20],[1,6,11,16,21],[4,3,2,7,12,17,22,21],[1, 6, 11, 16, 21, 22, 23, 24, 25, 20],[11, 16, 21, 22, 23, 24, 25],[11, 6, 1, 2, 3, 4, 9, 14, 19, 24, 25],[11, 16, 21]]
		paths = [[1, 2, 3, 4, 9, 14, 19, 24, 25],[4, 5, 10, 15, 20],[11, 12, 13, 14, 19, 24, 25],[1, 2, 3, 4, 9, 14, 19, 24, 23],[4, 5, 10, 15, 20, 25, 24, 23, 22, 21]]
		new_vehicles = [Vehicle(rand(paths),grids) for _ in 1:10]
		for vehicle in new_vehicles
			current_grid = vehicle.current_grid
			incr_vehicle_count!(grids[current_grid]) 
		end
		push!(vehicles,new_vehicles...)
		return 
	end
	"""
	function init_grid_states!(vehicles::Vector{Vehicle},grids::Vector{Grid})
		for vehicle in vehicles
			current_grid = vehicle.current_grid
			incr_vehicle_count!(grids[current_grid]) 
		end
		return
	end
"""
	function init_simulation!(vehicles::Vector{Vehicle},grids::Vector{Grid})
		for grid in grids
			grid.n_vehs = 0
		end
		inject_vehicles!(vehicles,grids)
		return
	end
	
	function step_simulation!(vehicles::Vector{Vehicle},grids::Vector{Grid})
		
		for (indx,vehicle) in enumerate(vehicles)
			
			if in_terminal_grid(vehicle)
				decr_vehicle_count!(grids[vehicle.current_grid])
				popat!(vehicles,indx)
				continue
			end

			update_state!(vehicle,grids)
		end
		
		inject_vehicles!(vehicles,grids)
		return
	end

	function count_vehicles(grids::Vector{Grid})
		counts = zeros(Int,length(grids))
		for (indx,grid) in enumerate(grids)
			counts[indx] = grid.n_vehs
		end
		return counts
	end
end

# ╔═╡ 647bd962-08ae-46ee-9b2a-93cc5526115d
begin
	vehicles = Vector{Vehicle}()
	init_simulation!(vehicles,grids)
	p = plot()
	n_vehs_grids = [[] for _ in 1:25]
	initial_counts = count_vehicles(grids)
	for (grid,counts) in enumerate(initial_counts)
		push!(n_vehs_grids[grid],counts)
	end
	
	for t_slot in 1:100
		step_simulation!(vehicles,grids)
		counts = count_vehicles(grids)
		for (grid,count) in enumerate(counts)
			push!(n_vehs_grids[grid],count)
		end
	end

	for grid_data in n_vehs_grids[1:5]
		plot!(p,grid_data)
	end
	
end
		

# ╔═╡ 5b545339-ba6b-4575-a821-25a92421cd5d
p

# ╔═╡ 68d82fba-021b-4555-adc5-1b61cd66a082
begin
	temp = zeros(5,5)
	for grid in grids
		flows = map(grid.flow_rates) do (_,y)
			y
		end
		flows = reshape(flows,5,5)
		temp = temp .+ flows
	end
	temp
	yflip!(heatmap(temp))
end		

# ╔═╡ Cell order:
# ╠═6c25b640-20a3-11ee-0011-45a111457fa3
# ╠═5a2aa0d0-3b25-49a9-b05c-c40126d6c64a
# ╠═c78cc19f-d0d4-40c9-84f4-3453f1b26de8
# ╠═2ee4f4aa-704f-4204-83c5-410b54856701
# ╠═3c7a89fe-2887-460d-97a9-d2c1c791cf68
# ╟─c34e7117-78fe-4fe3-85e2-93203a120830
# ╠═fab8fd33-8861-4353-80a8-e2b936b34c19
# ╠═76dcaae7-da58-45d0-b1f7-418fcd2f95a1
# ╠═a414381e-e2f5-4913-87be-3e23c9d9c9ce
# ╠═647bd962-08ae-46ee-9b2a-93cc5526115d
# ╠═5b545339-ba6b-4575-a821-25a92421cd5d
# ╠═68d82fba-021b-4555-adc5-1b61cd66a082
