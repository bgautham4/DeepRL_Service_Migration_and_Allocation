module UtilsAndConstants# Constants and Util functions
	

module Constants #Deifining some constants

export SLOT_DUR, RSU_R, alpha, N0, epsilon, d0, gain, P_tr, N_RSU, B̄, C̄, W #Export list for calling scope

const SLOT_DUR = 1.5 #In minutes
const RSU_R = 0.5 #In Kilometers
const alpha = 3.75 # Path loss exponent
const N0 = 10^(-17.4) #In mW/Hz
const epsilon = 0.25 # power control factor
const d0 = 500 #Reference distance for path loss in meters
const gain = 3
const P_tr = 200 #Transmit power in mW
const N_RSU = 5
const B̄ = 50 #Max number of BRBs
const C̄ = 100 #MAX number of CRBs
const W = 180e3 #Bandwidth of a resource block

end

module ServiceDef

export Service, SERVICE_SET, N_SERVICE

struct Service
	service_indx::Int
	data_size::Int #In Mb
	crb_needed::Real #In Gcycles i.e x10⁹
	thresh::Real
	max_thresh::Real
end

begin
	const service1 = Service(
			1,
			5,
			.2,
			0.35,
			0.5
		)

		const service2 = Service(
			2,
			7,
			.3,
			0.35,
			0.5
		)
		const service3 = Service(
			3,
			10,
			.4,
			0.35,
			0.5
		)
	const SERVICE_SET = (service1, service2, service3)
	const N_SERVICE = length(SERVICE_SET)
end 

end

module UtilFunctions

export gen_snr, transmit_rate #Export these functions to using scope
using ..Constants:N0, P_tr, N0, W, gain, d0, alpha, epsilon#Get these constants into this scope for use
function gen_snr(lv::Real, BRB::Integer)
		P_recv = P_tr*(gain)*((lv/d0)^(alpha*(epsilon-1)))
		P_noise = N0 * W * 180e3
		return P_recv / P_noise
end

function transmit_rate(lv::Real, BRB::Integer)
        SNR = gen_snr(lv, BRB)
        #SNR == Inf ? SNR = 1 : nothing
        R = BRB * W * log2(1+SNR)
        return R
end

end

using .Constants #Export some useful constants to calling scope(i.e parent module UtilsAndConstants)
export N_RSU, SLOT_DUR, RSU_R, B̄, C̄ #Export out into calling scope (i.e main scope)

using .ServiceDef  
export Service, SERVICE_SET, N_SERVICE

using .UtilFunctions
export transmit_rate

end