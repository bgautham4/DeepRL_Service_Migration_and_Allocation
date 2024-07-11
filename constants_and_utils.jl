module UtilsAndConstants# Constants and Util functions
	
module Constants 

export alpha, N0, epsilon, d0, gain, P_tr, W 

const alpha = 3.75 # Path loss exponent
const N0 = 10^(-17.4) #In mW/Hz
const epsilon = 0.25 # power control factor
const d0 = 500 #Reference distance for path loss in meters
const gain = 3
const P_tr = 200 #Transmit power in mW
const W = 180e3 #Bandwidth of a resource block in Hz

end

module UtilFunctions

export gen_snr, transmit_rate 
using ..Constants

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

using .UtilFunctions 
export transmit_rate 
#Add export list for any other constant/func if needed..
end
