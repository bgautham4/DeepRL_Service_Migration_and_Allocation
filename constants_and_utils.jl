module UtilsAndConstants# Constants and Util functions
	

module Constants #Deifining some constants

export alpha, N0, epsilon, d0, gain, P_tr, W #Export list for calling scope

const alpha = 3.75 # Path loss exponent
const N0 = 10^(-17.4) #In mW/Hz
const epsilon = 0.25 # power control factor
const d0 = 500 #Reference distance for path loss in meters
const gain = 3
const P_tr = 200 #Transmit power in mW
const W = 180e3 #Bandwidth of a resource block

end

module UtilFunctions

export gen_snr, transmit_rate #Export these functions to using scope
using ..Constants#Get constants into this scope for use

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

using .UtilFunctions #Bring into scope of parent module
export transmit_rate #Export this function to calling scope of parent module.

end