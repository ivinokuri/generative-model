struct Wheel
	velocity::Float16
	radialVelocity::Float16
	angle::Float16
end

struct GrabHand
	fail_prob::Float16
	value::Float16
	holding::Bool
end