extends Control



func _ready() -> void:
	prints(generate_noise(10))
	prints(generate_noise(10))
	prints(generate_noise(10))








func generate_noise(size: int) -> Array:
	var noise = []

	for i in range(size):
		noise.append(randf_range(0.9, 0.0001) + randf_range(0.0001, 0.0) )  # Genera un valor aleatorio entre 0 y 1
	return noise
