extends Control


var generator_structure = [100, 128, 256, 512, size*size]  # Estructura de ejemplo para el generador
var discriminator_structure = [size*size, 512, 256, 128, 1]  # Estructura de ejemplo para el discriminador
var use_bias = true

func _ready() -> void:
	var gan = GAN.new(generator_structure, discriminator_structure, use_bias)
	gan.save_gan("res://data/gans")
	return
