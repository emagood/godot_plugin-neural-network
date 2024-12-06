extends Control


func _ready() -> void:
	var nn = NNET.new([2, 5, 6, 1], false)
	nn.load_data("res://trained_model.bin")
	print("Modelo cargado exitosamente.")
	
	# Proveer algunos datos de entrada para probar el modelo cargado
	var input_data = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
	
	for data in input_data:
		nn.set_input(data)
		nn.propagate_forward()
		print("Input: ", data, " Output: ", nn.get_output())
