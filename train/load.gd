extends Control
var mode_capa = [2, 5, 6, 1]
var dir_data = "res://data/trained_model.bin"

func _ready() -> void:
	var nn = NNET.new(mode_capa, false)
	nn.load_data(dir_data)# cambiar
	print("Modelo cargado exitosamente.")
	
	# Proveer algunos datos de entrada para probar el modelo cargado
	var input_data = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
	
	for data in input_data:
		nn.set_input(data)
		nn.propagate_forward()
		print("Input: ", data, " Output: ", nn.get_output())
func _process(delta: float) -> void:
	queue_free()
