extends Node

func _ready() -> void:
	# Lista de configuraciones de algoritmos
	var algorithms = [
		"Adam",
		"NAG",
		"Nadam",
		"Adamax",
		"Adadelta",
		"Yogi"
	]
	
	# Definir algunos datos de entrada
	var input_data = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
	
	for algorithm in algorithms:
		print("Testing with algorithm: " + algorithm)
		# Inicializa la red sin entrenarla
		var nn = NNET.new([2, 5, 6, 1], false)
		nn.set_loss_function(BNNET.LossFunctions.MSE)
		set_algorithm(nn, algorithm)
		nn.set_batch_size(4)
		
		# Probar cada input y mostrar la salida
		for data in input_data:
			nn.set_input(data)
			nn.propagate_forward()  # Propaga hacia adelante sin entrenamiento
			print("Input: ", data, " Output: ", nn.get_output())
		print("----------------------------------------------------")

func set_algorithm(nn: NNET, algorithm: String) -> void:
	match algorithm:
		"Adam":
			nn.use_Adam(0.001)
		"NAG":
			nn.use_NAG(0.1, 0.9)
		"Nadam":
			nn.use_Nadam(0.001)
		"Adamax":
			nn.use_Adamax(0.001)
		"Yogi":
			nn.use_Yogi(0.001)
		"Adadelta":
			nn.use_Adadelta(0.9)
		_:
			nn.use_Rprop(0.3)  # Algoritmo predeterminado
