'''
red neuronal realiza aproximadamente 172,809,000 operaciones en la prueba :)
'''
extends Node

func _ready() -> void:
	print("Prueba de redes XOR simples con múltiples algoritmos:")
	test_xor_network()
	
	print("\nProbar redes más complejas con varios algoritmos e inicializaciones de peso:")
	
	# Pruebas adicionales
	var configurations = [
		["Adam", "Xavier"],
		["NAG", "He"],
		["Nadam", "Random"],
		["Adamax", "Random"],
		["Adadelta", "Xavier"],
		["Yogi", "He"]
	]
	
	for config in configurations:
		print("Testing conf / test: " + config[0] + " with " + config[1] + " Initialization")
		test_complex_network([2, 10, 10, 1], config[0], config[1])
		test_complex_network([2, 50, 50, 1], config[0], config[1])
		test_complex_network([2, 10, 10, 10, 10, 1], config[0], config[1])

func test_xor_network() -> void:
	var nn = NNET.new([2, 5, 6, 1], false)
	nn.set_loss_function(BNNET.LossFunctions.MSE)
	nn.use_Rprop(0.3)
	nn.set_batch_size(4)

	run_training(nn, 1500, "XOR (Rprop)")
	XOR_test(nn)

func test_complex_network(architecture: Array, algorithm: String, init_method: String) -> void:
	var nn = NNET.new(architecture, false)
	nn.set_loss_function(BNNET.LossFunctions.MSE)
	set_algorithm(nn, algorithm)
	nn.set_batch_size(4)

	# Inicializar pesos
	match init_method:
		"Xavier":
			nn.reinit()
			print("Pesos inicializados con Xavier")
		"He":
			nn.reinit()
			print("Pesos inicializados con He")
		_:
			nn.reinit()  # Inicialización aleatoria por defecto

	run_training(nn, 1500, "Complex (" + algorithm + ", " + init_method + ")")
	XOR_test(nn)

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

func run_training(nn: NNET, iterations: int, label: String) -> void:
	for i in range(iterations):
		nn.train(
			[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
			[[0.0], [1.0], [1.0], [0.0]]
		)
		if i % 100 == 0:
			print(label, " entrenando la redes - Iteración ", i)
	print(label, "entrenamiento completo / Network Training Completed")

func XOR_test(nn: NNET) -> void:
	print("----------------------------------------------------")
	print("Loss: ", nn.get_loss([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[0.0], [1.0], [1.0], [0.0]]))
	nn.set_input([0.0, 0.0]); nn.propagate_forward()
	print("0, 0: ", nn.get_output())
	nn.set_input([1.0, 0.0]); nn.propagate_forward()
	print("1, 0: ", nn.get_output())
	nn.set_input([0.0, 1.0]); nn.propagate_forward()
	print("0, 1: ", nn.get_output())
	nn.set_input([1.0, 1.0]); nn.propagate_forward()
	print("1, 1: ", nn.get_output())
	print("----------------------------------------------------")
