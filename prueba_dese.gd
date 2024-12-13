extends Control

# Datos de 121 carros -> [Antigüedad, costo de salida al mercado]

var x = [[0.0, 1.0], [0.1, 1.0], [0.2, 1.0], [0.3, 1.0], [0.4, 1.0],
	[0.5, 1.0], [0.6, 1.0], [0.7, 1.0], [0.8, 1.0], [0.9, 1.0],
	[1.0, 1.0], [0.0, 0.9], [0.1, 0.9], [0.2, 0.9], [0.3, 0.9],
	[0.4, 0.9], [0.5, 0.9], [0.6, 0.9], [0.7, 0.9], [0.8, 0.9],
	[0.9, 0.9], [1.0, 0.9], [0.0, 0.8], [0.1, 0.8], [0.2, 0.8],
	[0.3, 0.8], [0.4, 0.8], [0.5, 0.8], [0.6, 0.8], [0.7, 0.8],
	[0.8, 0.8], [0.9, 0.8], [1.0, 0.8], [0.0, 0.7], [0.1, 0.7],
	[0.2, 0.7], [0.3, 0.7], [0.4, 0.7], [0.5, 0.7], [0.6, 0.7],
	[0.7, 0.7], [0.8, 0.7], [0.9, 0.7], [1.0, 0.7], [0.0, 0.6],
	[0.1, 0.6], [0.2, 0.6], [0.3, 0.6], [0.4, 0.6], [0.5, 0.6],
	[0.6, 0.6], [0.7, 0.6], [0.8, 0.6], [0.9, 0.6], [1.0, 0.6],
	[0.0, 0.5], [0.1, 0.5], [0.2, 0.5], [0.3, 0.5], [0.4, 0.5],
	[0.5, 0.5], [0.6, 0.5], [0.7, 0.5], [0.8, 0.5], [0.9, 0.5],
	[1.0, 0.5], [0.0, 0.4], [0.1, 0.4], [0.2, 0.4], [0.3, 0.4],
	[0.4, 0.4], [0.5, 0.4], [0.6, 0.4], [0.7, 0.4], [0.8, 0.4],
	[0.9, 0.4], [1.0, 0.4], [0.0, 0.3], [0.1, 0.3], [0.2, 0.3],
	[0.3, 0.3], [0.4, 0.3], [0.5, 0.3], [0.6, 0.3], [0.7, 0.3],
	[0.8, 0.3], [0.9, 0.3], [1.0, 0.3], [0.0, 0.2], [0.1, 0.2],
	[0.2, 0.2], [0.3, 0.2], [0.4, 0.2], [0.5, 0.2], [0.6, 0.2],
	[0.7, 0.2], [0.8, 0.2], [0.9, 0.2], [1.0, 0.2], [0.0, 0.1],
	[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1], [0.5, 0.1],
	[0.6, 0.1], [0.7, 0.1], [0.8, 0.1], [0.9, 0.1], [1.0, 0.1],
	[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0],
	[0.5, 0.0], [0.6, 0.0], [0.7, 0.0], [0.8, 0.0], [0.9, 0.0],
	[1.0, 0.0]]

# 0 : normal    1 : coleccionable

var y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
	0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
	0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
	0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
	0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
	0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
	0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
	0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
	
	

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
	var nn = NNET.new([2, 5, 6, 1], true)
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
		
		var tag = []
		for k in range(y.size()):
			tag.append([y[k]])
		
		nn.train(x,tag)
		if i % 100 == 0:
			print(label, " entrenando la redes - Iteración ", i)
	print(label, "entrenamiento completo / Network Training Completed")

func XOR_test(nn: NNET) -> void:
	var select = randi() % 100
	var aux = []
	var aux2 = []
	aux.append(y[select])
	
	aux2.append([x[select]])
	print("----------------------------------------------------")
	print("Loss: ", nn.get_loss(x[select],aux))# roto[1.0],[0.0]]
	prints(x[select], " " , aux  , "  " , [y[select]] , "   " , x[select].size() )
	nn.set_input(x[select]); nn.propagate_forward()
	print(":" + str(y[select]), nn.get_output())
	nn.set_input([0.0, 0.9]); nn.propagate_forward()
	print("0.0, 0.9: ", nn.get_output())
	nn.set_input([0.1, 0.9]); nn.propagate_forward()
	print("0.1, 0.9: ", nn.get_output())
	nn.set_input([0.3, 1.0]); nn.propagate_forward()
	print("0.1, 0.9: ", nn.get_output())
	prints("se espera 0 , 1 , 1 , 0 : en al salida ")
	print("----------------------------------------------------")
