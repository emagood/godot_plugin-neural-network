#extends Control
#
#func _ready() -> void:
	#var algorithms = [
		#"Rprop",
		#"Adam",
		#"Nadam",
		#"Adamax",
		#"Yogi",
		#"Adadelta"
	#]
#
	#var input_data = []
	#input_data.append([0.0 as Variant, 0.0 as Variant] as Variant)
	#input_data.append([1.0 as Variant, 0.0 as Variant] as Variant)
	#input_data.append([0.0 as Variant, 1.0 as Variant] as Variant)
	#input_data.append([1.0 as Variant, 1.0 as Variant] as Variant)
#
	#var target_data = []
	#target_data.append([0.0 as Variant] as Variant)
	#target_data.append([1.0 as Variant] as Variant)
	#target_data.append([1.0 as Variant] as Variant)
	#target_data.append([0.0 as Variant] as Variant)
#
	## Imprimir tipos y estructura de los datos en español
	#print("Tipo de input_data: ", typeof(input_data))  # Esperamos TYPE_ARRAY (19)
	#for item in input_data:
		#print("Tipo de elemento en input_data: ", typeof(item), " longitud: ", item.size())  # Esperamos TYPE_VARIANT (28)
		#for subitem in item:
			#print("Tipo de subelemento en input_data: ", typeof(subitem))  # Esperamos TYPE_VARIANT (28)
#
	#print("Tipo de target_data: ", typeof(target_data))  # Esperamos TYPE_ARRAY (19)
	#for item in target_data:
		#print("Tipo de elemento en target_data: ", typeof(item), " longitud: ", item.size())  # Esperamos TYPE_VARIANT (28)
		#for subitem in item:
			#print("Tipo de subelemento en target_data: ", typeof(subitem))  # Esperamos TYPE_VARIANT (28)
#
	#for algorithm in algorithms:
		#print("Probando con el algoritmo: " + algorithm)
		#var nn = NNET.new([2, 10, 10, 1], false)
		#nn.set_loss_function(BNNET.LossFunctions.MSE)
		#set_algorithm(nn, algorithm)
		#nn.set_batch_size(4)
#
		## Entrenar la red y mostrar pérdida
		#for i in range(1000):
			#nn.train(input_data, target_data)
			#if i % 100 == 0:
				#print("Iteración ", i, " - Pérdida: ", nn.get_loss(input_data, target_data))
#
		## Guardar el modelo entrenado
		#nn.save_binary("res://trained_model_" + algorithm + ".bin")
		#print("Modelo guardado exitosamente con el algoritmo ", algorithm)
		#XOR_test(nn)
#
#func set_algorithm(nn: NNET, algorithm: String) -> void:
	#match algorithm:
		#"Rprop":
			#nn.use_Rprop(0.3)
		#"Adam":
			#nn.use_Adam(0.01)
		#"Nadam":
			#nn.use_Nadam(0.01)
		#"Adamax":
			#nn.use_Adamax(0.01)
		#"Yogi":
			#nn.use_Yogi(0.01)
		#"Adadelta":
			#nn.use_Adadelta(0.9)
		#_:
			#nn.use_Rprop(0.3)
#
#func XOR_test(nn: NNET) -> void:
	#var input_data = []
	#input_data.append([0.0 as Variant, 0.0 as Variant] as Variant)
	#input_data.append([1.0 as Variant, 0.0 as Variant] as Variant)
	#input_data.append([0.0 as Variant, 1.0 as Variant] as Variant)
	#input_data.append([1.0 as Variant, 1.0 as Variant] as Variant)
#
	#print("----------------------------------------------------")
	#print("Pérdida: ", nn.get_loss(input_data, [
		#[0.0 as Variant] as Variant,
		#[1.0 as Variant] as Variant,
		#[1.0 as Variant] as Variant,
		#[0.0 as Variant] as Variant
	#]))
	#for data in input_data:
		#nn.set_input(data)
		#nn.propagate_forward()
		#print("Entrada: ", data, " Salida: ", nn.get_output())
	#print("----------------------------------------------------")
	
extends Control

func _ready() -> void:
	var algorithms = [
		"Rprop",
		"Adam",
		"Nadam",
		"Adamax",
		"Yogi",
		"Adadelta"
	]

	var input_data = [
		[0.0, 0.0],
		[1.0, 0.0],
		[0.0, 1.0],
		[1.0, 1.0]
	]

	var target_data = [
		[0.0],
		[1.0],
		[1.0],
		[0.0]
	]

	# Imprimir tipos y estructura de los datos en español
	print("Tipo de input_data: ", typeof(input_data))  # Esperamos TYPE_ARRAY (19)
	for item in input_data:
		print("Tipo de elemento en input_data: ", typeof(item), " longitud: ", item.size())  # Esperamos TYPE_ARRAY (19)
		for subitem in item:
			print("Tipo de subelemento en input_data: ", typeof(subitem))  # Esperamos TYPE_REAL (4)

	print("Tipo de target_data: ", typeof(target_data))  # Esperamos TYPE_ARRAY (19)
	for item in target_data:
		print("Tipo de elemento en target_data: ", typeof(item), " longitud: ", item.size())  # Esperamos TYPE_ARRAY (19)
		for subitem in item:
			print("Tipo de subelemento en target_data: ", typeof(subitem))  # Esperamos TYPE_REAL (4)

	for algorithm in algorithms:
		print("Probando con el algoritmo: " + algorithm)
		var nn = NNET.new([2, 10, 10, 1], false)
		nn.set_loss_function(BNNET.LossFunctions.MSE)
		set_algorithm(nn, algorithm)
		nn.set_batch_size(4)

		# Entrenar la red y mostrar pérdida
		for i in range(1000):
			nn.train(input_data, target_data)
			if i % 100 == 0:
				print("Iteración ", i, " - Pérdida: ", nn.get_loss(input_data, target_data))

		# Guardar el modelo entrenado
		nn.save_binary("res://trained_model_" + algorithm + ".bin")
		print("Modelo guardado exitosamente con el algoritmo ", algorithm)
		XOR_test(nn)

func set_algorithm(nn: NNET, algorithm: String) -> void:
	match algorithm:
		"Rprop":
			nn.use_Rprop(0.3)
		"Adam":
			nn.use_Adam(0.01)
		"Nadam":
			nn.use_Nadam(0.01)
		"Adamax":
			nn.use_Adamax(0.01)
		"Yogi":
			nn.use_Yogi(0.01)
		"Adadelta":
			nn.use_Adadelta(0.9)
		_:
			nn.use_Rprop(0.3)

func XOR_test(nn: NNET) -> void:
	var input_data = [
		[0.0, 0.0],
		[1.0, 0.0],
		[0.0, 1.0],
		[1.0, 1.0]
	]

	print("----------------------------------------------------")
	print("Pérdida: ", nn.get_loss(input_data, [
		[0.0],
		[1.0],
		[1.0],
		[0.0]
	]))
	for data in input_data:
		nn.set_input(data)
		nn.propagate_forward()
		print("Entrada: ", data, " Salida: ", nn.get_output())
	print("----------------------------------------------------")

	
	
	
