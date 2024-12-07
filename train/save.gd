#extends Control
#
#func _ready() -> void:
	#var nn = NNET.new([2, 5, 6, 1], false)
	#nn.set_loss_function(BNNET.LossFunctions.MSE)
	#nn.use_Adam(0.001)
	#nn.set_batch_size(4)
	#
	## Entrenar la red (opcional, para demostrar el proceso completo)
	#for i in range(1000):
		#nn.train(
			#[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
			#[[0.0], [1.0], [1.0], [0.0]]
		#)
	#
	## Guardar el modelo entrenado
	#nn.save_binary("res://trained_model.bin")
	#print("Modelo guardado exitosamente.")
extends Control

func _ready() -> void:
	var nn = NNET.new([2, 5, 6, 1], false)
	nn.set_loss_function(BNNET.LossFunctions.MSE)
	nn.use_Adam(0.01)
	nn.set_batch_size(4)
	
	# Entrenar la red y mostrar pérdida
	for i in range(1000):
		nn.train(
			[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
			[[0.0], [1.0], [1.0], [0.0]]
		)
		if i % 100 == 0:
			print("Iteración ", i, " - Pérdida: ", nn.get_loss(
				[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
				[[0.0], [1.0], [1.0], [0.0]]
			))
	
	# Guardar el modelo entrenado
	nn.save_binary("res://data/trained_model.bin")
	print("Modelo guardado exitosamente.")
	
	# Mostrar resultados finales
	print("Resultados finales:")
	XOR_test(nn)

func XOR_test(nn: NNET) -> void:
	print("----------------------------------------------------")
	print("Loss: ", nn.get_loss([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[0.0], [1.0], [1.0], [0.0]]))
	nn.set_input([0.0, 0.0])
	nn.propagate_forward()
	print("0, 0: ", nn.get_output())
	nn.set_input([1.0, 0.0])
	nn.propagate_forward()
	print("1, 0: ", nn.get_output())
	nn.set_input([0.0, 1.0])
	nn.propagate_forward()
	print("0, 1: ", nn.get_output())
	nn.set_input([1.0, 1.0])
	nn.propagate_forward()
	print("1, 1: ", nn.get_output())
	print("----------------------------------------------------")
