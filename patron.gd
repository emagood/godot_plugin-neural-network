extends Control
'''
################esta en desarollo aun falta mucho pero vamos ##############

Etiqueta Descripción
0	Camiseta/top
1	Pantalón
2	Jersey
3	Vestido
4	Abrigo
5	Sandalia
6	Camisa
7	Zapatillas de deporte
8	Bolsa
9	Botín
'''
# Diccionario de etiquetas con su descripción
var etiquetas = {
	0: "Camiseta/top ",
	1: "Pantalón 👖",
	2: "Jersey",
	3: "Vestido 👗",
	4: "Abrigo 👕",
	5: "Sandalia",
	6: "Camisa 🧥",
	7: "Zapatillas de deporte 👟",
	8: "Bolsa 👜",
	9: "Botín 👢"
}

var inig = NNET
var history = ""
var load_data
var image_paths = []
var tag = []
var tag_data = []
var dir_fashion = "C:/Users/Emabe/Desktop/entrena/fashion/train/"
var max = 100# para todo cargar imagen leer el cvs y mas 
var leer = max
var real_data
var fashion = "C:/Users/Emabe/Desktop/entrena/fashion/train.csv"

func _ready():
	''' solo ia '''
	randomize()
	var discriminator_structure = [784,1280,512,320,80,10]  # estructura de ejemplo 
	var use_bias = true
	
	#var gan = IMG.new(discriminator_structure, use_bias)
	#gan.load_gan("res://data/etiqueta_img_train.bin")

	
	
	
	
	''' archivo csv'''
	
	
	var file_path = fashion # Ruta al archivo CSV
	var file = FileAccess.open(file_path, FileAccess.READ)




	if file:
		while not file.eof_reached() :
			
			if leer <= 0:
				break
			var line = file.get_line()
			var columns = line.split(",")
			var first_value = columns[0]  # primer valor de la línea
			var last_value = columns[columns.size() - 1]  # ultimo valor de la línea
			print("primer valor: ", first_value.to_int(), ", ultimo valor: ", last_value.to_int())
			tag.append(last_value.to_int()) # genero el tag en el index de la imagen
			leer -= 1
		
		file.close()
	else:
		print("el archivo no existe: ", file_path)
		
	
	'''etiquetamos '''
	for i in range(tag.size()):
		#var etiqueta = i 
		var array = etiqueta(tag[i])
		tag_data.append(array)
		print("Etiqueta: ", etiquetas[tag[i]], " Array: ", array)
	
	
	''' cargamos los archivos '''
	
	
	
	
	
	var size = 28  # Tamaño de las imagenes
	'''dir no lee en orden '''
	#dir_contents(dir_fashion , max) # cargamos los archivos primero , cargamos todo ?? 
	dir_count(dir_fashion, max)
	
	while !load_data :
		load_data = load_training_images(image_paths, size)

	print("Datos cargados: ", load_data.size())
	prints("tag cargados: " ,tag_data.size())
	prints(tag_data)
	
	inig = NNET.new(discriminator_structure, true)
	inig.set_loss_function(BNNET.LossFunctions.MAE)
	inig.activation_to_string(BNNET.ActivationFunctions.softmax)
	inig.use_Adam(0.01)
	#inig.use_Adadelta(0.5)
	#inig.set_loss_function(BNNET.LossFunctions.CCE)
	#inig.use_Rprop(0.9)
	inig.set_batch_size(1)
	#inig.use_NAG(0.9,0.1)
	#inig.use_Yogi(0.1)
	
	
	''' entrenamiento '''
	for li in range(20):
		for i in range(10):
			inig.load_data("res://data/etiqueta_img_train.bin")
			trai(1,inig,load_data,tag_data,i)

		
			inig.save_binary("res://data/etiqueta_img_train.bin")
	prints("guardamos")
	
	
	
	loss_tri(inig,load_data , tag_data)

	

''' solo cargamos y comprobamos '''











func trai(bucle ,inig,load_data , tag_data , idex):

	var count_index =  0
	
	
	for i in range(bucle):
		

		
		prints("entrenamiento en ronda 10 por : " , i , " de tantas : " , bucle )
		for j in range(1):
			for k in range(load_data.size()):
				var load_data2 = []
				var tag_data2 = []
				var array = etiqueta(idex)
				if tag_data[k] == array and array != null:
					count_index += 1
					load_data2.append(load_data[k])
					tag_data2.append(tag_data[k])
					prints(k)
				else:
					continue
			#inig.apply_gradients(0.01)
				for h in range(1):
					inig.set_input( load_data2[0])
					inig.set_target( tag_data2[0])
					inig.propagate_forward()
					#inig.apply_gradients(0.01)
					inig.propagate_backward()
					inig.train(load_data2 ,tag_data2)
					#inig.propagate_backward()
					if j % 2 == 0:
						loss_tri(inig,load_data2 , tag_data2)
						print("✨✨✨entrenando la red idex -:",idex," Iteración✨✨: ", j ," %  ✨✨porciento✨✨ de: ", k , " :✨✨ bucles✨✨")
	prints("total index " , count_index ," en modo :" ,idex )


func loss_tri(nn,load_data , tag_data):
	
	for i in range(1):
		var hit = 1
		var load_data2 = []
		var tag_data2 = []
		load_data2.append(load_data[0])
		tag_data2.append(tag_data[0])
		
		
		nn.set_input(load_data2[0]); nn.propagate_forward()
		
		print("Loss: ", nn.get_loss(load_data2, tag_data2) , "  imagen: " , hit,".png")
		
		
		var array = etiqueta(tag[hit])
		
		print("😊Etiqueta😊: ", etiquetas[tag[hit]], "🌟 Array 🌟: ", array)
		prints("🚀data out clasifi🚀: ",nn.get_output()," ⭐️data⭐️ :", tag_data[0])
		prints("⭐️num aleatorio⭐️: " ,hit )
	pass







func load_training_images(image_paths: Array, size: int) -> Array:
	var loaded_data = []
	
	for path in image_paths:
		var image = Image.new()
		var error = image.load(path)
		
		if error != OK:
			print("Failed to load image: ", path)
			continue
		
		if image.get_width() != size or image.get_height() != size:
			print("Image size does not match. Width: ", image.get_width(), " Height: ", image.get_height(), " Expected: ", size)
			continue
		
		var image_data = []
		for y in range(size):
			for x in range(size):
				var color = image.get_pixel(x, y)
				var value = color.r  # Asumiendo escala de grises, usar solo el canal rojo
				image_data.append(value)
		
		loaded_data.append(image_data)
	
	return loaded_data


func dir_count(path, count):
	for i in range(count):
		var file_name = str(i) + ".png"
		prints(file_name)
		image_paths.append(path + "/" + file_name)
	

func dir_contents(path , max):
	
	var dir = DirAccess.open(path)
	if dir:
		dir.list_dir_begin()
		var file_name = dir.get_next()
		while file_name != "" and max > 0:
			max -= 1
			if dir.current_is_dir():
				print("⭐️DIRECTORIO ENCONTRADO⭐️ : " + file_name)
			else:
				print("⭐ARCHIVO ENCONTRADO⭐️ : " + file_name)
				print("f⭐️EXTENCION DEL ARCHIVO⭐️: " + file_name.get_extension())

				if file_name.get_extension() == "png" or file_name.get_extension() == "jpg":
					prints("estencion gd")
					image_paths.append(path + "/" + file_name)
					prints( "⭐️DIRECTORIO⭐️  ",path + "/" + file_name)
			file_name = dir.get_next()
	else:
		print("⭐️ EDITOR FALLO : An error occurred when trying to access the path ⭐️")






func etiqueta(etiqueta: int) -> Array:
	var array_resultado = Array()
	for i in range(10):
		if i == etiqueta:
			array_resultado.append(0.9)
		else:
			array_resultado.append(0.01)
	return array_resultado
