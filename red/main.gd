extends Control





func _on_train_all_pressed() -> void:
	var MENU = load("res://train/save_algoritm.tscn").instantiate()
	add_child(MENU)
	prints("instancio escena")
	await get_tree().create_timer(1).timeout


func _on_cargar_modelo_pressed() -> void:
	var MENU = load("res://train/load.tscn").instantiate()
	add_child(MENU)
	prints("instancio escena")
	await get_tree().create_timer(1).timeout
